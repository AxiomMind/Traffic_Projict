"""
enrich_names.py — 高德逆地理编码路口中文名称增强

功能：
  对每个城市的高风险路口（按风险排序取 top-N），调用高德逆地理编码 API，
  获取真实中文路口名（如"长安街与建国门大街交叉口"），
  并写回 results/web_cache/{city}.parquet，实时刷新看板显示。

API 申请（免费）：
  https://lbs.amap.com → 控制台 → 创建应用 → 添加 Key（服务平台选 "Web服务"）
  个人认证用户：5,000 次/天
  企业认证用户：300,000 次/天（建议申请）

用法：
  # 方式1：命令行传入
  python scripts/enrich_names.py --api-key YOUR_AMAP_KEY

  # 方式2：环境变量（推荐在服务器上设置）
  export AMAP_API_KEY=YOUR_AMAP_KEY
  python scripts/enrich_names.py

  # 可选参数
  python scripts/enrich_names.py --api-key KEY --top-n 2000 --qps 10
  (top-n: 每城市处理的路口数，默认1000；qps: 请求速率，默认10)

注意：
  - 运行前需先完成 train.py（web_cache/*.parquet 必须存在）
  - 脚本会自动跳过已有中文名的路口（增量更新）
  - 结果缓存到 results/name_cache/{city}_amap_names.json 避免重复请求
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils import get_logger, load_settings, resolve_project_paths


# ── 高德逆地理编码 API 常量 ─────────────────────────────────────────────────
REGEO_URL = "https://restapi.amap.com/v3/geocode/regeo"


def _call_regeo(lon: float, lat: float, api_key: str, radius: int = 50) -> dict | None:
    """
    调用高德逆地理编码 API，返回原始 JSON 或 None（失败时）。

    radius: 搜索半径（米），对于路口一般50m足够
    """
    params = {
        "key": api_key,
        "location": f"{lon:.6f},{lat:.6f}",
        "radius": radius,
        "extensions": "base",
        "roadlevel": "0",   # 0=全级别道路，1=主干道
    }
    try:
        resp = requests.get(REGEO_URL, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        if str(data.get("status")) == "1":
            return data
        return None
    except Exception:
        return None


def _parse_intersection_name(regeo_result: dict) -> str:
    """
    从高德逆地理编码结果中提取路口名称。

    优先级：
      1. roadinters（交叉路口）→ "X路与Y路交叉口"
      2. roads（附近道路）→ "X路路口"
      3. formatted_address（完整地址，取最后段）→ 兜底
    """
    try:
        component = regeo_result["regeocode"]["addressComponent"]

        # 1. 交叉路口（最理想）
        roadinters = component.get("roadinters", [])
        if isinstance(roadinters, list) and roadinters:
            first = roadinters[0]
            fname = str(first.get("firstname", "")).strip()
            sname = str(first.get("secondname", "")).strip()
            if fname and sname:
                return f"{fname}与{sname}交叉口"
            elif fname:
                return f"{fname}路口"

        # 2. 附近道路
        roads = component.get("roads", [])
        if isinstance(roads, list) and roads:
            road_names = [str(r.get("name", "")).strip() for r in roads[:2] if r.get("name")]
            if len(road_names) >= 2:
                return f"{road_names[0]}与{road_names[1]}交叉口"
            elif road_names:
                return f"{road_names[0]}路口"

        # 3. 格式化地址兜底
        addr = str(regeo_result["regeocode"].get("formatted_address", "")).strip()
        if addr:
            # 取最后 2 段（通常是街道级）
            parts = [p for p in addr.replace("市", "市|").replace("区", "区|")
                      .replace("街道", "街道|").split("|") if p.strip()]
            if parts:
                return parts[-1].strip()

    except (KeyError, TypeError, IndexError):
        pass

    return ""


def _load_name_cache(cache_path: Path) -> dict[str, str]:
    """加载本地 JSON 名称缓存（node_id → name）。"""
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_name_cache(cache_path: Path, cache: dict[str, str]) -> None:
    """保存名称缓存到 JSON。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _is_fallback_name(name: str) -> bool:
    """判断是否为 OSM 兜底名（路口-XXXXXX 格式），需要高德替换。"""
    if not name or not isinstance(name, str):
        return True
    stripped = name.strip()
    if not stripped:
        return True
    if stripped.startswith("路口-"):
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  单城市处理
# ══════════════════════════════════════════════════════════════════════════════
def _enrich_city(
    city_key: str,
    parquet_path: Path,
    cache_path: Path,
    api_key: str,
    top_n: int,
    qps: float,
    logger,
) -> int:
    """
    对单个城市的 top-N 高风险路口做名称增强。

    Returns: 本次实际调用 API 的次数
    """
    if not parquet_path.exists():
        logger.warning("[%s] parquet 不存在，跳过：%s", city_key, parquet_path)
        return 0

    df = pd.read_parquet(parquet_path)
    if not {"node_id", "lat", "lon", "risk"}.issubset(df.columns):
        logger.warning("[%s] 缺少必要字段，跳过", city_key)
        return 0

    df["node_id"] = df["node_id"].astype(str)
    df["risk"] = pd.to_numeric(df["risk"], errors="coerce").fillna(0)

    # 确保有 intersection_name 列
    if "intersection_name" not in df.columns:
        df["intersection_name"] = ""
    df["intersection_name"] = df["intersection_name"].fillna("").astype(str)

    # 加载本地缓存
    name_cache = _load_name_cache(cache_path)
    logger.info("[%s] 本地缓存已有 %d 条名称", city_key, len(name_cache))

    # 选取 top-N 路口（优先处理高风险 + 当前为兜底名的）
    candidates = df.sort_values("risk", ascending=False).head(top_n)

    interval = 1.0 / qps  # 请求间隔（秒）
    api_calls = 0
    updated = 0

    for _, row in candidates.iterrows():
        node_id = str(row["node_id"])
        current_name = str(row["intersection_name"])

        # 已有高质量名称（不是兜底名），跳过
        if not _is_fallback_name(current_name):
            # 但如果本地缓存有更好的名字，也更新
            if node_id in name_cache:
                df.loc[df["node_id"] == node_id, "intersection_name"] = name_cache[node_id]
            continue

        # 已在缓存里
        if node_id in name_cache:
            df.loc[df["node_id"] == node_id, "intersection_name"] = name_cache[node_id]
            updated += 1
            continue

        # 调用高德 API
        lon = float(row["lon"])
        lat = float(row["lat"])
        result = _call_regeo(lon, lat, api_key)
        api_calls += 1

        if result:
            name = _parse_intersection_name(result)
            if name:
                df.loc[df["node_id"] == node_id, "intersection_name"] = name
                name_cache[node_id] = name
                updated += 1
            else:
                # API 成功但无路口信息，记录"未命名"防止重查
                name_cache[node_id] = current_name
        else:
            logger.warning("[%s] API 调用失败 node=%s (%.6f, %.6f)", city_key, node_id, lon, lat)

        time.sleep(interval)

        # 每 200 次保存一次缓存（防中断丢失）
        if api_calls % 200 == 0:
            _save_name_cache(cache_path, name_cache)
            logger.info("[%s] 已调用 %d 次 API，更新 %d 个路口名", city_key, api_calls, updated)

    # 同步更新 label_name（用于地图标注，截断18字）
    df["intersection_name"] = df["intersection_name"].astype(str)
    df["label_name"] = df["intersection_name"].str.slice(0, 18)

    # 保存缓存和 parquet
    _save_name_cache(cache_path, name_cache)
    df.to_parquet(parquet_path, index=False)
    logger.info(
        "[%s] 完成！API调用=%d，更新路口名=%d，总缓存=%d",
        city_key, api_calls, updated, len(name_cache),
    )
    return api_calls


# ══════════════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="高德逆地理编码路口名称增强",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("AMAP_API_KEY", ""),
        help="高德 Web服务 API Key（或设环境变量 AMAP_API_KEY）",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1000,
        help="每城市处理的最高风险路口数（默认1000，8城市共8000次API调用）",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=10.0,
        help="每秒最大请求数（默认10，免费版安全上限）",
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="只处理指定城市（默认全部，例如 --cities Beijing Shanghai）",
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "\n错误：未提供高德 API Key！\n"
            "请到 https://lbs.amap.com 免费申请 Web服务类型的 Key，然后：\n"
            "  python scripts/enrich_names.py --api-key YOUR_KEY\n"
            "或设置环境变量：\n"
            "  export AMAP_API_KEY=YOUR_KEY   (Linux/Mac)\n"
            "  set AMAP_API_KEY=YOUR_KEY      (Windows)\n"
        )
        sys.exit(1)

    cfg = load_settings()
    paths = resolve_project_paths(cfg)
    logger = get_logger("enrich_names", paths["log_dir"])

    cache_dir = paths["web_cache_dir"]
    name_cache_dir = paths["result_dir"] / "name_cache"
    name_cache_dir.mkdir(parents=True, exist_ok=True)

    cities = args.cities or cfg.get("cities", [])
    logger.info(
        "开始高德路口名称增强：%d 城市，每城市 top-%d，QPS=%.1f",
        len(cities), args.top_n, args.qps,
    )
    logger.info("预计 API 调用上限：%d 次（约 %.0f 分钟）",
                len(cities) * args.top_n,
                len(cities) * args.top_n / args.qps / 60)

    # 先测试 API Key 是否有效
    logger.info("验证 API Key 有效性...")
    test = _call_regeo(116.3975, 39.9085, args.api_key)  # 天安门
    if test is None:
        logger.error("API Key 无效或网络异常，请检查后重试！")
        sys.exit(1)
    test_name = _parse_intersection_name(test)
    logger.info("API Key 验证通过！测试路口名：%s", test_name or "（无命名）")

    total_api_calls = 0
    for city_key in cities:
        parquet_path = cache_dir / f"{city_key}.parquet"
        cache_path = name_cache_dir / f"{city_key}_amap_names.json"

        logger.info("─── 处理城市：%s ───", city_key)
        calls = _enrich_city(
            city_key=city_key,
            parquet_path=parquet_path,
            cache_path=cache_path,
            api_key=args.api_key,
            top_n=args.top_n,
            qps=args.qps,
            logger=logger,
        )
        total_api_calls += calls

    logger.info(
        "\n全部完成！总 API 调用 %d 次。\n"
        "刷新 Streamlit 看板后，路口名称将显示真实中文交叉路口名。\n"
        "（看板侧边栏点击「刷新缓存」按钮即生效）",
        total_api_calls,
    )


if __name__ == "__main__":
    main()
