"""
pipeline.py — 城市交通韧性平台 · 一键全流程运行器

用法：
  # 完整流程（首次运行）
  python pipeline.py --amap-key YOUR_KEY

  # 跳过数据下载（已有 data/raw/*.graphml 时）
  python pipeline.py --amap-key YOUR_KEY --skip-collect

  # 只运行部分步骤
  python pipeline.py --amap-key YOUR_KEY --only preprocess train enrich

  # 不做名称增强（无 Key 时）
  python pipeline.py --skip-enrich

可用步骤名：collect | preprocess | train | enrich | validate | ablation
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ── 步骤定义（有序） ─────────────────────────────────────────────────────────
STEPS = [
    "collect",     # 下载城市路网 graphml
    "preprocess",  # 提取 12 维真实特征
    "train",       # 三模型对比训练 + 建 Web 缓存
    "enrich",      # 高德逆地理编码增强路口名（需 API Key）
    "validate",    # 拓扑有效性验证（Spearman 相关）
    "ablation",    # 消融实验（6 种模型变体）
]

STEP_SCRIPTS = {
    "collect":    "scripts/collect_data.py",
    "preprocess": "scripts/preprocess.py",
    "train":      "scripts/train.py",
    "enrich":     "scripts/enrich_names.py",   # 需额外传 --api-key
    "validate":   "scripts/validate.py",
    "ablation":   "scripts/ablation.py",
}

STEP_LABELS = {
    "collect":    "数据下载（osmnx 路网）",
    "preprocess": "特征提取（12维真实路网特征）",
    "train":      "多模型训练（Transformer / GCN / GAT）",
    "enrich":     "路口名称增强（高德逆地理编码）",
    "validate":   "拓扑验证（Spearman 相关分析）",
    "ablation":   "消融实验（6种模型变体对比）",
}


def _run_step(step: str, extra_args: list[str], dry_run: bool) -> bool:
    """运行单个步骤，返回是否成功。"""
    script = PROJECT_ROOT / STEP_SCRIPTS[step]
    cmd = [sys.executable, str(script)] + extra_args

    label = STEP_LABELS[step]
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"▶  步骤：{label}")
    print(f"   命令：{' '.join(cmd)}")
    print(sep)

    if dry_run:
        print("   [DRY-RUN] 跳过实际执行")
        return True

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        print(f"\n✅  {label} 完成（{elapsed:.0f}s）")
        return True
    else:
        print(f"\n❌  {label} 失败（returncode={result.returncode}）")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="城市交通韧性平台 · 全流程运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--amap-key",
        default="",
        help="高德 Web服务 API Key（逆地理编码，免费注册申请）",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=STEPS,
        metavar="STEP",
        help=f"只运行指定步骤，可选：{' | '.join(STEPS)}",
    )
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="跳过 collect 步骤（已有 data/raw/*.graphml 时推荐加此选项）",
    )
    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        help="跳过 enrich 步骤（无高德 Key 时）",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="跳过消融实验（节省时间，ablation 独立于主流程）",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1000,
        help="高德名称增强：每城市处理的高风险路口数（默认1000）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印命令，不实际执行",
    )
    args = parser.parse_args()

    # ── 确定要跑哪些步骤 ──────────────────────────────────────────────────
    if args.only:
        run_steps = [s for s in STEPS if s in args.only]
    else:
        run_steps = list(STEPS)
        if args.skip_collect:
            run_steps = [s for s in run_steps if s != "collect"]
        if args.skip_enrich or not args.amap_key:
            run_steps = [s for s in run_steps if s != "enrich"]
            if not args.skip_enrich and not args.amap_key:
                print("⚠️  未提供 --amap-key，跳过路口名称增强步骤。")
        if args.skip_ablation:
            run_steps = [s for s in run_steps if s != "ablation"]

    print("=" * 60)
    print("  城市交通韧性平台 · 全流程运行器")
    print("=" * 60)
    print(f"  计划执行步骤：{' → '.join(run_steps)}")
    print("=" * 60)

    # ── 依次执行 ─────────────────────────────────────────────────────────
    failed_steps = []
    for step in run_steps:
        extra: list[str] = []
        if step == "enrich":
            if not args.amap_key:
                print("⚠️  跳过 enrich：未提供 --amap-key")
                continue
            extra = ["--api-key", args.amap_key, "--top-n", str(args.top_n)]

        ok = _run_step(step, extra, args.dry_run)
        if not ok:
            failed_steps.append(step)
            print(f"\n⛔  步骤 [{step}] 失败，后续步骤不中断，继续执行...")

    # ── 汇总 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failed_steps:
        print(f"⚠️  完成（有 {len(failed_steps)} 个步骤失败：{failed_steps}）")
    else:
        print("🎉  全部步骤执行成功！")

    print("\n下一步 → 启动看板：")
    print("  streamlit run run_web.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
