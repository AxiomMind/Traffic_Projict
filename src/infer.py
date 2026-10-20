from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import osmnx as ox
import pandas as pd
import torch
from torch_geometric.data import Data

from src.model import CityGraphTransformer
from src.utils import get_logger, load_settings, resolve_device, resolve_project_paths


def _to_numpy(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _normalize_0_100(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value < 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value) * 100.0).astype(np.float32)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        for item in value:
            text = _safe_text(item)
            if text:
                return text
        return ""

    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return ""
    return text


def _extract_names(value: Any) -> list[str]:
    if value is None:
        return []

    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = text
        else:
            parsed = text

    names: list[str] = []
    if isinstance(parsed, (list, tuple, set)):
        for item in parsed:
            names.extend(_extract_names(item))
    else:
        text = _safe_text(parsed)
        if text:
            pieces = [p.strip() for p in text.replace("|", "/").replace(";", "/").split("/") if p.strip()]
            names.extend(pieces if pieces else [text])

    dedup: list[str] = []
    seen: set[str] = set()
    for item in names:
        if item not in seen:
            seen.add(item)
            dedup.append(item)
    return dedup


def _fallback_name(node_id: str) -> str:
    tail = node_id[-6:] if len(node_id) > 6 else node_id
    return f"路口-{tail}"


def _build_intersection_name_map(raw_graph_path: Path) -> dict[str, str]:
    if not raw_graph_path.exists():
        return {}

    graph = ox.load_graphml(raw_graph_path)
    name_map: dict[str, str] = {}

    for node, attrs in graph.nodes(data=True):
        node_id = str(node)
        candidate = ""
        for key in ("name", "ref", "junction", "street"):
            candidate = _safe_text(attrs.get(key))
            if candidate:
                break

        if not candidate:
            road_names: list[str] = []
            for _, _, edge_attr in graph.edges(node, data=True):
                road_names.extend(_extract_names(edge_attr.get("name")))
            if road_names:
                uniq = []
                seen = set()
                for name in road_names:
                    if name not in seen:
                        seen.add(name)
                        uniq.append(name)
                if len(uniq) >= 2:
                    candidate = f"{uniq[0]} × {uniq[1]}"
                else:
                    candidate = f"{uniq[0]} 路口"

        if not candidate:
            candidate = _fallback_name(node_id)
        name_map[node_id] = candidate

    return name_map


def _read_graph_data(pt_path: Path) -> Data:
    try:
        loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        loaded = torch.load(pt_path, map_location="cpu")

    if isinstance(loaded, Data):
        return loaded

    if isinstance(loaded, dict):
        data = Data(x=loaded["x"], edge_index=loaded["edge_index"])
        if "node_ids" in loaded:
            data.node_ids = loaded["node_ids"]
        if "lat" in loaded:
            data.lat = loaded["lat"]
        if "lon" in loaded:
            data.lon = loaded["lon"]
        return data

    raise TypeError(f"Unsupported data format: {type(loaded)}")


def _extract_node_columns(data: Data, node_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_ids = getattr(data, "node_ids", None)
    if node_ids is None:
        node_ids_np = np.asarray(list(range(node_count))).astype(str)
    else:
        node_ids_np = np.asarray(node_ids).astype(str)[:node_count]

    lat = getattr(data, "lat", None)
    lon = getattr(data, "lon", None)

    if lat is None or lon is None:
        lat = data.x[:, 1] if data.x.size(1) > 1 else torch.zeros(node_count)
        lon = data.x[:, 0] if data.x.size(1) > 0 else torch.zeros(node_count)

    lat_np = _to_numpy(lat).astype(np.float32)[:node_count]
    lon_np = _to_numpy(lon).astype(np.float32)[:node_count]
    return node_ids_np, lat_np, lon_np


def _build_report(data: Data, risk_scores: np.ndarray, name_map: dict[str, str]) -> pd.DataFrame:
    node_count = len(risk_scores)
    node_ids, lat, lon = _extract_node_columns(data, node_count)
    risk = np.asarray(risk_scores, dtype=np.float32)[:node_count]

    frame = pd.DataFrame(
        {
            "node_id": node_ids,
            "lat": lat,
            "lon": lon,
            "risk": risk,
        }
    )

    frame["intersection_name"] = frame["node_id"].map(name_map)
    frame["intersection_name"] = frame.apply(
        lambda row: row["intersection_name"] if _safe_text(row["intersection_name"]) else _fallback_name(row["node_id"]),
        axis=1,
    )
    frame["label_name"] = frame["intersection_name"].astype(str).str.slice(0, 18)
    frame["risk_level"] = pd.cut(
        frame["risk"],
        bins=[-1, 60, 85, 101],
        labels=["低风险 (III级)", "中风险 (II级)", "高风险 (I级)"],
    ).astype(str)
    frame["suggestion"] = np.where(
        frame["risk"] > 90,
        "建议增设排水泵站",
        np.where(frame["risk"] > 80, "建议优化信号灯", "纳入日常巡查"),
    ).astype(str)

    frame.sort_values("risk", ascending=False, inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def _build_road_segments(data: Data, risk_scores: np.ndarray, max_segments: int) -> pd.DataFrame:
    node_count = len(risk_scores)
    _, lat, lon = _extract_node_columns(data, node_count)
    edge_index = _to_numpy(data.edge_index)

    if edge_index.ndim != 2:
        return pd.DataFrame(columns=["source_lon", "source_lat", "target_lon", "target_lat", "edge_risk"])
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T
    if edge_index.shape[0] != 2:
        return pd.DataFrame(columns=["source_lon", "source_lat", "target_lon", "target_lat", "edge_risk"])

    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    valid = (src >= 0) & (src < node_count) & (dst >= 0) & (dst < node_count) & (src != dst)
    src = src[valid]
    dst = dst[valid]
    if src.size == 0:
        return pd.DataFrame(columns=["source_lon", "source_lat", "target_lon", "target_lat", "edge_risk"])

    u = np.minimum(src, dst)
    v = np.maximum(src, dst)
    keys = u.astype(np.int64) * np.int64(node_count) + v.astype(np.int64)
    _, unique_idx = np.unique(keys, return_index=True)

    src = src[unique_idx]
    dst = dst[unique_idx]
    edge_risk = ((risk_scores[src] + risk_scores[dst]) / 2.0).astype(np.float32)

    roads = pd.DataFrame(
        {
            "source_lon": lon[src],
            "source_lat": lat[src],
            "target_lon": lon[dst],
            "target_lat": lat[dst],
            "edge_risk": edge_risk,
        }
    ).dropna()

    roads.sort_values("edge_risk", ascending=False, inplace=True)
    roads.reset_index(drop=True, inplace=True)

    if max_segments > 0 and len(roads) > max_segments:
        head_size = max(1, int(max_segments * 0.45))
        head = roads.head(head_size)
        tail = roads.iloc[head_size:]
        rest_size = max_segments - len(head)
        if rest_size > 0 and len(tail) > rest_size:
            tail = tail.sample(n=rest_size, random_state=42)
        roads = pd.concat([head, tail], ignore_index=True)
        roads.sort_values("edge_risk", ascending=False, inplace=True)
        roads.reset_index(drop=True, inplace=True)

    return roads


def infer_one_city(
    pt_path: Path,
    model_path: Path,
    model_cfg: dict[str, Any],
    device: torch.device,
    road_cache_max_segments: int,
    name_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = _read_graph_data(pt_path)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    model = CityGraphTransformer(
        in_channels=int(x.size(1)),
        hidden_channels=int(model_cfg.get("hidden_dim", 512)),
        out_channels=int(x.size(1)),
        heads=int(model_cfg.get("heads", 8)),
        layers=int(model_cfg.get("layers", 6)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)

    state = torch.load(model_path, map_location=device)
    cleaned_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned_state, strict=False)
    model.eval()

    with torch.no_grad():
        out, _ = model(x, edge_index)
        residual = torch.norm(out - x, dim=1).detach().cpu().numpy()

    risk_scores = _normalize_0_100(residual)
    report = _build_report(data, risk_scores, name_map=name_map)
    roads = _build_road_segments(data, risk_scores, max_segments=road_cache_max_segments)
    return report, roads


def export_city_outputs(
    city_key: str,
    cfg: dict[str, Any],
    device: torch.device,
    logger,
) -> bool:
    paths = resolve_project_paths(cfg)
    web_cfg = cfg.get("web", {})
    road_cache_max_segments = int(web_cfg.get("road_cache_max_segments", 180000))

    pt_path = paths["processed_dir"] / f"{city_key}.pt"
    model_path = paths["model_dir"] / f"transformer_{city_key}.pth"
    raw_graph_path = paths["raw_dir"] / f"{city_key}.graphml"
    csv_path = paths["result_dir"] / f"{city_key}.csv"
    node_parquet_path = paths["web_cache_dir"] / f"{city_key}.parquet"
    road_parquet_path = paths["web_cache_dir"] / f"{city_key}_roads.parquet"

    if not pt_path.exists():
        logger.warning("Skip %s: processed file not found", city_key)
        return False
    if not model_path.exists():
        logger.warning("Skip %s: model file not found", city_key)
        return False

    paths["result_dir"].mkdir(parents=True, exist_ok=True)
    paths["web_cache_dir"].mkdir(parents=True, exist_ok=True)

    try:
        name_map = _build_intersection_name_map(raw_graph_path)
    except Exception as exc:
        logger.warning("Name extraction failed for %s: %s", city_key, exc)
        name_map = {}

    report, roads = infer_one_city(
        pt_path=pt_path,
        model_path=model_path,
        model_cfg=cfg.get("model", {}),
        device=device,
        road_cache_max_segments=road_cache_max_segments,
        name_map=name_map,
    )

    report.to_csv(csv_path, index=False, encoding="utf-8-sig")
    report.to_parquet(node_parquet_path, index=False)
    roads.to_parquet(road_parquet_path, index=False)
    logger.info("Exported %s (nodes=%d, roads=%d)", city_key, len(report), len(roads))
    return True


def main() -> None:
    cfg = load_settings()
    paths = resolve_project_paths(cfg)
    logger = get_logger("infer", paths["log_dir"])
    device = resolve_device(cfg.get("train", {}).get("device", "auto"))
    logger.info("Build risk outputs (device=%s)", device)

    success_count = 0
    for city_key in cfg.get("cities", []):
        if export_city_outputs(city_key, cfg, device, logger):
            success_count += 1

    logger.info("Inference export finished: %d cities", success_count)


if __name__ == "__main__":
    main()
