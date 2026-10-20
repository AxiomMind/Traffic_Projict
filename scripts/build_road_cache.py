from __future__ import annotations

import ast
import os
import sys
from typing import Any

import osmnx as ox
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils import get_logger, load_settings, resolve_project_paths


def _node_risk_map(node_frame: pd.DataFrame) -> dict[str, float]:
    frame = node_frame.copy()
    frame["node_id"] = frame["node_id"].astype(str)
    frame["risk"] = pd.to_numeric(frame["risk"], errors="coerce").fillna(0.0)
    return dict(zip(frame["node_id"], frame["risk"]))


def _safe_text(value: Any) -> str:
    if value is None:
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
            parts = [p.strip() for p in text.replace("|", "/").replace(";", "/").split("/") if p.strip()]
            names.extend(parts if parts else [text])

    uniq: list[str] = []
    seen = set()
    for item in names:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def _fallback_name(node_id: str) -> str:
    tail = node_id[-6:] if len(node_id) > 6 else node_id
    return f"路口-{tail}"


def _build_intersection_name_map(graph) -> dict[str, str]:
    name_map: dict[str, str] = {}
    for node, attrs in graph.nodes(data=True):
        node_id = str(node)
        label = ""
        for key in ("name", "ref", "junction", "street"):
            label = _safe_text(attrs.get(key))
            if label:
                break
        if not label:
            road_names: list[str] = []
            for _, _, edge_attr in graph.edges(node, data=True):
                road_names.extend(_extract_names(edge_attr.get("name")))
            if road_names:
                unique = []
                seen = set()
                for name in road_names:
                    if name not in seen:
                        seen.add(name)
                        unique.append(name)
                label = f"{unique[0]} × {unique[1]}" if len(unique) >= 2 else f"{unique[0]} 路口"
        if not label:
            label = _fallback_name(node_id)
        name_map[node_id] = label
    return name_map


def _enrich_node_names(node_frame: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    output = node_frame.copy()
    output["node_id"] = output["node_id"].astype(str)
    output["intersection_name"] = output["node_id"].map(name_map)
    output["intersection_name"] = output.apply(
        lambda row: row["intersection_name"] if _safe_text(row["intersection_name"]) else _fallback_name(row["node_id"]),
        axis=1,
    )
    output["label_name"] = output["intersection_name"].astype(str).str.slice(0, 18)
    return output


def _build_roads_from_graph(graph, risk_map: dict[str, float], max_segments: int) -> pd.DataFrame:
    seen: set[tuple[str, str]] = set()
    rows: list[tuple[float, float, float, float, float]] = []

    for u, v in graph.edges():
        u_str, v_str = str(u), str(v)
        pair = tuple(sorted((u_str, v_str)))
        if pair in seen:
            continue
        seen.add(pair)

        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        u_lon = u_data.get("x", u_data.get("lon"))
        u_lat = u_data.get("y", u_data.get("lat"))
        v_lon = v_data.get("x", v_data.get("lon"))
        v_lat = v_data.get("y", v_data.get("lat"))
        if None in (u_lon, u_lat, v_lon, v_lat):
            continue

        u_risk = float(risk_map.get(u_str, 0.0))
        v_risk = float(risk_map.get(v_str, 0.0))
        rows.append((float(u_lon), float(u_lat), float(v_lon), float(v_lat), (u_risk + v_risk) / 2.0))

    roads = pd.DataFrame(rows, columns=["source_lon", "source_lat", "target_lon", "target_lat", "edge_risk"])
    roads.dropna(inplace=True)
    roads.sort_values("edge_risk", ascending=False, inplace=True)
    roads.reset_index(drop=True, inplace=True)

    if max_segments > 0 and len(roads) > max_segments:
        head_size = max(1, int(max_segments * 0.45))
        head = roads.head(head_size)
        tail = roads.iloc[head_size:]
        keep_tail = max_segments - len(head)
        if keep_tail > 0 and len(tail) > keep_tail:
            tail = tail.sample(n=keep_tail, random_state=42)
        roads = pd.concat([head, tail], ignore_index=True)
        roads.sort_values("edge_risk", ascending=False, inplace=True)
        roads.reset_index(drop=True, inplace=True)

    return roads


def main() -> None:
    cfg = load_settings()
    paths = resolve_project_paths(cfg)
    logger = get_logger("build_road_cache", paths["log_dir"])

    max_segments = int(cfg.get("web", {}).get("road_cache_max_segments", 180000))
    raw_dir = paths["raw_dir"]
    web_cache_dir = paths["web_cache_dir"]
    web_cache_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for city in cfg.get("cities", []):
        node_path = web_cache_dir / f"{city}.parquet"
        road_path = web_cache_dir / f"{city}_roads.parquet"
        graph_path = raw_dir / f"{city}.graphml"

        if not node_path.exists():
            logger.warning("Skip %s: node parquet not found", city)
            continue
        if not graph_path.exists():
            logger.warning("Skip %s: graphml not found", city)
            continue

        graph = ox.load_graphml(graph_path)
        if graph.is_directed():
            graph = graph.to_undirected()

        node_frame = pd.read_parquet(node_path)
        if "node_id" not in node_frame.columns or "risk" not in node_frame.columns:
            logger.warning("Skip %s: node parquet missing node_id/risk", city)
            continue

        risk_map = _node_risk_map(node_frame)
        name_map = _build_intersection_name_map(graph)
        enriched_nodes = _enrich_node_names(node_frame, name_map=name_map)
        enriched_nodes.to_parquet(node_path, index=False)

        roads = _build_roads_from_graph(graph, risk_map=risk_map, max_segments=max_segments)
        roads.to_parquet(road_path, index=False)
        logger.info("Saved %s and refreshed names for %s (%d roads)", road_path, city, len(roads))
        count += 1

    logger.info("Road cache finished for %d cities", count)


if __name__ == "__main__":
    main()
