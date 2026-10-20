"""
preprocess.py — 城市路网结构特征提取

将 osmnx 路网图转换为 12 维真实拓扑特征向量：
  0  degree_norm          — 节点总度（归一化）
  1  in_degree_norm       — 入度（归一化）
  2  out_degree_norm      — 出度（归一化）
  3  pagerank_norm        — PageRank 中心性（归一化）
  4  edge_len_log_norm    — 相邻边平均长度（log 归一化）
  5  road_rank_norm       — 最佳邻接道路等级 0=主干/1=末端（归一化）
  6  is_major_junction    — 是否主干道路口（binary）
  7  is_secondary_junction— 是否次干道路口（binary）
  8  road_type_diversity  — 路口道路类型多样性（归一化）
  9  neighbor_degree_norm — 邻居平均度（归一化）
  10 lat_norm             — 纬度（城市内归一化）
  11 lon_norm             — 经度（城市内归一化）
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils import get_logger, load_settings, resolve_project_paths

# ── 道路等级映射：数值越小 = 越重要 ──────────────────────────────────────
ROAD_RANK: dict[str, int] = {
    "motorway": 0, "motorway_link": 0,
    "trunk": 1,    "trunk_link": 1,
    "primary": 2,  "primary_link": 2,
    "secondary": 3, "secondary_link": 3,
    "tertiary": 4,  "tertiary_link": 4,
    "unclassified": 5,
    "residential": 5,
    "living_street": 6,
    "service": 6,
    "pedestrian": 7,
    "cycleway": 7,
    "path": 7,
    "footway": 7,
    "steps": 7,
    "track": 7,
}
MAX_RANK = 7  # 归一化分母


def _road_rank(highway_val) -> float:
    """获取单个或列表 highway 标签的最低（最重要）等级数值。"""
    if highway_val is None:
        return 5.0
    if isinstance(highway_val, list):
        return float(min(ROAD_RANK.get(str(h).strip(), 5) for h in highway_val))
    return float(ROAD_RANK.get(str(highway_val).strip(), 5))


def _norm(arr: np.ndarray) -> np.ndarray:
    """min-max 归一化到 [0, 1]，全零时返回零向量。"""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _extract_features(
    graph: nx.MultiDiGraph,
) -> tuple[list[str], np.ndarray, np.ndarray, torch.Tensor]:
    """
    从 osmnx 图中提取节点级结构特征。

    Returns
    -------
    node_ids   : 节点 id 字符串列表
    lat_np     : float32 array, shape (N,)
    lon_np     : float32 array, shape (N,)
    feature_tensor : float32 tensor, shape (N, 12)
    """
    node_ids = list(graph.nodes())
    n = len(node_ids)
    node_idx = {nid: i for i, nid in enumerate(node_ids)}
    node_attrs = graph.nodes(data=True)
    attr_dict = {nid: attrs for nid, attrs in node_attrs}

    # ── 1. 度特征 ────────────────────────────────────────────────────────
    degree = np.array([graph.degree(nid) for nid in node_ids], dtype=np.float32)
    in_degree = np.array([graph.in_degree(nid) for nid in node_ids], dtype=np.float32)
    out_degree = np.array([graph.out_degree(nid) for nid in node_ids], dtype=np.float32)

    # ── 2. PageRank（快，O(k*(n+m))，k 次迭代） ─────────────────────────
    try:
        pr_dict = nx.pagerank(graph, alpha=0.85, max_iter=150, tol=1e-6)
        pagerank = np.array([pr_dict.get(nid, 0.0) for nid in node_ids], dtype=np.float32)
    except Exception:
        pagerank = degree.copy()

    # ── 3. 边级特征（按节点聚合） ────────────────────────────────────────
    avg_edge_len = np.full(n, 100.0, dtype=np.float32)   # 默认 100m
    road_rank_min = np.full(n, 5.0, dtype=np.float32)    # 默认 residential
    road_type_set_size = np.zeros(n, dtype=np.float32)

    for i, nid in enumerate(node_ids):
        lengths: list[float] = []
        ranks: list[float] = []
        rtypes: set[int] = set()

        # 出边
        for _, _, edata in graph.edges(nid, data=True):
            lng = edata.get("length")
            if lng is not None:
                try:
                    lengths.append(float(lng))
                except (TypeError, ValueError):
                    pass
            rk = _road_rank(edata.get("highway"))
            ranks.append(rk)
            rtypes.add(int(rk))

        # 入边（补充长度信息）
        for _, _, edata in graph.in_edges(nid, data=True):
            lng = edata.get("length")
            if lng is not None:
                try:
                    lengths.append(float(lng))
                except (TypeError, ValueError):
                    pass

        if lengths:
            avg_edge_len[i] = float(np.mean(lengths))
        if ranks:
            road_rank_min[i] = float(min(ranks))
        road_type_set_size[i] = float(len(rtypes))

    is_major = (road_rank_min <= 2).astype(np.float32)       # motorway/trunk/primary
    is_secondary = ((road_rank_min >= 3) & (road_rank_min <= 4)).astype(np.float32)

    # ── 4. 邻居平均度 ────────────────────────────────────────────────────
    neighbor_deg = np.zeros(n, dtype=np.float32)
    for i, nid in enumerate(node_ids):
        neigh = list(graph.neighbors(nid))
        if neigh:
            neighbor_deg[i] = float(np.mean([graph.degree(nb) for nb in neigh]))

    # ── 5. 空间坐标 ──────────────────────────────────────────────────────
    lon_np = np.array(
        [attr_dict[nid].get("x", attr_dict[nid].get("lon", 0.0)) for nid in node_ids],
        dtype=np.float32,
    )
    lat_np = np.array(
        [attr_dict[nid].get("y", attr_dict[nid].get("lat", 0.0)) for nid in node_ids],
        dtype=np.float32,
    )

    # ── 6. 组合 12 维特征矩阵 ─────────────────────────────────────────────
    # 道路等级：小→重要，归一化后 0=最重要，1=最次要
    road_rank_norm = (road_rank_min / MAX_RANK).astype(np.float32)

    features = np.stack(
        [
            _norm(degree),                        # 0
            _norm(in_degree),                     # 1
            _norm(out_degree),                    # 2
            _norm(pagerank),                      # 3
            _norm(np.log1p(avg_edge_len)),        # 4  log scale
            road_rank_norm,                       # 5  already [0,1]
            is_major,                             # 6  binary
            is_secondary,                         # 7  binary
            _norm(road_type_set_size),            # 8
            _norm(neighbor_deg),                  # 9
            _norm(lat_np),                        # 10
            _norm(lon_np),                        # 11
        ],
        axis=1,
    )  # (N, 12)

    return node_ids, lat_np, lon_np, torch.from_numpy(features)


def _build_edge_index(graph: nx.MultiDiGraph, node_idx: dict) -> torch.Tensor:
    """手动构建 edge_index，避免 from_networkx 的类型转换问题。"""
    src_list, dst_list = [], []
    for u, v in graph.edges():
        if u in node_idx and v in node_idx:
            src_list.append(node_idx[u])
            dst_list.append(node_idx[v])
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src_list, dst_list], dtype=torch.long)


def preprocess() -> None:
    cfg = load_settings()
    paths = resolve_project_paths(cfg)
    logger = get_logger("preprocess", paths["log_dir"])

    raw_dir = paths["raw_dir"]
    processed_dir = paths["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    cities = cfg.get("cities", [])
    logger.info("Start preprocessing %d cities → 12-dim real road-network features", len(cities))

    for city_key in cities:
        graph_path = raw_dir / f"{city_key}.graphml"
        save_path = processed_dir / f"{city_key}.pt"

        if not graph_path.exists():
            logger.warning("Skip %s: graphml not found (%s)", city_key, graph_path)
            continue

        t0 = time.perf_counter()
        logger.info("[%s] Loading graph ...", city_key)
        graph: nx.MultiDiGraph = ox.load_graphml(graph_path)
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        logger.info("[%s] Graph loaded: %d nodes, %d edges", city_key, n_nodes, n_edges)

        logger.info("[%s] Extracting 12-dim features (PageRank + topology) ...", city_key)
        node_ids, lat_np, lon_np, feature_tensor = _extract_features(graph)

        node_idx = {nid: i for i, nid in enumerate(node_ids)}
        edge_index = _build_edge_index(graph, node_idx)

        # 存储为标准 dict（兼容旧版 torch.load）
        payload = {
            "x": feature_tensor,                                    # (N, 12)
            "edge_index": edge_index,                               # (2, E)
            "node_ids": [str(nid) for nid in node_ids],
            "lat": torch.from_numpy(lat_np),
            "lon": torch.from_numpy(lon_np),
            "feature_names": [
                "degree", "in_degree", "out_degree", "pagerank",
                "avg_edge_len", "road_rank", "is_major", "is_secondary",
                "road_type_diversity", "neighbor_degree", "lat", "lon",
            ],
        }
        torch.save(payload, save_path)

        elapsed = time.perf_counter() - t0
        logger.info(
            "[%s] Saved → %s  (nodes=%d, features=%s, edges=%d, %.1fs)",
            city_key, save_path.name, len(node_ids), list(feature_tensor.shape), edge_index.shape[1], elapsed,
        )

    logger.info("Preprocess finished. Feature dim = 12 (real road-network topology)")


if __name__ == "__main__":
    preprocess()
