from __future__ import annotations

from pathlib import Path
from typing import Any

import osmnx as ox
import torch
from torch_geometric.data import Data

from src.utils import get_project_root, resolve_project_paths


class TrafficDataset:
    def __init__(self, config: dict[str, Any], project_root: Path | None = None):
        self.cfg = config
        self.project_root = project_root or get_project_root()
        self.paths = resolve_project_paths(config, project_root=self.project_root)

    def load_city_data(self, city_key: str) -> tuple[Any, Data] | None:
        graph_path = self.paths["raw_dir"] / f"{city_key}.graphml"
        pt_path = self.paths["processed_dir"] / f"{city_key}.pt"

        if not graph_path.exists() or not pt_path.exists():
            return None

        graph = ox.load_graphml(graph_path)
        if graph.is_directed():
            graph = graph.to_undirected()

        try:
            data = torch.load(pt_path, weights_only=False)
        except Exception:
            data = torch.load(pt_path)

        if isinstance(data, dict):
            data = Data(x=data["x"], edge_index=data["edge_index"])

        return graph, data
