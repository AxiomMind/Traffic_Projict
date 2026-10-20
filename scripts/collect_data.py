from __future__ import annotations

import os
import sys
import time

import osmnx as ox

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils import get_logger, load_settings, resolve_project_paths


def _query_candidates(city_key: str, display_name: str) -> list[str]:
    candidates = [f"{city_key}, China", city_key]
    if display_name and display_name not in candidates:
        candidates.append(display_name)
    return candidates


def collect_data() -> None:
    cfg = load_settings()
    paths = resolve_project_paths(cfg)

    raw_dir = paths["raw_dir"]
    log_dir = paths["log_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("collect_data", log_dir)

    ox.settings.log_console = False
    ox.settings.use_cache = True
    ox.settings.timeout = 300

    cities = cfg.get("cities", [])
    name_map = cfg.get("localization", {})
    logger.info("Start collecting graph data for %d cities", len(cities))

    for city_key in cities:
        file_path = raw_dir / f"{city_key}.graphml"
        if file_path.exists():
            logger.info("Skip %s: graph already exists", city_key)
            continue

        display_name = name_map.get(city_key, city_key)
        done = False
        for query in _query_candidates(city_key, display_name):
            start = time.perf_counter()
            try:
                graph = ox.graph_from_place(query, network_type="drive", simplify=True)
                ox.save_graphml(graph, file_path)
                elapsed = time.perf_counter() - start
                logger.info(
                    "Collected %s (query=%s, nodes=%d, edges=%d, %.2fs)",
                    city_key,
                    query,
                    graph.number_of_nodes(),
                    graph.number_of_edges(),
                    elapsed,
                )
                done = True
                break
            except Exception as exc:
                logger.warning("Query failed for %s using %s: %s", city_key, query, exc)

        if not done:
            logger.error("Failed to collect city %s", city_key)

    logger.info("Data collection finished")


if __name__ == "__main__":
    collect_data()
