from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:
    import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    return PROJECT_ROOT


def load_settings(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "settings.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_paths(cfg: dict[str, Any], project_root: Path | None = None) -> dict[str, Path]:
    root = project_root or PROJECT_ROOT
    paths_cfg = cfg.get("paths", {})
    resolved: dict[str, Path] = {}

    for key, value in paths_cfg.items():
        path_value = Path(value)
        resolved[key] = path_value if path_value.is_absolute() else (root / path_value).resolve()

    return resolved


def ensure_project_dirs(cfg: dict[str, Any], project_root: Path | None = None) -> dict[str, Path]:
    resolved = resolve_project_paths(cfg, project_root=project_root)
    for path in resolved.values():
        path.mkdir(parents=True, exist_ok=True)
    return resolved


def set_seed(seed: int = 42) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_cfg: str = "auto") -> "torch.device":
    import torch

    value = (device_cfg or "auto").lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def get_logger(name: str, log_dir: str | Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    target_dir = Path(log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(target_dir / "system_run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def as_float_array(values: list[Any], default: float = 0.0) -> np.ndarray:
    output = []
    for value in values:
        try:
            output.append(float(value))
        except (TypeError, ValueError):
            output.append(default)
    return np.asarray(output, dtype=np.float32)
