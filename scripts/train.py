"""
train.py — 多模型对比训练 + 推理导出

训练三种模型：
  - CityGraphTransformer  （主模型）
  - GCNBaseline           （基线 1）
  - GATBaseline           （基线 2）

输出：
  - weights/transformer_{city}.pth     ← 主模型权重
  - weights/gcn_{city}.pth            ← GCN 权重
  - weights/gat_{city}.pth            ← GAT 权重
  - results/model_comparison.csv      ← 对比表（MSE + 训练时长）
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.infer import export_city_outputs
from src.model import CityGraphTransformer, GCNBaseline, GATBaseline
from src.utils import (
    get_logger,
    load_settings,
    resolve_device,
    resolve_project_paths,
    set_seed,
)


# ── 统一读取数据（兼容 dict 和 Data 两种格式） ──────────────────────────────
def _load_data(pt_path: Path, device: torch.device) -> Data | None:
    if not pt_path.exists():
        return None
    try:
        loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        loaded = torch.load(pt_path, map_location="cpu")

    if isinstance(loaded, Data):
        data = loaded
    elif isinstance(loaded, dict):
        data = Data(x=loaded["x"], edge_index=loaded["edge_index"])
        for key in ("node_ids", "lat", "lon"):
            if key in loaded:
                setattr(data, key, loaded[key])
    else:
        return None

    return data.to(device)


# ── 单城市单模型训练 ─────────────────────────────────────────────────────────
def _train_model(
    model: torch.nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    city_key: str,
    model_name: str,
    logger,
) -> float:
    """训练并返回最终 MSE loss。"""
    model.train()
    last_loss = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index)
        loss = F.mse_loss(out, data.x)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
        if (epoch + 1) % 20 == 0 or epoch == 0 or (epoch + 1) == epochs:
            logger.info(
                "[%s|%s] epoch %d/%d  loss=%.6f",
                city_key, model_name, epoch + 1, epochs, last_loss,
            )
    return last_loss


# ── 主流程 ───────────────────────────────────────────────────────────────────
def train_pipeline() -> None:
    cfg = load_settings()
    paths = resolve_project_paths(cfg)
    logger = get_logger("train", paths["log_dir"])

    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    set_seed(int(train_cfg.get("seed", 42)))

    device = resolve_device(train_cfg.get("device", "auto"))
    lr = float(train_cfg.get("lr", 1e-3))
    epochs = int(train_cfg.get("epochs", 100))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    model_dir = paths["model_dir"]
    result_dir = paths["result_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    cities = cfg.get("cities", [])
    processed_dir = paths["processed_dir"]

    hidden_dim = int(model_cfg.get("hidden_dim", 512))
    heads = int(model_cfg.get("heads", 8))
    tf_layers = int(model_cfg.get("layers", 6))
    dropout = float(model_cfg.get("dropout", 0.1))

    logger.info(
        "Multi-model training: device=%s, epochs=%d, lr=%.5f, cities=%d",
        device, epochs, lr, len(cities),
    )

    # 记录对比结果
    records: list[dict] = []

    for city_key in cities:
        pt_path = processed_dir / f"{city_key}.pt"
        data = _load_data(pt_path, device)
        if data is None:
            logger.warning("Skip %s: processed file not found", city_key)
            continue

        input_dim = int(data.x.size(1))
        logger.info("=== %s | nodes=%d | feature_dim=%d ===", city_key, data.x.size(0), input_dim)

        # ── 模型配置字典 ──────────────────────────────────────────────────
        model_configs = {
            "CityGraphTransformer": CityGraphTransformer(
                in_channels=input_dim,
                hidden_channels=hidden_dim,
                out_channels=input_dim,
                heads=heads,
                layers=tf_layers,
                dropout=dropout,
            ),
            "GCNBaseline": GCNBaseline(
                in_channels=input_dim,
                hidden_channels=hidden_dim,
                out_channels=input_dim,
                layers=3,
                dropout=dropout,
            ),
            "GATBaseline": GATBaseline(
                in_channels=input_dim,
                hidden_channels=hidden_dim,
                out_channels=input_dim,
                heads=4,
                layers=3,
                dropout=dropout,
            ),
        }

        # 文件名前缀映射
        save_prefix = {
            "CityGraphTransformer": "transformer",
            "GCNBaseline": "gcn",
            "GATBaseline": "gat",
        }

        for model_name, model in model_configs.items():
            model = model.to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            t0 = time.perf_counter()
            final_loss = _train_model(
                model, data, optimizer, epochs, city_key, model_name, logger
            )
            elapsed = time.perf_counter() - t0

            prefix = save_prefix[model_name]
            save_path = model_dir / f"{prefix}_{city_key}.pth"
            torch.save(model.state_dict(), save_path)
            logger.info("[%s|%s] Saved → %s  (loss=%.6f, %.1fs)", city_key, model_name, save_path.name, final_loss, elapsed)

            records.append({
                "city": city_key,
                "model": model_name,
                "final_mse": round(final_loss, 6),
                "train_seconds": round(elapsed, 1),
                "params": sum(p.numel() for p in model.parameters()),
            })

    # ── 保存对比表 ────────────────────────────────────────────────────────
    if records:
        df = pd.DataFrame(records)
        # 宽表：每城市一行，各模型 MSE 为列
        pivot = df.pivot_table(index="city", columns="model", values="final_mse", aggfunc="first")
        pivot["best_model"] = pivot.idxmin(axis=1)
        pivot.reset_index(inplace=True)

        cmp_path = result_dir / "model_comparison.csv"
        pivot.to_csv(cmp_path, index=False, encoding="utf-8-sig")
        logger.info("Model comparison table saved → %s", cmp_path)

        # 打印汇总
        logger.info("\n%s", df.groupby("model")["final_mse"].agg(["mean", "min", "max"]).to_string())

    # ── 用主模型（CityGraphTransformer）跑推理并更新 Web 缓存 ────────────
    logger.info("Building web cache with CityGraphTransformer ...")
    for city_key in cities:
        export_city_outputs(city_key=city_key, cfg=cfg, device=device, logger=logger)

    logger.info("Training pipeline finished.")


if __name__ == "__main__":
    train_pipeline()
