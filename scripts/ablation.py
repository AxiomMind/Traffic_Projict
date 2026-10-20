"""
ablation.py — CityGraphTransformer 消融实验

比较以下变体（在第一个可用城市上训练，快速验证）：

  full          — 完整模型（6层, 8头, 残差, BatchNorm）
  no_residual   — 去掉残差连接
  3_layers      — 层数从 6 减至 3
  4_heads       — 注意力头数从 8 减至 4
  1_head        — 退化到单头注意力
  no_batchnorm  — 去掉 BatchNorm

输出：
  results/ablation_results.csv
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils import get_logger, load_settings, resolve_device, resolve_project_paths, set_seed


# ══════════════════════════════════════════════════════════════════════════════
#  消融用模型（参数化 CityGraphTransformer 变体）
# ══════════════════════════════════════════════════════════════════════════════
class _GraphTransformerVariant(nn.Module):
    """可配置的 CityGraphTransformer，支持消融开关。"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 8,
        layers: int = 6,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.dropout = dropout

        self.embedding = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(layers):
            self.convs.append(
                TransformerConv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    dropout=dropout,
                )
            )
            if use_batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            else:
                self.bns.append(nn.Identity())

        self.decoder = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.embedding(x))
        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index)
            x = F.relu(bn(x))
            if self.use_residual and x.shape == residual.shape:
                x = x + residual
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.decoder(x), x


# ══════════════════════════════════════════════════════════════════════════════
#  消融变体定义
# ══════════════════════════════════════════════════════════════════════════════
def _build_variants(input_dim: int, hidden_dim: int, dropout: float) -> dict[str, nn.Module]:
    common = dict(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=input_dim, dropout=dropout)
    return {
        "full (6L-8H-residual-BN)": _GraphTransformerVariant(**common, heads=8, layers=6, use_residual=True,  use_batchnorm=True),
        "no_residual":              _GraphTransformerVariant(**common, heads=8, layers=6, use_residual=False, use_batchnorm=True),
        "3_layers":                 _GraphTransformerVariant(**common, heads=8, layers=3, use_residual=True,  use_batchnorm=True),
        "4_heads":                  _GraphTransformerVariant(**common, heads=4, layers=6, use_residual=True,  use_batchnorm=True),
        "1_head":                   _GraphTransformerVariant(**common, heads=1, layers=6, use_residual=True,  use_batchnorm=True),
        "no_batchnorm":             _GraphTransformerVariant(**common, heads=8, layers=6, use_residual=True,  use_batchnorm=False),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  数据加载工具
# ══════════════════════════════════════════════════════════════════════════════
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
    else:
        return None
    return data.to(device)


# ══════════════════════════════════════════════════════════════════════════════
#  训练单个变体
# ══════════════════════════════════════════════════════════════════════════════
def _train(model: nn.Module, data: Data, lr: float, epochs: int, logger, tag: str) -> tuple[float, float]:
    """返回 (best_loss, train_seconds)。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    best_loss = float("inf")
    t0 = time.perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index)
        loss = F.mse_loss(out, data.x)
        loss.backward()
        optimizer.step()
        best_loss = min(best_loss, float(loss.item()))
        if (epoch + 1) % 25 == 0 or epoch == 0:
            logger.info("  [%s] epoch %d/%d  loss=%.6f", tag, epoch + 1, epochs, float(loss.item()))
    return best_loss, time.perf_counter() - t0


# ══════════════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════════════
def run_ablation() -> None:
    cfg = load_settings()
    paths = resolve_project_paths(cfg)
    logger = get_logger("ablation", paths["log_dir"])

    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    set_seed(int(train_cfg.get("seed", 42)))

    device = resolve_device(train_cfg.get("device", "auto"))
    lr = float(train_cfg.get("lr", 1e-3))
    # 消融实验轮数可设置更少加速，需要时可调
    epochs = int(train_cfg.get("ablation_epochs", train_cfg.get("epochs", 100)))
    hidden_dim = int(model_cfg.get("hidden_dim", 512))
    dropout = float(model_cfg.get("dropout", 0.1))

    cities = cfg.get("cities", [])
    processed_dir = paths["processed_dir"]
    result_dir = paths["result_dir"]
    result_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    # 全部城市都跑，与训练保持一致
    for city_key in cities:
        pt_path = processed_dir / f"{city_key}.pt"
        data = _load_data(pt_path, device)
        if data is None:
            logger.warning("Skip %s: processed data not found", city_key)
            continue

        input_dim = int(data.x.size(1))
        logger.info("=== Ablation: %s | nodes=%d | feature_dim=%d ===", city_key, data.x.size(0), input_dim)
        variants = _build_variants(input_dim, hidden_dim, dropout)

        for variant_name, model in variants.items():
            model = model.to(device)
            n_params = sum(p.numel() for p in model.parameters())
            logger.info("  Variant: %-35s params=%d", variant_name, n_params)

            best_loss, elapsed = _train(model, data, lr, epochs, logger, variant_name)
            logger.info("  → best_loss=%.6f  (%.1fs)", best_loss, elapsed)

            records.append({
                "city": city_key,
                "variant": variant_name,
                "best_mse": round(best_loss, 6),
                "params": n_params,
                "train_seconds": round(elapsed, 1),
            })

    if not records:
        logger.error("No ablation records: check that preprocess data exists.")
        return

    df = pd.DataFrame(records)

    # 各变体在所有城市上的平均 MSE
    summary = (
        df.groupby("variant")["best_mse"]
        .agg(mean_mse="mean", min_mse="min", max_mse="max")
        .sort_values("mean_mse")
        .reset_index()
    )
    summary["vs_full_pct"] = (
        (summary["mean_mse"] / summary.loc[summary["variant"].str.startswith("full"), "mean_mse"].values[0] - 1) * 100
    ).round(2)

    out_detail = result_dir / "ablation_detail.csv"
    out_summary = result_dir / "ablation_summary.csv"
    df.to_csv(out_detail, index=False, encoding="utf-8-sig")
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")

    logger.info("\n消融实验汇总（按平均MSE升序）：\n%s", summary.to_string(index=False))
    logger.info("详细结果 → %s", out_detail)
    logger.info("汇总结果 → %s", out_summary)


if __name__ == "__main__":
    run_ablation()
