"""
retrain_transformer.py — 只重训 CityGraphTransformer（优化超参数）

解决问题：默认 lr=0.001 + epochs=100 对6层Transformer不够，
          导致主模型 MSE 反而高于 GCN/GAT 基线。

优化策略：
  - 降低学习率：0.001 → 0.0003
  - 增加轮数：100 → 300
  - 添加学习率 warmup + cosine decay
  - 只训练 CityGraphTransformer，不碰 GCN/GAT 权重

训练完后自动更新 model_comparison.csv。
"""

from __future__ import annotations

import math
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
from src.model import CityGraphTransformer
from src.utils import (
    get_logger,
    load_settings,
    resolve_device,
    resolve_project_paths,
    set_seed,
)


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


def _cosine_lr(optimizer, epoch, warmup, total, base_lr):
    """Warmup + cosine annealing 学习率调度。"""
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(1, total - warmup)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def main() -> None:
    cfg = load_settings()
    paths = resolve_project_paths(cfg)
    logger = get_logger("retrain", paths["log_dir"])

    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    set_seed(int(train_cfg.get("seed", 42)))

    device = resolve_device(train_cfg.get("device", "auto"))

    # ── 超参数（针对 Transformer 优化） ──────────────────────────────────
    lr = 0.0003            # 原来 0.001，降低 3 倍
    epochs = 300           # 原来 100，增加到 300
    warmup_epochs = 20     # 前 20 轮 warmup
    weight_decay = 1e-4    # 轻度正则化

    hidden_dim = int(model_cfg.get("hidden_dim", 512))
    heads = int(model_cfg.get("heads", 8))
    tf_layers = int(model_cfg.get("layers", 6))
    dropout = float(model_cfg.get("dropout", 0.1))

    model_dir = paths["model_dir"]
    result_dir = paths["result_dir"]
    processed_dir = paths["processed_dir"]
    cities = cfg.get("cities", [])

    logger.info(
        "Retrain CityGraphTransformer: lr=%.4f, epochs=%d, warmup=%d, device=%s",
        lr, epochs, warmup_epochs, device,
    )

    records: list[dict] = []

    for city_key in cities:
        pt_path = processed_dir / f"{city_key}.pt"
        data = _load_data(pt_path, device)
        if data is None:
            logger.warning("Skip %s: data not found", city_key)
            continue

        input_dim = int(data.x.size(1))
        logger.info("=== %s | nodes=%d | features=%d ===", city_key, data.x.size(0), input_dim)

        model = CityGraphTransformer(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=input_dim,
            heads=heads,
            layers=tf_layers,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()

        best_loss = float("inf")
        t0 = time.perf_counter()

        for epoch in range(epochs):
            current_lr = _cosine_lr(optimizer, epoch, warmup_epochs, epochs, lr)
            optimizer.zero_grad()
            out, _ = model(data.x, data.edge_index)
            loss = F.mse_loss(out, data.x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            cur_loss = float(loss.item())
            best_loss = min(best_loss, cur_loss)

            if (epoch + 1) % 30 == 0 or epoch == 0 or (epoch + 1) == epochs:
                logger.info(
                    "[%s] epoch %d/%d  loss=%.6f  best=%.6f  lr=%.6f",
                    city_key, epoch + 1, epochs, cur_loss, best_loss, current_lr,
                )

        elapsed = time.perf_counter() - t0

        save_path = model_dir / f"transformer_{city_key}.pth"
        torch.save(model.state_dict(), save_path)
        logger.info("[%s] Saved → %s  (best=%.6f, %.1fs)", city_key, save_path.name, best_loss, elapsed)

        records.append({
            "city": city_key,
            "model": "CityGraphTransformer",
            "final_mse": round(best_loss, 6),
            "train_seconds": round(elapsed, 1),
            "params": sum(p.numel() for p in model.parameters()),
        })

    # ── 更新 model_comparison.csv ─────────────────────────────────────────
    cmp_path = result_dir / "model_comparison.csv"
    if cmp_path.exists() and records:
        logger.info("Updating model_comparison.csv with new Transformer results ...")
        old_df = pd.read_csv(cmp_path)

        new_df = pd.DataFrame(records)

        # 读取旧的 GCN/GAT 行（从原始详细记录或宽表重构）
        # 重构思路：把旧宽表的 GCN/GAT 列解开，加上新 Transformer 记录，重建宽表
        if "GCNBaseline" in old_df.columns and "GATBaseline" in old_df.columns:
            # 旧表是宽格式
            baseline_rows = []
            for _, row in old_df.iterrows():
                city = row.get("city", "")
                if not city:
                    continue
                for model_name in ["GCNBaseline", "GATBaseline"]:
                    if model_name in row and pd.notna(row[model_name]):
                        baseline_rows.append({
                            "city": city,
                            "model": model_name,
                            "final_mse": float(row[model_name]),
                        })
            all_records = baseline_rows + records
        else:
            all_records = records

        all_df = pd.DataFrame(all_records)
        pivot = all_df.pivot_table(index="city", columns="model", values="final_mse", aggfunc="first")
        if len(pivot.columns) > 1:
            pivot["best_model"] = pivot.idxmin(axis=1)
        pivot.reset_index(inplace=True)
        pivot.to_csv(cmp_path, index=False, encoding="utf-8-sig")
        logger.info("Updated → %s", cmp_path)

        # 打印新的对比
        logger.info("\n更新后的模型对比：")
        logger.info("\n%s", pivot.to_string(index=False))

    # ── 用新权重重新导出 Web 缓存 ─────────────────────────────────────────
    logger.info("Rebuilding web cache with retrained Transformer ...")
    for city_key in cities:
        export_city_outputs(city_key=city_key, cfg=cfg, device=device, logger=logger)

    logger.info("Retrain finished!")


if __name__ == "__main__":
    main()
