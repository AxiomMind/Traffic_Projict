"""
model.py — 城市图神经网络模型集合

包含三个模型：
  1. CityGraphTransformer  （主模型，图变换器 + 残差）
  2. GCNBaseline           （GCN 基线）
  3. GATBaseline           （GAT 基线）
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, TransformerConv


# ══════════════════════════════════════════════════════════════════════════════
#  主模型：CityGraphTransformer
# ══════════════════════════════════════════════════════════════════════════════
class CityGraphTransformer(nn.Module):
    """
    基于 TransformerConv 的图自编码器（Graph Autoencoder）。

    使用无监督重建范式：训练时最小化 MSE(x_hat, x)，
    推理时以残差范数作为节点异常风险分。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 8,
        layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
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
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.decoder = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.embedding(x))

        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index)
            x = F.relu(bn(x))
            # 残差连接：形状一致时才加
            if x.shape == residual.shape:
                x = x + residual
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.decoder(x)
        return out, x  # (重建输出, 嵌入)


# ══════════════════════════════════════════════════════════════════════════════
#  基线模型 1：GCNBaseline
# ══════════════════════════════════════════════════════════════════════════════
class GCNBaseline(nn.Module):
    """
    简单 GCN 自编码器基线。

    GCN 没有注意力机制，所有邻居权重相同（度归一化平均）。
    作为 CityGraphTransformer 的对比基线。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(layers)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(layers)]
        )
        self.decoder = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.embedding(x))

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = F.relu(bn(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.decoder(x)
        return out, x


# ══════════════════════════════════════════════════════════════════════════════
#  基线模型 2：GATBaseline
# ══════════════════════════════════════════════════════════════════════════════
class GATBaseline(nn.Module):
    """
    GAT 自编码器基线（使用注意力但无 Transformer 位置编码和残差）。

    作为比 GCN 更强但比 CityGraphTransformer 更弱的中间基线。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_channels % heads == 0, "hidden_channels must be divisible by heads"
        self.embedding = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList(
            [
                GATConv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(layers)]
        )
        self.decoder = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.embedding(x))

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = F.relu(bn(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.decoder(x)
        return out, x
