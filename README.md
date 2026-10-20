# Traffic_Projict · 城市交通韧性治理智能分析平台

本项目面向现代城市交通管理中的核心痛点：在极端天气（如暴雨积水）、交通早晚高峰、大型活动等高压场景下，城市路网的脆弱点往往难以被及时发现，且缺乏科学的预警与治理依据。为此，本项目开发了一套“城市交通韧性治理智能分析平台”。系统基于开源真实路网数据，利用自主设计的图神经网络（Graph Transformer）构建路口级风险评估模型。平台不仅能够输出精确的节点风险指数，还能通过拓扑关系计算道路联动风险，并自动生成“建议增设排水泵站”、“建议优化信号灯”或“纳入日常巡查”等具有极强实操性的分级治理建议。结合 WebGL 高性能可视化看板，系统为城市交通管理部门提供从研判到决策的一站式支持。

## 功能概览

| 模块 | 说明 |
| --- | --- |
| 数据采集 | 基于 `osmnx` 从 OpenStreetMap 拉取城市机动车路网，导出为 `graphml` |
| 特征工程 | 从真实路网提取 12 维节点/边特征（拓扑、中心性、车道数、限速、道路等级等） |
| 风险建模 | 自主设计的 `CityGraphTransformer`（残差 + BatchNorm + 多头注意力），与 GCN / GAT 基线对比 |
| 拓扑验证 | 风险分与 PageRank、介数中心性、度中心性的 Spearman 相关分析 |
| 消融实验 | 6 种模型变体（去残差、去 BN、层数、头数）对比 |
| 名称增强 | 高德逆地理编码把 `node_id` 映射为真实路口名称（需免费 API Key） |
| 可视化看板 | `streamlit` + `pydeck`（WebGL）渲染热力图、高风险路口表、分级治理建议 |

## 目录结构

```
Traffic_Projict/
├── pipeline.py                 # 一键全流程运行器（collect → preprocess → train → enrich → validate → ablation）
├── run_web.py                  # Streamlit + pydeck 可视化看板
├── requirements.txt
├── config/
│   └── settings.yaml           # 全局配置：数据路径、城市列表、中英文映射、超参数
├── src/
│   ├── model.py                # CityGraphTransformer / GCN / GAT 定义
│   ├── dataset.py              # 图数据加载
│   ├── infer.py                # 推理与风险分级、治理建议生成
│   └── utils.py                # 配置加载、日志、随机种子、设备选择
├── scripts/
│   ├── collect_data.py         # 下载路网 graphml
│   ├── preprocess.py           # 12 维特征提取 → .pt
│   ├── train.py                # 三模型对比训练
│   ├── retrain_transformer.py  # 单独重训 Transformer
│   ├── enrich_names.py         # 高德逆地理编码补全路口名
│   ├── validate.py             # 拓扑有效性验证
│   ├── ablation.py             # 消融实验
│   ├── build_web_cache.py      # 生成看板 parquet 缓存
│   └── build_road_cache.py     # 生成道路段缓存
├── data/
│   ├── raw/                    # 原始 graphml（不入库，见下）
│   └── processed/              # 预处理产物
├── results/                    # 结果表与缓存
│   ├── model_comparison.csv
│   ├── ablation_summary.csv
│   ├── validation_summary.csv
│   └── name_cache/             # 高德路口名缓存
├── weights/                    # 模型权重（不入库）
└── logs/
```

## 快速开始

```bash
pip install -r requirements.txt
```

**完整流程**（首次运行，需联网下载路网，耗时较长）：

```bash
python pipeline.py --amap-key YOUR_AMAP_KEY
```

**已有路网数据时跳过下载**：

```bash
python pipeline.py --amap-key YOUR_AMAP_KEY --skip-collect
```

**只跑部分步骤**：

```bash
python pipeline.py --only preprocess train            # 无高德 Key 也能跑
python pipeline.py --only validate ablation           # 验证与消融
python pipeline.py --amap-key YOUR_KEY --only enrich  # 补全路口名称
```

可用步骤名：`collect | preprocess | train | enrich | validate | ablation`。

高德 Key 可在 [lbs.amap.com](https://lbs.amap.com) 免费申请（服务平台选「Web 服务」），也可通过环境变量 `AMAP_API_KEY` 传入。

**启动看板**：

```bash
streamlit run run_web.py
```

## 数据说明（重要）

原始路网 `data/raw/*.graphml` **未纳入版本库**：单城市 30–170MB，8 城合计约 543MB，其中 `Chongqing.graphml` 达 169MB，已超过 GitHub 单文件 100MB 的硬性上限。

需要数据时重新生成即可（`osmnx` 从 OSM 拉取，城市列表见 `config/settings.yaml`）：

```bash
python pipeline.py --only collect
```

同理，逐城市结果明细 `results/{City}.csv`（每个 4–18MB）、看板缓存 `results/web_cache/*.parquet`、模型权重 `weights/*.pth` 均按 `.gitignore` 排除，可由流水线重新生成；仓库中只保留 5 个小型汇总表与路口名称缓存。

## 结果文件

| 文件 | 内容 |
| --- | --- |
| `results/model_comparison.csv` | 8 个城市下 Transformer / GCN / GAT 的 MSE 对比与最优模型 |
| `results/ablation_summary.csv` | 6 种模型变体的均/最小/最大 MSE 及相对 full 模型的百分比变化 |
| `results/ablation_detail.csv` | 消融实验逐城市明细 |
| `results/validation_summary.csv` | 风险分与 PageRank / 介数 / 度的 Spearman 相关、Top-20 重合度（Jaccard） |
| `results/validation_report.csv` | 拓扑验证逐城市明细 |
| `results/name_cache/*.json` | 高德逆地理编码得到的高风险路口名称缓存 |

## 说明

- 环境：Python 3.10+，`torch` + `torch-geometric`；CPU 可运行，GPU 更快。
- `config/settings.yaml` 中的 `paths`、`cities`、`localization` 为必需项；其余超参缺省时使用代码内置默认值（默认值已在配置文件末尾注释列出）。
- 本项目数据来源为 OpenStreetMap（ODbL 协议）与高德地图开放平台，请遵守相应使用条款。
