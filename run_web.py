from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from src.utils import load_settings, resolve_project_paths


st.set_page_config(
    page_title="交通韧性治理看板",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700&family=Noto+Sans+SC:wght@400;500;700&display=swap');

:root {
  --bg-a: #f1f6ff;
  --bg-b: #eef3f8;
  --ink: #142033;
  --muted: #617086;
  --card: rgba(255, 255, 255, 0.9);
  --edge: rgba(22, 42, 77, 0.15);
}

html, body, [class*="css"] {
  font-family: "Noto Sans SC", "Manrope", sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(circle at 12% 8%, rgba(35,166,213,0.11), transparent 34%),
    radial-gradient(circle at 88% 2%, rgba(245,159,67,0.14), transparent 28%),
    linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 72%);
}

[data-testid="stSidebar"] {
  background: rgba(255, 255, 255, 0.94);
  border-right: 1px solid var(--edge);
}

div[data-testid="stMetric"] {
  border: 1px solid var(--edge);
  border-radius: 14px;
  background: var(--card);
  padding: 12px 14px;
  box-shadow: 0 10px 25px rgba(25, 54, 97, 0.08);
}

#MainMenu, footer {
  visibility: hidden;
}

.title {
  margin: 4px 0 4px 0;
  font-size: 1.65rem;
  font-weight: 700;
  letter-spacing: 0.2px;
}

.subtitle {
  margin: 0 0 10px 0;
  color: var(--muted);
  font-size: 0.95rem;
}

.panel-title {
  margin: 8px 0 6px 0;
  font-size: 1.05rem;
  font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)


CFG = load_settings()
PATHS = resolve_project_paths(CFG)
WEB_CFG = CFG.get("web", {})
CITY_LIST = CFG.get("cities", [])
LOCALIZATION = CFG.get("localization", {})


def _city_label(city_key: str) -> str:
    return LOCALIZATION.get(city_key, city_key)


def _available_cities() -> list[str]:
    cache_dir = PATHS["web_cache_dir"]
    if not cache_dir.exists():
        return []
    return [city for city in CITY_LIST if (cache_dir / f"{city}.parquet").exists()]


def _format_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _fallback_name(node_id: str) -> str:
    tail = node_id[-6:] if len(node_id) > 6 else node_id
    return f"路口-{tail}"


@st.cache_data(show_spinner=False)
def _read_node_cache(cache_file: str) -> pd.DataFrame:
    frame = pd.read_parquet(cache_file)
    required = ["node_id", "lat", "lon", "risk"]
    missing = [key for key in required if key not in frame.columns]
    if missing:
        raise ValueError(f"点位缓存缺少字段: {missing}")

    frame["node_id"] = frame["node_id"].astype(str)
    frame["risk"] = pd.to_numeric(frame["risk"], errors="coerce").fillna(0.0)

    if "intersection_name" not in frame.columns:
        frame["intersection_name"] = frame["node_id"].map(_fallback_name)
    frame["intersection_name"] = frame["intersection_name"].fillna("")
    frame["intersection_name"] = frame.apply(
        lambda row: row["intersection_name"] if str(row["intersection_name"]).strip() else _fallback_name(row["node_id"]),
        axis=1,
    )

    if "label_name" not in frame.columns:
        frame["label_name"] = frame["intersection_name"].astype(str).str.slice(0, 18)
    else:
        frame["label_name"] = frame["label_name"].astype(str).str.slice(0, 18)

    if "risk_level" not in frame.columns:
        frame["risk_level"] = pd.cut(
            frame["risk"],
            bins=[-1, 60, 85, 101],
            labels=["低风险 (III级)", "中风险 (II级)", "高风险 (I级)"],
        )
    if "suggestion" not in frame.columns:
        frame["suggestion"] = np.where(
            frame["risk"] > 90,
            "建议增设排水泵站",
            np.where(frame["risk"] > 80, "建议优化信号灯", "纳入日常巡查"),
        )

    keep_cols = [
        "node_id",
        "intersection_name",
        "label_name",
        "lat",
        "lon",
        "risk",
        "risk_level",
        "suggestion",
    ]
    frame = frame[keep_cols].dropna(subset=["lat", "lon"]).copy()
    frame["risk_level"] = frame["risk_level"].astype(str)
    frame.sort_values("risk", ascending=False, inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


@st.cache_data(show_spinner=False)
def _read_road_cache(cache_file: str) -> pd.DataFrame:
    frame = pd.read_parquet(cache_file)
    required = ["source_lon", "source_lat", "target_lon", "target_lat", "edge_risk"]
    missing = [key for key in required if key not in frame.columns]
    if missing:
        raise ValueError(f"道路缓存缺少字段: {missing}")

    frame = frame[required].dropna().copy()
    frame["edge_risk"] = pd.to_numeric(frame["edge_risk"], errors="coerce").fillna(0.0)
    frame.sort_values("edge_risk", ascending=False, inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def _sample_frame(frame: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def _apply_preset(preset: str) -> None:
    if preset == "综合治理":
        st.session_state["show_roads"] = True
        st.session_state["show_nodes"] = True
        st.session_state["show_hot"] = True
        st.session_state["show_heat"] = True
        st.session_state["show_labels"] = True
    elif preset == "路口巡检":
        st.session_state["show_roads"] = True
        st.session_state["show_nodes"] = True
        st.session_state["show_hot"] = True
        st.session_state["show_heat"] = False
        st.session_state["show_labels"] = True
    else:
        st.session_state["show_roads"] = False
        st.session_state["show_nodes"] = False
        st.session_state["show_hot"] = True
        st.session_state["show_heat"] = True
        st.session_state["show_labels"] = False


def _init_layer_state() -> None:
    if "show_roads" not in st.session_state:
        _apply_preset("路口巡检")


def _resolve_map_style(style_name: str) -> str:
    style_map = {
        "标注底图": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "彩色底图": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
        "深色底图": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    }
    return style_map.get(style_name, style_map["标注底图"])


def _build_map_layers(
    node_frame: pd.DataFrame,
    road_frame: pd.DataFrame,
    heat_max_points: int,
    top_n: int,
    node_max_points: int,
    road_max_segments: int,
    label_max_points: int,
) -> list[pdk.Layer]:
    layers: list[pdk.Layer] = []

    if st.session_state.get("show_roads", True) and not road_frame.empty:
        roads = _sample_frame(road_frame, road_max_segments)
        roads["line_width"] = np.clip(0.8 + roads["edge_risk"] / 55.0, 0.8, 2.6)
        roads["line_color"] = roads["edge_risk"].map(
            lambda x: [58, max(70, int(190 - x)), min(255, int(145 + x * 0.8)), 95]
        )
        layers.append(
            pdk.Layer(
                "LineLayer",
                data=roads,
                get_source_position=["source_lon", "source_lat"],
                get_target_position=["target_lon", "target_lat"],
                get_width="line_width",
                width_scale=1,
                width_min_pixels=1,
                get_color="line_color",
                pickable=False,
            )
        )

    if st.session_state.get("show_nodes", True):
        points = _sample_frame(node_frame[["lon", "lat"]], node_max_points)
        points["radius"] = 26
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=points,
                get_position=["lon", "lat"],
                get_fill_color=[67, 94, 128, 95],
                get_radius="radius",
                radius_min_pixels=1,
                pickable=False,
            )
        )

    if st.session_state.get("show_heat", True):
        heat_data = _sample_frame(node_frame[["lon", "lat", "risk"]], heat_max_points)
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=heat_data,
                get_position=["lon", "lat"],
                get_weight="risk",
                radiusPixels=54,
                intensity=1.1,
                threshold=0.07,
            )
        )

    if st.session_state.get("show_hot", True):
        hot = node_frame.head(top_n).copy()
        hot["radius"] = np.clip(130 + hot["risk"] * 2.2, 130, 360)
        hot["risk_text"] = hot["risk"].map(lambda x: f"{x:.1f}")
        hot["color"] = hot["risk"].map(
            lambda x: [219, 76, 64, 230] if x >= 90 else ([245, 159, 67, 218] if x >= 80 else [72, 135, 245, 195])
        )
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=hot,
                get_position=["lon", "lat"],
                get_fill_color="color",
                get_radius="radius",
                radius_min_pixels=2,
                stroked=True,
                get_line_color=[255, 255, 255, 145],
                line_width_min_pixels=1,
                pickable=True,
            )
        )

    if st.session_state.get("show_labels", True):
        labels = node_frame.head(label_max_points).copy()
        labels["label_name"] = labels["label_name"].astype(str)
        layers.append(
            pdk.Layer(
                "TextLayer",
                data=labels,
                get_position=["lon", "lat"],
                get_text="label_name",
                get_size=13,
                size_units="pixels",
                size_min_pixels=10,
                get_color=[28, 45, 72, 220],
                get_text_anchor="'start'",
                get_alignment_baseline="'center'",
                get_pixel_offset=[10, -2],
                pickable=False,
            )
        )

    return layers


def _render_governance_panel(frame: pd.DataFrame) -> None:
    st.markdown('<div class="panel-title">治理分析</div>', unsafe_allow_html=True)

    risk_count = (
        frame["risk_level"]
        .value_counts()
        .reindex(["高风险 (I级)", "中风险 (II级)", "低风险 (III级)"], fill_value=0)
        .reset_index()
    )
    risk_count.columns = ["level", "count"]

    pie = go.Figure(
        data=[go.Pie(labels=risk_count["level"], values=risk_count["count"], hole=0.56, marker=dict(colors=["#db4c40", "#f59f43", "#4c84e0"]))]
    )
    pie.update_layout(height=235, margin=dict(l=4, r=4, t=4, b=4), showlegend=True, legend_orientation="h")
    st.plotly_chart(pie, use_container_width=True)

    suggestion_count = frame["suggestion"].value_counts().head(3).reset_index()
    suggestion_count.columns = ["suggestion", "count"]
    bar = go.Figure(
        data=[
            go.Bar(
                y=suggestion_count["suggestion"],
                x=suggestion_count["count"],
                orientation="h",
                marker_color=["#23a6d5", "#f59f43", "#db4c40"][: len(suggestion_count)],
            )
        ]
    )
    bar.update_layout(height=220, margin=dict(l=6, r=6, t=6, b=6))
    st.plotly_chart(bar, use_container_width=True)

    top_risk = frame[["intersection_name", "node_id", "risk", "suggestion"]].head(10).copy()
    top_risk.rename(columns={"intersection_name": "路口名称", "node_id": "路口编号", "risk": "风险指数", "suggestion": "建议"}, inplace=True)
    st.dataframe(top_risk, use_container_width=True, height=260)


def _render_map_view(city_key: str, node_frame: pd.DataFrame, road_frame: pd.DataFrame, update_text: str) -> None:
    st.markdown(f'<p class="title">{_city_label(city_key)} 路网风险态势</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">更新于 {update_text} · 支持路口名称标注与图层切换</p>', unsafe_allow_html=True)

    preset_cols = st.columns([1, 1, 1, 4])
    if preset_cols[0].button("综合治理视图", use_container_width=True):
        _apply_preset("综合治理")
    if preset_cols[1].button("路口巡检视图", use_container_width=True):
        _apply_preset("路口巡检")
    if preset_cols[2].button("热力研判视图", use_container_width=True):
        _apply_preset("热力研判")

    layer_cols = st.columns(5)
    layer_cols[0].checkbox("道路网络", key="show_roads")
    layer_cols[1].checkbox("路口点位", key="show_nodes")
    layer_cols[2].checkbox("高风险路口", key="show_hot")
    layer_cols[3].checkbox("风险热力", key="show_heat")
    layer_cols[4].checkbox("路口名称", key="show_labels")

    control_cols = st.columns(6)
    heat_max_points = int(
        control_cols[0].slider("热力采样", min_value=1000, max_value=30000, value=int(WEB_CFG.get("heatmap_max_points", 10000)), step=500)
    )
    top_n = int(
        control_cols[1].slider("高风险路口数量", min_value=50, max_value=5000, value=int(WEB_CFG.get("top_n_points", 300)), step=50)
    )
    node_max_points = int(
        control_cols[2].slider("路口点位上限", min_value=5000, max_value=60000, value=int(WEB_CFG.get("node_max_points", 20000)), step=1000)
    )
    road_max_segments = int(
        control_cols[3].slider("道路渲染上限", min_value=5000, max_value=80000, value=int(WEB_CFG.get("road_max_render_segments", 35000)), step=1000)
    )
    label_max_points = int(
        control_cols[4].slider("名称标注数量", min_value=20, max_value=500, value=int(WEB_CFG.get("label_max_points", 120)), step=10)
    )
    map_options = ["标注底图", "彩色底图", "深色底图"]
    default_map_style = str(WEB_CFG.get("default_map_style", "标注底图"))
    if default_map_style not in map_options:
        default_map_style = "标注底图"
    map_style_name = control_cols[5].selectbox("底图样式", options=map_options, index=map_options.index(default_map_style))

    body_cols = st.columns([3.2, 1.25])
    with body_cols[0]:
        if node_frame.empty:
            st.warning("当前城市暂无可视化数据。")
            return

        center_lat = float(node_frame["lat"].mean())
        center_lon = float(node_frame["lon"].mean())
        deck = pdk.Deck(
            map_style=_resolve_map_style(map_style_name),
            initial_view_state=pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=float(WEB_CFG.get("default_zoom", 10.8)),
                pitch=float(WEB_CFG.get("default_pitch", 0)),
                bearing=float(WEB_CFG.get("default_bearing", 0)),
            ),
            layers=_build_map_layers(
                node_frame=node_frame,
                road_frame=road_frame,
                heat_max_points=heat_max_points,
                top_n=top_n,
                node_max_points=node_max_points,
                road_max_segments=road_max_segments,
                label_max_points=label_max_points,
            ),
            tooltip={
                "html": "<b>路口名称:</b> {intersection_name}<br/><b>路口编号:</b> {node_id}<br/><b>风险指数:</b> {risk_text}<br/><b>风险等级:</b> {risk_level}<br/><b>建议:</b> {suggestion}",
                "style": {"backgroundColor": "#0e1b2f", "color": "#eaf3ff"},
            },
        )
        st.pydeck_chart(deck, use_container_width=True)
        st.caption("建议缩放到 14 级以上查看路口名称与道路关系。")

    with body_cols[1]:
        _render_governance_panel(node_frame)


def _render_table_view(frame: pd.DataFrame) -> None:
    st.markdown('<p class="title">隐患台账</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">支持按风险等级、路口名称、路口编号检索</p>', unsafe_allow_html=True)

    risk_levels = ["高风险 (I级)", "中风险 (II级)", "低风险 (III级)"]
    selected_levels = st.multiselect("风险等级", options=risk_levels, default=risk_levels)
    keyword = st.text_input("路口名称或编号检索", value="").strip()

    filtered = frame.copy()
    if selected_levels:
        filtered = filtered[filtered["risk_level"].isin(selected_levels)]
    else:
        filtered = filtered.iloc[0:0]
    if keyword:
        name_match = filtered["intersection_name"].str.contains(keyword, case=False, na=False)
        id_match = filtered["node_id"].str.contains(keyword, case=False, na=False)
        filtered = filtered[name_match | id_match]

    page_sizes = [int(x) for x in WEB_CFG.get("table_page_size_options", [50, 100, 200, 500])]
    default_size = int(WEB_CFG.get("default_page_size", 100))
    if default_size not in page_sizes:
        page_sizes = sorted(set(page_sizes + [default_size]))

    page_size = st.selectbox("每页条数", options=page_sizes, index=page_sizes.index(default_size))
    total_rows = len(filtered)
    total_pages = max(1, int(np.ceil(total_rows / page_size)))
    current_page = st.number_input("页码", min_value=1, max_value=total_pages, value=1, step=1)

    start = (int(current_page) - 1) * page_size
    page = filtered.iloc[start : start + page_size].copy()
    page.rename(
        columns={
            "intersection_name": "路口名称",
            "node_id": "路口编号",
            "lat": "纬度",
            "lon": "经度",
            "risk": "风险指数",
            "risk_level": "风险等级",
            "suggestion": "整改建议",
        },
        inplace=True,
    )

    st.dataframe(
        page[["路口名称", "路口编号", "纬度", "经度", "风险指数", "风险等级", "整改建议"]],
        use_container_width=True,
        height=620,
        column_config={"风险指数": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100)},
    )
    st.caption(f"共 {total_rows:,} 条，当前第 {int(current_page)}/{total_pages} 页")


def _compute_intervention(
    frame: pd.DataFrame,
    scenario: str,
    top_n: int,
    reduction_pct: float,
) -> pd.DataFrame:
    """
    节点级干预模型：
      1. 按场景给高风险路口施加额外风险扰动
      2. 对前 top_n 个高风险节点实施治理（降低 reduction_pct% 的风险）
      3. 返回干预后的 DataFrame（含 risk_after 列）
    """
    scenario_boost = {
        "早晚高峰": 8.0,
        "恶劣天气": 14.0,
        "重点活动": 6.0,
        "突发积水": 18.0,
    }
    boost = scenario_boost.get(scenario, 8.0)

    df = frame.copy()
    # 施加扰动：高风险节点扰动更剧烈
    df["risk_after"] = np.clip(df["risk"] + boost * (df["risk"] / 100.0), 0, 100)

    # 干预：对 top_n 高风险节点降低风险
    if top_n > 0 and reduction_pct > 0:
        top_idx = df.nlargest(top_n, "risk_after").index
        df.loc[top_idx, "risk_after"] = np.clip(
            df.loc[top_idx, "risk_after"] * (1.0 - reduction_pct), 0, 100
        )
    return df


def _render_simulation_view(frame: pd.DataFrame) -> None:
    st.markdown('<p class="title">治理推演</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">基于节点级干预模型，量化不同治理投入的韧性改善效果</p>',
        unsafe_allow_html=True,
    )

    if frame.empty:
        st.warning("暂无路口数据，无法进行推演。")
        return

    total_nodes = len(frame)

    ctrl_cols = st.columns([1.2, 1.2, 1.2, 1.5])
    with ctrl_cols[0]:
        scenario = st.selectbox("压力场景", ["早晚高峰", "恶劣天气", "重点活动", "突发积水"])
    with ctrl_cols[1]:
        top_n = st.slider(
            "干预路口数量",
            min_value=0,
            max_value=min(2000, total_nodes),
            value=min(300, total_nodes // 10),
            step=50,
            help="优先对高风险路口实施治理干预",
        )
    with ctrl_cols[2]:
        reduction_pct = st.slider(
            "单节点风险削减比例",
            min_value=0.0,
            max_value=0.8,
            value=0.35,
            step=0.05,
            format="%.0f%%",
            help="每个被干预路口的风险降幅",
        )
    with ctrl_cols[3]:
        st.markdown("")
        st.caption(f"当前场景：**{scenario}**，干预 **{top_n}** 个路口，每口削减 **{reduction_pct*100:.0f}%** 风险")

    # ── 节点级干预计算 ─────────────────────────────────────────────────────
    simulated = _compute_intervention(frame, scenario, top_n, reduction_pct)

    risk_before = simulated["risk"].values
    risk_after = simulated["risk_after"].values

    # 风险等级分箱（干预前/后）
    bins = [-1, 60, 85, 101]
    labels = ["低风险 (III级)", "中风险 (II级)", "高风险 (I级)"]
    level_before = pd.cut(risk_before, bins=bins, labels=labels).value_counts().reindex(labels, fill_value=0)
    level_after  = pd.cut(risk_after,  bins=bins, labels=labels).value_counts().reindex(labels, fill_value=0)

    resilience_before = float(100.0 - np.mean(risk_before))
    resilience_after  = float(100.0 - np.mean(risk_after))
    delta_resilience  = resilience_after - resilience_before

    high_before = int((risk_before >= 85).sum())
    high_after  = int((risk_after  >= 85).sum())

    # ── 关键指标卡 ─────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("干预前韧性评分", f"{resilience_before:.1f}")
    m2.metric("干预后韧性评分", f"{resilience_after:.1f}", delta=f"{delta_resilience:+.1f}")
    m3.metric("高风险路口（干预前）", f"{high_before:,}")
    m4.metric("高风险路口（干预后）", f"{high_after:,}", delta=f"{high_after - high_before:+,}", delta_color="inverse")

    # ── 左：风险等级分布对比柱状图 ────────────────────────────────────────
    left_col, right_col = st.columns([1.6, 1.4])

    with left_col:
        st.markdown('<p class="panel-title">风险等级节点数：干预前 vs 干预后</p>', unsafe_allow_html=True)
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            name="干预前", x=labels, y=level_before.values,
            marker_color=["#4c84e0", "#f59f43", "#db4c40"],
            opacity=0.6,
        ))
        bar_fig.add_trace(go.Bar(
            name="干预后", x=labels, y=level_after.values,
            marker_color=["#4c84e0", "#f59f43", "#db4c40"],
            opacity=1.0,
        ))
        bar_fig.update_layout(
            barmode="group",
            height=280,
            margin=dict(l=6, r=6, t=10, b=6),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # ── 右：风险分布密度曲线（干预前/后） ─────────────────────────────────
    with right_col:
        st.markdown('<p class="panel-title">风险分布密度变化</p>', unsafe_allow_html=True)

        # 简单直方图对比
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=risk_before, name="干预前",
            nbinsx=40, opacity=0.55,
            marker_color="#f59f43",
        ))
        hist_fig.add_trace(go.Histogram(
            x=risk_after, name="干预后",
            nbinsx=40, opacity=0.75,
            marker_color="#23a6d5",
        ))
        hist_fig.update_layout(
            barmode="overlay",
            height=280,
            margin=dict(l=6, r=6, t=10, b=6),
            xaxis_title="风险指数",
            yaxis_title="路口数",
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    # ── 干预效果说明 ───────────────────────────────────────────────────────
    improve_pct = abs(high_before - high_after) / max(high_before, 1) * 100
    st.info(
        f"📊 **推演结果**：在「{scenario}」压力场景下，对 **{top_n}** 个最高风险路口实施治理干预后，"
        f"全网韧性评分提升 **{delta_resilience:+.1f}** 分，"
        f"高风险路口数量从 **{high_before:,}** 降至 **{high_after:,}**（减少 {improve_pct:.1f}%）。"
    )


def _choose_city(available: list[str]) -> str:
    default_city = WEB_CFG.get("default_city", "")
    if default_city not in available:
        default_city = available[0]

    label_to_city = {f"{_city_label(city)} ({city})": city for city in available}
    labels = list(label_to_city.keys())
    default_label = next(label for label, city in label_to_city.items() if city == default_city)
    selected_label = st.sidebar.selectbox("监测城市", labels, index=labels.index(default_label))
    return label_to_city[selected_label]


def main() -> None:
    _init_layer_state()
    available_cities = _available_cities()
    if not available_cities:
        st.error("未找到缓存数据，请先执行 `python scripts/build_web_cache.py`。")
        return

    selected_city = _choose_city(available_cities)
    node_cache_path = PATHS["web_cache_dir"] / f"{selected_city}.parquet"
    road_cache_path = PATHS["web_cache_dir"] / f"{selected_city}_roads.parquet"

    try:
        node_frame = _read_node_cache(str(node_cache_path))
    except Exception as exc:
        st.error(f"点位缓存读取失败：{exc}")
        return

    road_frame = pd.DataFrame(columns=["source_lon", "source_lat", "target_lon", "target_lat", "edge_risk"])
    road_missing = not road_cache_path.exists()
    if not road_missing:
        try:
            road_frame = _read_road_cache(str(road_cache_path))
        except Exception as exc:
            st.warning(f"道路缓存读取失败，仅展示点位与热力图。原因：{exc}")
    else:
        st.warning(
            "当前城市缺少道路缓存，可执行 `python scripts/build_web_cache.py`。"
            "若仅补路网与路口名称，可执行 `python scripts/build_road_cache.py`。"
        )

    st.sidebar.markdown("---")
    view_mode = st.sidebar.radio("工作视图", ["地图治理", "隐患台账", "治理推演"], index=0)
    if st.sidebar.button("刷新缓存", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    update_text = _format_time(node_cache_path.stat().st_mtime)
    high_threshold = float(WEB_CFG.get("high_risk_threshold", 85.0))
    critical_threshold = float(WEB_CFG.get("critical_risk_threshold", 90.0))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("路口总数", f"{len(node_frame):,}")
    m2.metric("高风险路口", f"{int((node_frame['risk'] >= high_threshold).sum()):,}")
    m3.metric("重点风险路口", f"{int((node_frame['risk'] >= critical_threshold).sum()):,}")
    m4.metric("道路样本", f"{len(road_frame):,}" if not road_missing else "未生成")

    if view_mode == "地图治理":
        _render_map_view(selected_city, node_frame, road_frame, update_text)
    elif view_mode == "隐患台账":
        _render_table_view(node_frame)
    else:
        _render_simulation_view(node_frame)


if __name__ == "__main__":
    main()
