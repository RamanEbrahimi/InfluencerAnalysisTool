import json
from pathlib import Path
from typing import Dict, List

import dash
from dash import Dash, html, dcc, Input, Output, State, ctx, dash_table
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go

from src.data_loader import load_all_platforms
from src.identity import resolve_entities
from src.graphs import build_topic_similarity_graph, build_numeric_knn_graph, build_multiplex_supra_graph_with_entities
from src.metrics import (
    compute_layer_centralities,
    compute_supra_pagerank,
    aggregate_entity_scores_from_supra,
    compute_nodes_present_in_all_layers,
    summarize_kpis,
    top_n_by_metric,
    engagement_vs_followers_data,
    compute_shapley_coalitions,
)

ROOT = Path(__file__).resolve().parents[1]

LAYER_COLORS = {
    "instagram": "#E1306C",
    "threads": "#111111",
    "tiktok": "#69C9D0",
    "youtube": "#FF0000",
}


def graph_to_cyto_elements(G: nx.Graph, layer_name: str) -> List[Dict]:
    elements = []
    for n, attrs in G.nodes(data=True):
        elements.append({
            "data": {
                "id": f"{layer_name}::{n}",
                "label": attrs.get("name") or n,
                "username": n,
                "layer": layer_name,
                **{k: v for k, v in attrs.items() if isinstance(v, (str, int, float)) or v is None},
            }
        })
    for u, v, e in G.edges(data=True):
        elements.append({
            "data": {
                "source": f"{layer_name}::{u}",
                "target": f"{layer_name}::{v}",
                "weight": float(e.get("weight", 1.0)),
                "relation": e.get("relation", ""),
            }
        })
    return elements


def fmt_int(x: float) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "—"


def fmt_pct(x: float) -> str:
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return "—"


external_stylesheets = [dbc.themes.COSMO]
app: Dash = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Influencer Multiplex Dashboard"

# Load and preprocess data
platform_df = load_all_platforms(ROOT)
entities, mapping = resolve_entities(platform_df)

# Build per-layer graphs (topic similarity) once on load
layer_graphs: Dict[str, nx.Graph] = {}
for platform_name, sub in platform_df.groupby("platform"):
    layer_graphs[platform_name] = build_topic_similarity_graph(sub, k=10)

# Use entity-based coupling for multiplex supra-graph
supra = build_multiplex_supra_graph_with_entities(layer_graphs, mapping, coupling_weight=0.5)
supra_pr = compute_supra_pagerank(supra)
entity_scores = aggregate_entity_scores_from_supra(supra_pr)

# Precompute per-layer stats
layer_stats: Dict[str, pd.DataFrame] = {}
for layer_name, G in layer_graphs.items():
    layer_stats[layer_name] = compute_layer_centralities(G)

platform_options = [{"label": p.capitalize(), "value": p} for p in layer_graphs.keys()]

# KPIs
kpi_vals = summarize_kpis(platform_df)

# Base stylesheet and color mapping
base_stylesheet = [
    {"selector": "node", "style": {"label": "data(label)", "width": 18, "height": 18, "font-size": 10}},
    {"selector": "edge", "style": {"width": 1, "line-color": "#ccc", "opacity": 0.6}},
]
for layer, color in LAYER_COLORS.items():
    base_stylesheet.append({
        "selector": f"node[layer = '{layer}']",
        "style": {"background-color": color}
    })

# Layout
app.layout = dbc.Container([
    html.Br(),
    dbc.Row([
        dbc.Col(html.H2("Influencer Multiplex Network Analysis"), width=8),
        dbc.Col(dbc.Badge("v1", color="primary", className="ms-2"), width=1)
    ], align="center"),
    html.Br(),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total Nodes"), html.H3(f"{kpi_vals['total_nodes']:,}")])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Layers"), html.H3(f"{kpi_vals['total_layers']}")])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Avg Engagement"), html.H3(f"{kpi_vals['avg_engagement']:.2f}%")])), md=2),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total Followers"), html.H3(f"{kpi_vals['total_followers']:.0f}")])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total Reach"), html.H3(f"{kpi_vals['total_reach']:.0f}")])), md=3),
    ], className="gy-3"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Label("Layers"),
            dcc.Dropdown(options=platform_options, value=[o["value"] for o in platform_options], multi=True, id="layers-select"),
            html.Br(),
            html.Label("Graph type"),
            dcc.RadioItems(options=[{"label": "Topic", "value": "topic"}, {"label": "Numeric KNN", "value": "knn"}], value="topic", id="graph-type"),
            html.Br(),
            html.Label("Highlight layer"),
            dcc.Dropdown(id="focus-layer", options=platform_options, placeholder="Optional"),
            html.Div(id="nodes-in-all-layers", className="text-muted", style={"marginTop": "8px", "fontSize": "12px"}),
        ], md=3),
        dbc.Col([
            cyto.Cytoscape(id="cytoscape", layout={'name': 'cose'}, style={'width': '100%', 'height': '600px', 'backgroundColor': '#f8f9fa'}, elements=[], stylesheet=base_stylesheet),
        ], md=6),
        dbc.Col([
            html.H5("Selected node"),
            html.Div(id="node-stats", style={"fontSize": "12px"}),
            html.Br(),
            html.H6("Nodes present in all selected layers"),
            dash_table.DataTable(id="all-layers-table", columns=[
                {"name": "username", "id": "username"},
                {"name": "layers_present", "id": "layers_present"},
                {"name": "total_followers", "id": "total_followers"},
                {"name": "total_potential_reach", "id": "total_potential_reach"},
            ], page_size=6, style_table={"height": "220px", "overflowY": "auto"}, style_cell={"fontSize": 11})
        ], md=3),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.H5("Top 5"),
            dcc.Dropdown(id="top-metric", options=[
                {"label": "Followers", "value": "followers"},
                {"label": "Engagement Rate", "value": "engagement_rate"},
                {"label": "Potential Reach", "value": "potential_reach"},
                {"label": "Multiplex Rank (supra PR)", "value": "total_supra_pr"}
            ], value="followers"),
            dcc.Graph(id="top5-chart")
        ], md=6),
        dbc.Col([
            html.H5("Engagement vs Followers"),
            dcc.Graph(id="scatter-eng-follow"),
        ], md=6)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.H5("Shapley Coalition Impact"),
            html.Div([
                dcc.Dropdown(id="shapley-metric", options=[
                    {"label": "Followers", "value": "followers"},
                    {"label": "Engagement Rate", "value": "engagement_rate"},
                    {"label": "Potential Reach", "value": "potential_reach"},
                ], value="followers"),
                dcc.Slider(id="coalition-k", min=1, max=4, step=1, value=2, marks={i: str(i) for i in range(1,5)}),
            ], style={"marginBottom": "8px"}),
            dash_table.DataTable(id="hidden-table", columns=[
                {"name": "username", "id": "username"},
                {"name": "best_value", "id": "best_value"},
                {"name": "best_coalition", "id": "best_coalition"},
                {"name": "shapley_total", "id": "shapley_total"},
            ], page_size=10, style_table={"height": "300px", "overflowY": "auto"}, style_cell={"fontSize": 11})
        ], md=6),
        dbc.Col([
            html.H5("3D Multiplex Layers"),
            dcc.Graph(id="graph-3d"),
        ], md=6)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.H5("Multiplex MultiRank (Top 20)"),
            dcc.Loading(dcc.Graph(id="multiplex-bar")),
        ], md=12)
    ]),
    html.Br(),
], fluid=True)


@app.callback(
    Output("cytoscape", "elements"),
    Output("cytoscape", "stylesheet"),
    Output("nodes-in-all-layers", "children"),
    Output("all-layers-table", "data"),
    Input("layers-select", "value"),
    Input("graph-type", "value"),
    Input("focus-layer", "value"),
)

def update_graph(selected_layers: List[str], graph_type: str, focus_layer: str):
    if not selected_layers:
        return [], base_stylesheet, "Select at least one layer", []

    # Build elements per selected layer on the fly based on graph_type
    elements: List[Dict] = []
    platform_dfs: Dict[str, pd.DataFrame] = {
        p: platform_df[platform_df["platform"] == p] for p in selected_layers
    }

    for p in selected_layers:
        sub = platform_dfs[p]
        G = build_topic_similarity_graph(sub) if graph_type == "topic" else build_numeric_knn_graph(sub)
        elements.extend(graph_to_cyto_elements(G, p))

    # Nodes present in all selected layers
    sub_map = mapping[mapping["platform"].isin(selected_layers)]
    all_df = compute_nodes_present_in_all_layers(sub_map, selected_layers)
    num_all = int(all_df[all_df["in_all_selected_layers"]].shape[0]) if not all_df.empty else 0

    # Build table data for nodes present in all layers
    usernames_all = set(all_df[all_df["in_all_selected_layers"]]["username"].tolist()) if not all_df.empty else set()
    table_rows: List[Dict] = []
    for uname in sorted(usernames_all):
        rows = []
        followers_sum = 0.0
        reach_sum = 0.0
        for p in selected_layers:
            r = platform_dfs[p][platform_dfs[p]["username"] == uname]
            if not r.empty:
                followers_sum += float(r["followers"].fillna(0).sum())
                reach_sum += float(r["potential_reach"].fillna(0).sum())
                rows.append(p)
        table_rows.append({
            "username": uname,
            "layers_present": len(set(rows)),
            "total_followers": int(followers_sum),
            "total_potential_reach": int(reach_sum),
        })

    # Stylesheet with optional focus layer highlight
    stylesheet = list(base_stylesheet)
    if focus_layer and focus_layer in selected_layers:
        stylesheet.extend([
            {"selector": "node", "style": {"opacity": 0.25}},
            {"selector": f"node[layer = '{focus_layer}']", "style": {"opacity": 1.0, "border-width": 2, "border-color": "#1f77b4", "width": 26, "height": 26}},
            {"selector": "edge", "style": {"opacity": 0.15}},
        ])

    return elements, stylesheet, f"Nodes present in all selected layers: {num_all}", table_rows


@app.callback(
    Output("node-stats", "children"),
    Input("cytoscape", "tapNodeData"),
    State("layers-select", "value"),
)

def on_node_click(tap_node, selected_layers):
    if tap_node is None:
        return dbc.Alert("Click a node to see stats", color="secondary", className="small")
    username = tap_node.get("username")

    # Per-layer stats for this user
    stats_per_layer = []
    total_followers = 0.0
    total_reach = 0.0
    for p in selected_layers:
        sub = platform_df[platform_df["platform"] == p]
        row = sub[sub["username"] == username]
        cent = layer_stats.get(p)
        cent_row = None
        if cent is not None:
            cr = cent[cent["username"] == username]
            cent_row = cr.iloc[0].to_dict() if not cr.empty else None
        if not row.empty:
            r = row.iloc[0]
            followers = float(r.get("followers", 0) or 0)
            reach = float(r.get("potential_reach", 0) or 0)
            total_followers += followers
            total_reach += reach
            stats_per_layer.append({
                "layer": p,
                "followers": followers,
                "engagement_rate": float(r.get("engagement_rate", 0) or 0),
                "potential_reach": reach,
                "topic": r.get("topic_of_influence"),
                "centrality": cent_row,
            })

    # Multiplex aggregates
    supra_rows = entity_scores[entity_scores["username"] == username]
    multiplex_score = float(supra_rows["total_supra_pr"].iloc[0]) if not supra_rows.empty else 0.0
    active_layers_row = entities[entities["username"] == username]
    active_layers = int(active_layers_row["active_layers"].iloc[0]) if not active_layers_row.empty else len(stats_per_layer)

    header = dbc.CardHeader([html.Strong(f"@{username}"), html.Span(f"  •  Layers: {active_layers}", className="ms-1 text-muted")])
    summary_badges = dbc.Row([
        dbc.Col(dbc.Badge(f"Followers {fmt_int(total_followers)}", color="primary", className="me-1"), md="auto"),
        dbc.Col(dbc.Badge(f"Reach {fmt_int(total_reach)}", color="info", className="me-1"), md="auto"),
        dbc.Col(dbc.Badge(f"Multiplex {multiplex_score:.4f}", color="success", className="me-1"), md="auto"),
    ], className="g-1")

    per_layer_cards = []
    for s in stats_per_layer:
        cent = s.get("centrality") or {}
        deg = float(cent.get("degree", 0) or 0)
        btw = float(cent.get("betweenness", 0) or 0)
        eig = float(cent.get("eigenvector", 0) or 0)
        clo = float(cent.get("closeness", 0) or 0)
        color = LAYER_COLORS.get(s["layer"], "#6c757d")
        card = dbc.Card([
            dbc.CardHeader([html.Span(s["layer"].capitalize(), className="fw-bold"), html.Span("  •  ", className="text-muted"), html.Span(s.get("topic") or "")], style={"borderLeft": f"4px solid {color}"}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.Div([html.Small("Followers"), html.H6(fmt_int(s["followers"]))]), md=3),
                    dbc.Col(html.Div([html.Small("Engagement"), html.H6(fmt_pct(s["engagement_rate"]))]), md=3),
                    dbc.Col(html.Div([html.Small("Reach"), html.H6(fmt_int(s["potential_reach"]))]), md=3),
                    dbc.Col(html.Div([html.Small("Closeness"), html.H6(f"{clo:.3f}")]), md=3),
                ]),
                html.Div([
                    html.Div([html.Small("Degree"), dbc.Badge(f"{deg:.3f}", color="light", text_color="dark", className="ms-2")]),
                    dbc.Progress(value=min(int(deg*100),100), color="primary", className="mb-2"),
                    html.Div([html.Small("Betweenness"), dbc.Badge(f"{btw:.3f}", color="light", text_color="dark", className="ms-2")]),
                    dbc.Progress(value=min(int(btw*100),100), color="warning", className="mb-2"),
                    html.Div([html.Small("Eigenvector"), dbc.Badge(f"{eig:.3f}", color="light", text_color="dark", className="ms-2")]),
                    dbc.Progress(value=min(int(eig*100),100), color="success", className="mb-2"),
                    html.Div([html.Small("Closeness"), dbc.Badge(f"{clo:.3f}", color="light", text_color="dark", className="ms-2")]),
                    dbc.Progress(value=min(int(clo*100),100), color="secondary"),
                ], className="mt-2")
            ])
        ], className="mb-2")
        per_layer_cards.append(card)

    return dbc.Card([header, dbc.CardBody([summary_badges] + per_layer_cards)], className="mb-2")


@app.callback(
    Output("top5-chart", "figure"),
    Input("top-metric", "value"),
    Input("layers-select", "value"),
)

def update_top5(metric, layers):
    import plotly.express as px
    if metric == "total_supra_pr":
        df = entity_scores.sort_values(metric, ascending=False).head(5)
        fig = px.bar(df, x="username", y=metric, title=f"Top 5 by {metric}")
        fig.update_layout(xaxis_tickangle=-30, template="plotly_white")
        return fig
    df = top_n_by_metric(platform_df, metric, n=5, layers=layers)
    fig = px.bar(df, x="username", y=metric, color="platform", title=f"Top 5 by {metric}")
    fig.update_layout(xaxis_tickangle=-30, template="plotly_white")
    return fig


@app.callback(
    Output("scatter-eng-follow", "figure"),
    Input("layers-select", "value"),
)

def update_scatter(layers):
    import plotly.express as px
    df = engagement_vs_followers_data(platform_df, layers)
    if df.empty:
        return go.Figure()
    fig = px.scatter(df, x="followers", y="engagement_rate", color="platform", size="potential_reach", hover_name="name",
                     title="Engagement vs Followers (bubble size: potential reach)", log_x=True)
    fig.update_layout(template="plotly_white")
    return fig


@app.callback(
    Output("hidden-table", "data"),
    Input("layers-select", "value"),
    Input("shapley-metric", "value"),
    Input("coalition-k", "value"),
)

def update_hidden(layers, base_metric, k):
    shapley_df, coal_df = compute_shapley_coalitions(platform_df, base_metric, layers)
    if coal_df.empty:
        return []
    # Filter for chosen coalition size and attach total value for context
    coal_k = coal_df[coal_df["k"] == int(k)].copy()
    shp = shapley_df[["username", "shapley_total"]]
    out = coal_k.merge(shp, on="username", how="left")
    out = out.sort_values(["best_value", "shapley_total"], ascending=False).head(30)
    out["best_value"] = out["best_value"].round(3)
    out["shapley_total"] = out["shapley_total"].round(3)
    return out.to_dict("records")


@app.callback(
    Output("graph-3d", "figure"),
    Input("layers-select", "value"),
)

def update_3d(layers):
    # 3D scatter: each platform is a z-layer; positions derived from followers and engagement
    fig = go.Figure()
    z_map = {p: i for i, p in enumerate(sorted(layers))} if layers else {}
    for p in layers:
        sub = platform_df[platform_df["platform"] == p].copy()
        if sub.empty:
            continue
        # Normalize positions for aesthetics
        x = np.log1p(sub["followers"].fillna(0).astype(float))
        y = sub["engagement_rate"].fillna(0).astype(float)
        z = np.full(len(sub), z_map[p])
        size = np.clip(np.log1p(sub["potential_reach"].fillna(0).astype(float)) * 2.0, 4, 20)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                   marker=dict(size=size, color=LAYER_COLORS.get(p, '#888'), opacity=0.7),
                                   text=sub["username"], name=p))
    fig.update_layout(scene=dict(zaxis=dict(title="Layer", tickvals=list(z_map.values()), ticktext=list(z_map.keys())),
                                 xaxis_title="log(1+followers)", yaxis_title="engagement_rate"),
                      height=500, margin=dict(l=0, r=0, t=30, b=0), template="plotly_white")
    return fig


@app.callback(
    Output("multiplex-bar", "figure"),
    Input("layers-select", "value"),
)

def update_multiplex_bar(_layers):
    df = entity_scores.sort_values("total_supra_pr", ascending=False).head(20)
    if df.empty:
        return go.Figure()
    fig = go.Figure(data=[go.Bar(x=df["username"], y=df["total_supra_pr"], marker_color="#1f77b4")])
    fig.update_layout(title="Top multiplex MultiRank (supra PageRank)", xaxis_tickangle=-45, margin=dict(l=10, r=10, t=30, b=80), template="plotly_white")
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
