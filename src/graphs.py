from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def build_topic_similarity_graph(layer_df: pd.DataFrame, k: int = 10) -> nx.Graph:
    """
    Build a graph where nodes are usernames for a platform layer and edges connect
    similar topic_of_influence using TF-IDF cosine similarity.
    """
    df = layer_df.copy().reset_index(drop=True)
    df["topic_of_influence"] = df["topic_of_influence"].fillna("")
    corpus = df["topic_of_influence"].astype(str).tolist()
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus)

    # Nearest neighbors on cosine distance
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(df)), metric="cosine").fit(X)
    distances, indices = nbrs.kneighbors(X)

    G = nx.Graph(platform=df["platform"].iloc[0] if "platform" in df else None, layer_type="topic")

    # Add nodes with attributes
    for i, row in df.iterrows():
        G.add_node(
            row["username"],
            name=row.get("name"),
            followers=float(row.get("followers", np.nan)) if pd.notna(row.get("followers")) else None,
            engagement_rate=float(row.get("engagement_rate", np.nan)) if pd.notna(row.get("engagement_rate")) else None,
            potential_reach=float(row.get("potential_reach", np.nan)) if pd.notna(row.get("potential_reach")) else None,
            country=row.get("country"),
            topic_of_influence=row.get("topic_of_influence"),
            platform=row.get("platform"),
        )

    # Add edges by similarity (skip self at index 0)
    for i in range(len(df)):
        u = df.at[i, "username"]
        for j_idx, j in enumerate(indices[i][1:]):
            v = df.at[int(j), "username"]
            sim = 1.0 - float(distances[i][j_idx + 1])
            if np.isfinite(sim) and sim > 0:
                G.add_edge(u, v, weight=sim, relation="topic_similarity")

    return G


def build_numeric_knn_graph(layer_df: pd.DataFrame, features: Optional[List[str]] = None, k: int = 10) -> nx.Graph:
    """
    Build KNN graph on numeric features like followers, engagement_rate, potential_reach.
    """
    df = layer_df.copy().reset_index(drop=True)
    if features is None:
        features = ["followers", "engagement_rate", "potential_reach"]

    X = df[features].fillna(0.0).astype(float).to_numpy()
    if len(df) == 0:
        return nx.Graph()

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(df)), metric="euclidean").fit(X)
    distances, indices = nbrs.kneighbors(X)

    G = nx.Graph(platform=df["platform"].iloc[0] if "platform" in df else None, layer_type="numeric_knn")

    for i, row in df.iterrows():
        G.add_node(
            row["username"],
            name=row.get("name"),
            followers=float(row.get("followers", np.nan)) if pd.notna(row.get("followers")) else None,
            engagement_rate=float(row.get("engagement_rate", np.nan)) if pd.notna(row.get("engagement_rate")) else None,
            potential_reach=float(row.get("potential_reach", np.nan)) if pd.notna(row.get("potential_reach")) else None,
            country=row.get("country"),
            topic_of_influence=row.get("topic_of_influence"),
            platform=row.get("platform"),
        )

    for i in range(len(df)):
        u = df.at[i, "username"]
        for j_idx, j in enumerate(indices[i][1:]):
            v = df.at[int(j), "username"]
            d = float(distances[i][j_idx + 1])
            if np.isfinite(d):
                w = 1.0 / (1.0 + d)
                G.add_edge(u, v, weight=w, relation="numeric_knn")

    return G


def build_multiplex_supra_graph(layer_graphs: Dict[str, nx.Graph], coupling_weight: float = 0.5) -> nx.Graph:
    """
    Construct a supra-graph for multiplex analysis:
      - Duplicate nodes per layer: node is (layer, username)
      - Intra-layer edges from each layer graph
      - Inter-layer coupling edges connecting the same username across layers with given weight
    """
    supra = nx.Graph(graph_type="multiplex")

    # Add intra-layer nodes and edges
    for layer_name, G in layer_graphs.items():
        for u, attrs in G.nodes(data=True):
            supra.add_node((layer_name, u), **attrs, layer=layer_name, username=u)
        for u, v, eattrs in G.edges(data=True):
            supra.add_edge((layer_name, u), (layer_name, v), **eattrs, layer=layer_name)

    # Inter-layer coupling by username presence
    layers = list(layer_graphs.keys())
    username_to_layers: Dict[str, List[str]] = {}
    for layer_name, G in layer_graphs.items():
        for u in G.nodes():
            username_to_layers.setdefault(u, []).append(layer_name)

    for username, ls in username_to_layers.items():
        if len(ls) < 2:
            continue
        for i in range(len(ls)):
            for j in range(i + 1, len(ls)):
                u = (ls[i], username)
                v = (ls[j], username)
                supra.add_edge(u, v, weight=coupling_weight, relation="interlayer")

    return supra


def build_multiplex_supra_graph_with_entities(layer_graphs: Dict[str, nx.Graph], entity_mapping: pd.DataFrame, coupling_weight: float = 0.5) -> nx.Graph:
    """
    Like build_multiplex_supra_graph but interlayer coupling is done using resolved entity_id
    so that cross-platform aliases are unified.
    entity_mapping columns: [entity_id, platform, username]
    """
    supra = nx.Graph(graph_type="multiplex")

    # Add intra-layer nodes and edges
    for layer_name, G in layer_graphs.items():
        for u, attrs in G.nodes(data=True):
            supra.add_node((layer_name, u), **attrs, layer=layer_name, username=u)
        for u, v, eattrs in G.edges(data=True):
            supra.add_edge((layer_name, u), (layer_name, v), **eattrs, layer=layer_name)

    # Build entity -> usernames per layer from mapping
    entity_to_nodes: Dict[int, Dict[str, List[str]]] = {}
    for _, row in entity_mapping.iterrows():
        eid = int(row["entity_id"]) if pd.notna(row["entity_id"]) else None
        plat = row["platform"]
        uname = row["username"]
        if eid is None:
            continue
        entity_to_nodes.setdefault(eid, {}).setdefault(plat, []).append(uname)

    # Add interlayer edges for any pair of layers within the same entity
    for eid, layer_to_users in entity_to_nodes.items():
        layers = sorted(layer_to_users.keys())
        if len(layers) < 2:
            continue
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                li, lj = layers[i], layers[j]
                for ui in layer_to_users[li]:
                    for uj in layer_to_users[lj]:
                        if (li, ui) in supra and (lj, uj) in supra:
                            supra.add_edge((li, ui), (lj, uj), weight=coupling_weight, relation="interlayer_entity")

    return supra
