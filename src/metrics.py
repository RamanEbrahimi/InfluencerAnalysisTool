from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import networkx as nx
import math
import itertools


def compute_layer_centralities(G: nx.Graph) -> pd.DataFrame:
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return pd.DataFrame(columns=["username", "degree", "betweenness", "eigenvector", "closeness"]) 

    degree = nx.degree_centrality(G)
    # Betweenness and closeness
    betweenness = nx.betweenness_centrality(G, normalized=True)
    closeness = nx.closeness_centrality(G) if G.number_of_nodes() > 1 else {n: 0.0 for n in nodes}
    # Eigenvector may fail on disconnected or small graphs; fallback zeros
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G)
    except Exception:
        eigenvector = {n: 0.0 for n in nodes}

    rows = []
    for n in nodes:
        rows.append({
            "username": n if not isinstance(n, tuple) else n[1],
            "degree": float(degree.get(n, 0.0)),
            "betweenness": float(betweenness.get(n, 0.0)),
            "eigenvector": float(eigenvector.get(n, 0.0)),
            "closeness": float(closeness.get(n, 0.0)),
        })
    return pd.DataFrame(rows)


def compute_supra_pagerank(supra: nx.Graph, alpha: float = 0.85, weight: str = "weight") -> pd.Series:
    if supra.number_of_nodes() == 0:
        return pd.Series(dtype=float)
    try:
        pr = nx.pagerank(supra, alpha=alpha, weight=weight)
    except Exception:
        pr = {n: 1.0 / supra.number_of_nodes() for n in supra.nodes()}
    return pd.Series(pr)


def aggregate_entity_scores_from_supra(pr: pd.Series) -> pd.DataFrame:
    data = []
    for node, score in pr.items():
        # node is (layer, username)
        layer, username = node
        data.append({"layer": layer, "username": username, "supra_pr": float(score)})
    df = pd.DataFrame(data)
    agg = df.groupby("username").agg(total_supra_pr=("supra_pr", "sum"), layers_present=("layer", "nunique")).reset_index()
    return agg


def compute_multiplex_reach(entity_mapping: pd.DataFrame, platform_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each username, sum the potential_reach across layers where present.
    """
    reach_rows = []
    usernames = entity_mapping["username"].unique().tolist()
    for username in usernames:
        total = 0.0
        present_layers: List[str] = []
        for platform, df in platform_dfs.items():
            sub = df[df["username"] == username]
            if not sub.empty:
                present_layers.append(platform)
                total += float(sub["potential_reach"].fillna(0).sum())
        reach_rows.append({
            "username": username,
            "multiplex_reach": total,
            "layers_present": len(set(present_layers)),
        })
    return pd.DataFrame(reach_rows)


def compute_nodes_present_in_all_layers(entity_mapping: pd.DataFrame, layers: List[str]) -> pd.DataFrame:
    target = set(layers)
    grouped = entity_mapping.groupby("username")["platform"].agg(lambda x: set(x)).reset_index()
    grouped["in_all_selected_layers"] = grouped["platform"].apply(lambda s: target.issubset(s))
    return grouped[["username", "in_all_selected_layers"]]


# New: KPI summary, Top 5 lists, hidden influencer score, chart prep

def summarize_kpis(platform_df: pd.DataFrame) -> Dict[str, float]:
    total_nodes = int(platform_df[["platform", "username"]].drop_duplicates().shape[0])
    total_layers = int(platform_df["platform"].nunique())
    avg_engagement = float(platform_df["engagement_rate"].astype(float).mean(skipna=True))
    total_followers = float(platform_df["followers"].fillna(0).sum())
    total_reach = float(platform_df["potential_reach"].fillna(0).sum())
    return {
        "total_nodes": total_nodes,
        "total_layers": total_layers,
        "avg_engagement": avg_engagement,
        "total_followers": total_followers,
        "total_reach": total_reach,
    }


def top_n_by_metric(platform_df: pd.DataFrame, metric: str, n: int = 5, layers: List[str] = None) -> pd.DataFrame:
    df = platform_df.copy()
    if layers:
        df = df[df["platform"].isin(layers)]
    return df.sort_values(metric, ascending=False).head(n)[["platform", "username", "name", metric]]


def engagement_vs_followers_data(platform_df: pd.DataFrame, layers: List[str] = None) -> pd.DataFrame:
    df = platform_df.copy()
    if layers:
        df = df[df["platform"].isin(layers)]
    df = df[["platform", "username", "name", "followers", "engagement_rate", "potential_reach"]].dropna(subset=["followers", "engagement_rate"], how="any")
    return df


def hidden_influencer_score(platform_df: pd.DataFrame, entity_scores: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy helper kept for compatibility; not used in Shapley view.
    """
    es = entity_scores.copy()
    if es.empty:
        return es
    es = es.rename(columns={"layers_present": "layers_present_total"})
    # Safe columns after possible collisions
    coverage = mapping.groupby("username")["platform"].nunique().reset_index().rename(columns={"platform": "layers_present_cov"})
    es = es.merge(coverage, on="username", how="left")
    es["layers_present_cov"] = es["layers_present_cov"].fillna(1)
    return es


def compute_shapley_coalitions(platform_df: pd.DataFrame, metric: str, layers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-influencer Shapley values across selected layers using value function
    v(S) = log(1 + sum_{p in S} x_p), where x_p = normalized metric value on platform p.

    Returns:
      - shapley_df: index per influencer with columns shapley_{platform} and shapley_total
      - coalition_df: best coalition value per k with columns [username, k, best_value, best_coalition]
    """
    if not layers:
        return pd.DataFrame(), pd.DataFrame()

    # Prepare pivot of metric per username x platform
    sub = platform_df[platform_df["platform"].isin(layers)][["platform", "username", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce").fillna(0.0)
    # Per-platform normalization (p99)
    p99 = sub.groupby("platform")[metric].quantile(0.99).replace(0, np.nan)
    pmax = sub.groupby("platform")[metric].max()
    norm_base = p99.fillna(pmax).replace(0, 1.0)

    pivot = sub.pivot_table(index="username", columns="platform", values=metric, aggfunc="max", fill_value=0.0)
    # Ensure all layers as columns
    for p in layers:
        if p not in pivot.columns:
            pivot[p] = 0.0

    # Normalize to [0,1]
    for p in layers:
        pivot[p] = np.clip(pivot[p] / float(norm_base.get(p, 1.0)), 0.0, 1.0)

    n = len(layers)
    fact = math.factorial

    def v_of(mask_layers: List[str], row_vals: pd.Series) -> float:
        s = float(row_vals[mask_layers].sum())
        return math.log1p(s)

    shapley_rows = []
    coalition_rows = []

    # Precompute weights for each subset size for efficiency
    subset_weights = {k: fact(k) * fact(n - k - 1) / fact(n) for k in range(n)}

    for username, row in pivot[layers].iterrows():
        # Shapley per platform
        shapley = {p: 0.0 for p in layers}
        # Enumerate all subsets S of N\{i}
        for i, p in enumerate(layers):
            others = [q for q in layers if q != p]
            for k in range(0, n):
                for S in itertools.combinations(others, k if k <= len(others) else len(others)):
                    weight = subset_weights[len(S)]
                    marginal = v_of(list(S) + [p], row) - v_of(list(S), row)
                    shapley[p] += weight * marginal
        # Total value v(N)
        total_val = v_of(layers, row)
        shapley_row = {f"shapley_{p}": shapley[p] for p in layers}
        shapley_row.update({"username": username, "shapley_total": total_val})
        shapley_rows.append(shapley_row)

        # Best coalition per k
        for k in range(1, n + 1):
            best_val = -1.0
            best_S = None
            for S in itertools.combinations(layers, k):
                val = v_of(list(S), row)
                if val > best_val:
                    best_val = val
                    best_S = S
            coalition_rows.append({
                "username": username,
                "k": k,
                "best_value": best_val,
                "best_coalition": ",".join(best_S) if best_S else "",
            })

    shapley_df = pd.DataFrame(shapley_rows)
    coalition_df = pd.DataFrame(coalition_rows)
    return shapley_df, coalition_df
