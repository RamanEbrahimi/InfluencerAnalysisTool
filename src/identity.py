from typing import List, Dict, Tuple
import pandas as pd
from rapidfuzz import process, fuzz
from unidecode import unidecode


def _normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = unidecode(str(text)).lower().strip()
    return "".join(ch for ch in text if ch.isalnum() or ch == "_")


def _candidate_index(df: pd.DataFrame) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for i, row in df.iterrows():
        keys = set()
        keys.add(_normalize_text(row.get("username", "")))
        keys.add(_normalize_text(row.get("name", "")))
        for key in keys:
            if not key:
                continue
            index.setdefault(key, []).append(i)
    return index


def _fuzzy_pair_score(a: str, b: str) -> float:
    return max(
        fuzz.token_sort_ratio(a, b),
        fuzz.ratio(a, b),
        fuzz.partial_ratio(a, b),
    ) / 100.0


def resolve_entities(df: pd.DataFrame, min_score: float = 0.92) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - entities: DataFrame with columns [entity_id, platform, username, name, followers, engagement_rate, potential_reach, country, topic_of_influence]
      - mapping: DataFrame with columns [entity_id, platform, username]
    """
    # Sort for stability
    df = df.copy().sort_values(["platform", "username"]).reset_index(drop=True)

    # Build string keys
    df["key_user"] = df["username"].astype(str).map(_normalize_text)
    df["key_name"] = df["name"].astype(str).map(_normalize_text)

    # Initial grouping by exact key match across any of the keys
    groups: List[List[int]] = []
    visited = set()

    key_to_indices = {}
    for col in ["key_user", "key_name"]:
        for key, sub in df.groupby(col).groups.items():
            if not key:
                continue
            key_to_indices.setdefault(key, set()).update(set(sub))

    # Union groups by overlapping keys
    for idx in range(len(df)):
        if idx in visited:
            continue
        queue = [idx]
        component = set()
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            component.add(cur)
            for key in [df.at[cur, "key_user"], df.at[cur, "key_name"]]:
                for j in key_to_indices.get(key, []):
                    if j not in visited:
                        queue.append(j)
        groups.append(sorted(component))

    # Optional fuzzy merging between groups using display names if usernames differ
    def group_repr(indices: List[int]) -> str:
        names = [_normalize_text(n) for n in df.loc[indices, "name"].astype(str).tolist() if n]
        users = [_normalize_text(u) for u in df.loc[indices, "username"].astype(str).tolist() if u]
        tokens = set(names + users)
        return " ".join(sorted(tokens))

    merged: List[List[int]] = []
    used = [False] * len(groups)
    for i in range(len(groups)):
        if used[i]:
            continue
        base = groups[i]
        base_repr = group_repr(base)
        cluster = set(base)
        used[i] = True
        for j in range(i + 1, len(groups)):
            if used[j]:
                continue
            cand = groups[j]
            score = _fuzzy_pair_score(base_repr, group_repr(cand))
            if score >= min_score:
                cluster.update(cand)
                used[j] = True
        merged.append(sorted(cluster))

    # Build entities
    records = []
    mapping_rows = []
    for eid, indices in enumerate(merged):
        subset = df.loc[indices]
        # Choose canonical name/username per highest followers/reach
        best_idx = subset[["followers", "potential_reach"]].fillna(0).sum(axis=1).idxmax()
        canonical_name = df.at[best_idx, "name"]
        canonical_username = df.at[best_idx, "username"]
        # Aggregate simple stats
        total_followers = subset["followers"].fillna(0).sum()
        mean_engagement = subset["engagement_rate"].astype(float).mean(skipna=True)
        total_reach = subset["potential_reach"].fillna(0).sum()
        active_layers = subset["platform"].nunique()

        records.append({
            "entity_id": eid,
            "name": canonical_name,
            "username": canonical_username,
            "total_followers": float(total_followers),
            "mean_engagement_rate": float(mean_engagement) if pd.notna(mean_engagement) else None,
            "total_potential_reach": float(total_reach),
            "active_layers": int(active_layers),
        })
        for _, r in subset.iterrows():
            mapping_rows.append({
                "entity_id": eid,
                "platform": r["platform"],
                "username": r["username"],
            })

    entities = pd.DataFrame.from_records(records)
    mapping = pd.DataFrame.from_records(mapping_rows)
    return entities, mapping
