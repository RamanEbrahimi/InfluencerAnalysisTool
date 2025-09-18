import pandas as pd
from pathlib import Path
from typing import Dict, List

DATA_FILES = {
    "instagram": "data/instagram_data_united_states.csv",
    "tiktok": "data/tiktok_data_united_states.csv",
    "threads": "data/threads_data_united_states.csv",
    "youtube": "data/youtube_data_united_states.csv",
    "combined": "data/social_media_clean_report.csv",
}

STANDARD_COLUMNS = [
    "platform",
    "rank",
    "name",
    "username",
    "followers",
    "engagement_rate",
    "country",
    "topic_of_influence",
    "potential_reach",
]

PLATFORM_DEFAULTS = {
    "instagram": {
        "platform": "instagram"
    },
    "tiktok": {
        "platform": "tiktok"
    },
    "threads": {
        "platform": "threads"
    },
    "youtube": {
        "platform": "youtube"
    },
}


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def _standardize_columns(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}

    # Build mapping to our standard schema
    mapping: Dict[str, str] = {}
    for std in ["rank", "name", "followers", "engagement_rate", "country", "topic_of_influence", "potential_reach"]:
        if std in cols:
            mapping[cols[std]] = std

    # Username presence varies by platform
    if "username" in cols:
        mapping[cols["username"]] = "username"

    # If platform column exists in combined file use it
    if "platform" in cols:
        mapping[cols["platform"]] = "platform"

    sdf = df.rename(columns=mapping)

    # Inject platform for per-platform files
    if "platform" not in sdf.columns:
        sdf["platform"] = platform

    # Ensure username exists; fallback to normalized name
    if "username" not in sdf.columns:
        sdf["username"] = (
            sdf["name"].astype(str)
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^0-9a-zA-Z_]+", "", regex=True)
            .str.lower()
        )

    # Types and cleaning
    for num_col in ["rank", "followers", "potential_reach"]:
        if num_col in sdf.columns:
            sdf[num_col] = pd.to_numeric(sdf[num_col], errors="coerce")

    if "engagement_rate" in sdf.columns:
        # Strip percent signs if present and cast to float
        sdf["engagement_rate"] = (
            sdf["engagement_rate"].astype(str).str.replace("%", "", regex=False)
        )
        sdf["engagement_rate"] = pd.to_numeric(sdf["engagement_rate"], errors="coerce")

    # Keep standard columns only
    keep: List[str] = [c for c in STANDARD_COLUMNS if c in sdf.columns]
    sdf = sdf[keep]

    return sdf


def load_platform_df(root: Path, platform_key: str) -> pd.DataFrame:
    if platform_key not in DATA_FILES:
        raise KeyError(f"Unknown platform_key: {platform_key}")
    path = root / DATA_FILES[platform_key]
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = _read_csv(path)
    return _standardize_columns(df, PLATFORM_DEFAULTS.get(platform_key, {}).get("platform", platform_key))


def load_all_platforms(root: Path) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for platform_key in ["instagram", "threads", "tiktok", "youtube"]:
        try:
            parts.append(load_platform_df(root, platform_key))
        except FileNotFoundError:
            continue
    # Append combined if present (already standardized and includes platform)
    try:
        combined = load_platform_df(root, "combined")
        parts.append(combined)
    except FileNotFoundError:
        pass

    if not parts:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = pd.concat(parts, ignore_index=True)

    # Deduplicate by platform+username, prefer rows with more completeness
    df["non_nulls"] = df.notna().sum(axis=1)
    df = (
        df.sort_values(["platform", "username", "non_nulls"], ascending=[True, True, False])
        .drop(columns=["non_nulls"], errors="ignore")
        .drop_duplicates(subset=["platform", "username"], keep="first")
    )
    return df
