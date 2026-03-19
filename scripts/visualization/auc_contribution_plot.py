from pathlib import Path
from typing import Dict, Tuple, Set, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================

DATA_DIR = Path("data/results")

RACE_FILES = {
    "White": DATA_DIR / "white_top_features.csv",
    "Black": DATA_DIR / "black_top_features.csv",
    "Asian": DATA_DIR / "asian_top_features.csv",
}

COMMON_CSV = DATA_DIR / "common_features.csv"
OUT_DIR = Path("outputs")


# =========================
# Helpers
# =========================

def normalize_feature_names(series) -> pd.Series:
    return pd.Series(series, copy=False).astype(str).str.strip().str.lower()


def read_auc_table(path: str, feature_col: str = "Feature") -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {feature_col, "Cumulative_AUC"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def common_features_from_files(race_files: Dict[str, str], feature_col: str = "Feature") -> Set[str]:
    sets = [set(normalize_feature_names(read_auc_table(p, feature_col)[feature_col]))
            for p in race_files.values()]
    return set.intersection(*sets)


def common_features_from_csv(common_csv: str, feature_col: str = "Feature") -> Set[str]:
    df = pd.read_csv(common_csv)
    if feature_col not in df.columns:
        raise ValueError(f"{common_csv} missing column '{feature_col}'")
    return set(normalize_feature_names(df[feature_col]))


# =========================
# Core computation
# =========================

def decompose_auc_gains(
    df: pd.DataFrame,
    common_set: Set[str],
    k: int,
    feature_col: str = "Feature",
    clip_negative_deltas: bool = True
) -> dict:
    """
    Baseline = AUC at row k (1-indexed).
    Splits gain beyond baseline into common vs population-specific features.
    """
    df = df.copy()
    df["_feat_norm"] = normalize_feature_names(df[feature_col])
    auc = df["Cumulative_AUC"].astype(float).to_numpy()

    base_idx = k - 1
    if base_idx < 0 or base_idx >= len(df):
        raise ValueError(f"k={k} is out of range for {len(df)} rows")

    deltas = np.zeros_like(auc)
    deltas[1:] = auc[1:] - auc[:-1]
    deltas[:base_idx + 1] = 0.0
    if clip_negative_deltas:
        deltas = np.where(deltas < 0, 0.0, deltas)

    is_common = df["_feat_norm"].isin(common_set).to_numpy()
    gain_common = float(deltas[is_common].sum())
    gain_specific = float(deltas[~is_common].sum())

    auc_base = float(auc[base_idx])
    auc_final = float(auc[-1])
    gain_total = auc_final - auc_base

    return {
        "auc_base": auc_base,
        "auc_final": auc_final,
        "gain_total": gain_total,
        "gain_common": gain_common,
        "gain_specific": gain_specific,
        "pct_common": 100.0 * gain_common / gain_total if gain_total > 0 else 0.0,
        "pct_specific": 100.0 * gain_specific / gain_total if gain_total > 0 else 0.0,
        "baseline_rank": k,
    }


def build_summary(
    race_files: Dict[str, str],
    k: int,
    feature_col: str = "Feature",
    common_features_csv: Optional[str] = None,
    common_feature_col: str = "Feature",
    clip_negative_deltas: bool = True
) -> pd.DataFrame:
    if common_features_csv:
        common_set = common_features_from_csv(common_features_csv, common_feature_col)
    else:
        common_set = common_features_from_files(race_files, feature_col)

    rows = []
    for race, path in race_files.items():
        df = read_auc_table(path, feature_col)
        stats = decompose_auc_gains(df, common_set, k=k, feature_col=feature_col,
                                    clip_negative_deltas=clip_negative_deltas)
        rows.append({"Race": race, **stats})
    return pd.DataFrame(rows).set_index("Race")


# =========================
# Plot
# =========================

def plot_fraction_of_gain(
    summary: pd.DataFrame,
    baseline_label: str,
    out_path: Optional[Path] = None,
    palette: Tuple[str, str] = ("#6baed6", "#08519c")
) -> None:
    """
    100% stacked bars of gain beyond baseline:
      lower = percent from common features
      upper = percent from population-specific features
    """
    races = summary.index.tolist()
    gain_common = summary["gain_common"].to_numpy()
    gain_spec = summary["gain_specific"].to_numpy()
    total_gain = gain_common + gain_spec

    frac_common = np.divide(gain_common, total_gain, out=np.zeros_like(gain_common), where=total_gain > 0)
    frac_spec = np.divide(gain_spec, total_gain, out=np.zeros_like(gain_spec), where=total_gain > 0)

    common_color, spec_color = palette

    plt.figure(figsize=(9, 5.2))
    plt.bar(races, frac_common * 100, label="Common features", color=common_color, edgecolor="white")
    plt.bar(races, frac_spec * 100, bottom=frac_common * 100, label="Population specific", color=spec_color, edgecolor="white")

    for i, _ in enumerate(races):
        plt.text(i, 101.5, f"{(frac_spec[i] * 100):.0f}% specific", ha="center", va="bottom", fontsize=9)

    plt.ylabel("Fraction of AUC gain beyond baseline (%)")
    plt.title(f"Share of improvement beyond {baseline_label} baseline")
    plt.ylim(0, 110)
    plt.legend(title=f"Baseline = {baseline_label}", frameon=True)
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")


# =========================
# Run
# =========================

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot 1: beyond Age baseline
summary_age = build_summary(
    race_files=RACE_FILES,
    k=1,
    common_features_csv=COMMON_CSV,
    common_feature_col="Feature",
)
plot_fraction_of_gain(summary_age, baseline_label="Age",
                      out_path=OUT_DIR / "auc_beyond_age.png")
print("Age baseline summary:")
print(summary_age.round(4))

# Plot 2: beyond Top-4 baseline
summary_top4 = build_summary(
    race_files=RACE_FILES,
    k=4,
    common_features_csv=COMMON_CSV,
    common_feature_col="Feature",
)
plot_fraction_of_gain(summary_top4, baseline_label="Top-4",
                      out_path=OUT_DIR / "auc_beyond_top4.png")
print("\nTop-4 baseline summary:")
print(summary_top4.round(4))
