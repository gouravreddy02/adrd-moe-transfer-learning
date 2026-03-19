import warnings
from pathlib import Path
from typing import Dict, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")


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

def build_common_csv(race_files: Dict[str, Path], out_path: Path) -> None:
    """Compute intersection of top features across all races and save to CSV."""
    sets = []
    for path in race_files.values():
        df = pd.read_csv(path).rename(columns={"Predictor": "Feature"})
        sets.append(set(df["Feature"]))
    common = set.intersection(*sets)
    pd.DataFrame({"Feature": sorted(common)}).to_csv(out_path, index=False)


def load_common_features(common_csv: Path, feature_col: str = "Feature") -> Set[str]:
    df = pd.read_csv(common_csv)
    if feature_col not in df.columns:
        raise ValueError(f"'{feature_col}' not in {common_csv} columns: {list(df.columns)}")
    return set(df[feature_col].astype(str).str.strip().str.lower().unique())


# =========================
# Plotting
# =========================

def plot_auc_single_sorted(
    merged_file: Path,
    race_name: str,
    use_metric: str = "SHAP_Normalized",
    limit: Optional[int] = None,
    output_filename: Optional[Path] = None,
    auc_ylim: tuple = (0.7, 1.0),
    shap_ylim: tuple = (0, 1),
    common_features_csv: Optional[Path] = None,
    common_feature_col: str = "Feature",
    other_color: str = "#1f77b4",
    common_color: str = "darkblue",
) -> dict:
    df = pd.read_csv(merged_file)

    required = ["Feature", "Cumulative_AUC", "Lower_95CI", "Upper_95CI", use_metric]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{race_name}: missing columns {missing}")

    if limit is not None and limit > 0:
        df = df.head(limit)
    if df.empty:
        raise ValueError(f"{race_name}: no rows to plot after applying limit")

    common_set = set()
    if common_features_csv:
        common_set = load_common_features(common_features_csv, feature_col=common_feature_col)

    desc_col = "Variable name" if "Variable name" in df.columns else "Feature"
    labels = df[desc_col].astype(str).to_list()
    x = np.arange(1, len(df) + 1)

    metric_vals = df[use_metric].astype(float).to_numpy()
    auc = df["Cumulative_AUC"].astype(float).to_numpy()
    ci_low = df["Lower_95CI"].astype(float).to_numpy()
    ci_high = df["Upper_95CI"].astype(float).to_numpy()

    if common_set:
        is_common = df["Feature"].astype(str).str.strip().str.lower().isin(common_set).to_numpy()
        colors = [common_color if flag else other_color for flag in is_common]
        n_common = int(is_common.sum())
    else:
        colors = [other_color] * len(df)
        n_common = 0

    fig, ax1 = plt.subplots(figsize=(18, 10))
    ax1.bar(x, metric_vals, alpha=0.85, color=colors, edgecolor="white", linewidth=0.6)
    ax1.set_xlabel("Feature Rank", fontsize=18, fontweight="bold")
    ax1.set_ylabel(use_metric, fontsize=18, fontweight="bold", color="darkblue")
    ax1.tick_params(axis="y", labelcolor="darkblue", labelsize=15)
    ax1.set_ylim(*shap_ylim)

    ax2 = ax1.twinx()
    ax2.plot(x, auc, color="darkred", marker="o", linewidth=3, markersize=5,
             markerfacecolor="red", markeredgecolor="white", markeredgewidth=1.5,
             label="Cumulative AUC", zorder=10)
    ax2.fill_between(x, ci_low, ci_high, color="red", alpha=0.15, label="95% CI")
    ax2.set_ylabel("Cumulative AUC", fontsize=18, fontweight="bold", color="darkred")
    ax2.tick_params(axis="y", labelcolor="darkred", labelsize=15)
    ax2.set_ylim(*auc_ylim)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=13)

    plt.title(f"{race_name} - Population", fontsize=20, fontweight="bold", pad=25)

    ax1.legend(
        handles=[
            Patch(facecolor=common_color, label="Common features"),
            Patch(facecolor=other_color, label="Race Specific features"),
        ],
        title="Bar Coloring",
        loc="upper right",
    )

    gain = auc[-1] - auc[0]
    ax1.text(
        0.02, 0.97,
        f"Max AUC: {auc.max():.4f}\nΔAUC: +{gain:.4f}\nFeatures: {len(df)}\nCommon: {n_common}",
        transform=ax1.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightsteelblue", alpha=0.95, edgecolor="navy"),
        va="top", ha="left", fontsize=14, fontweight="bold",
    )

    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.2, linestyle=":")
    ax1.set_facecolor("#fafafa")

    plt.tight_layout()
    if output_filename:
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    return {
        "race": race_name,
        "features": int(len(df)),
        "max_auc": float(auc.max()),
        "auc_improvement": float(gain),
        "common_count": n_common,
    }


def plot_auc_all_separate(
    race_files: Dict[str, Path],
    outdir: Path = OUT_DIR,
    **kwargs,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    for race, csv_path in race_files.items():
        results[race] = plot_auc_single_sorted(
            merged_file=csv_path,
            race_name=race,
            output_filename=outdir / f"Cumulative_AUC_{race.lower()}.png",
            **kwargs,
        )
    return results


# =========================
# Run
# =========================

if __name__ == "__main__":
    build_common_csv(RACE_FILES, COMMON_CSV)

    results = plot_auc_all_separate(
        race_files=RACE_FILES,
        outdir=OUT_DIR,
        use_metric="SHAP_Normalized",
        limit=None,
        auc_ylim=(0.7, 1.0),
        shap_ylim=(0, 1),
        common_features_csv=COMMON_CSV,
        common_feature_col="Feature",
    )
    print(results)