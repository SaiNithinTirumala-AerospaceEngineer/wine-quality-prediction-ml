"""
data_exploration.py
-------------------
Comprehensive exploratory data analysis of the UCI Red Wine Quality dataset.

Generates:
  1. Quality distribution — class imbalance visualisation
  2. Feature distributions — histogram grid with KDE overlay
  3. Correlation heatmap — feature-feature and feature-quality relationships
  4. Box plots — feature distributions per quality score
  5. Outlier analysis — IQR-based detection per feature

Inputs : data/winequality_red.csv
Outputs: results/quality_distribution.png
         results/feature_distributions.png
         results/correlation_heatmap.png
         results/feature_vs_quality_boxplots.png

Usage:
    python src/data_exploration.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH   = os.path.join(ROOT, "data", "winequality_red.csv")
RESULTS_DIR = os.path.join(ROOT, "results")

FEATURES = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
    'density', 'pH', 'sulphates', 'alcohol'
]
FEATURE_LABELS = [
    'Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar',
    'Chlorides', 'Free SO₂', 'Total SO₂',
    'Density', 'pH', 'Sulphates', 'Alcohol'
]
QUALITY_COLOURS = {
    3: '#E24B4A', 4: '#D85A30', 5: '#BA7517',
    6: '#1D9E75', 7: '#378ADD', 8: '#7F77DD'
}


def plot_quality_distribution(df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Wine Quality Distribution — Class Imbalance Analysis",
                 fontsize=12, fontweight="bold")

    # Bar chart
    counts  = df.quality.value_counts().sort_index()
    colours = [QUALITY_COLOURS[q] for q in counts.index]
    bars = axes[0].bar(counts.index, counts.values, color=colours,
                       edgecolor="white", width=0.7)
    axes[0].bar_label(bars, fmt="%d", padding=4, fontsize=10,
                      fontweight="bold")
    axes[0].set_xlabel("Quality Score", fontsize=11)
    axes[0].set_ylabel("Number of Wines", fontsize=11)
    axes[0].set_title("Sample Count per Quality Score",
                      fontsize=11, fontweight="bold")
    axes[0].set_facecolor("#FAFAFA")
    axes[0].grid(axis="y", alpha=0.25)

    # Cumulative percentage
    cumulative = counts.cumsum() / counts.sum() * 100
    axes[1].bar(counts.index, counts.values / counts.sum() * 100,
                color=colours, edgecolor="white", width=0.7)
    axes[1].plot(counts.index, cumulative.values, "ko-",
                 linewidth=2, markersize=6, label="Cumulative %")
    axes[1].axhline(80, color="grey", linewidth=0.8, linestyle="--",
                    alpha=0.6, label="80% threshold")
    axes[1].set_xlabel("Quality Score", fontsize=11)
    axes[1].set_ylabel("Percentage (%)", fontsize=11)
    axes[1].set_title("Quality Distribution with Cumulative %",
                      fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].set_facecolor("#FAFAFA")
    axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_feature_distributions(df, output_path):
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    fig.suptitle("Physicochemical Feature Distributions — Red Wine Dataset\n"
                 "KDE overlay coloured by quality score",
                 fontsize=12, fontweight="bold")

    for ax, feat, label in zip(axes.flat, FEATURES, FEATURE_LABELS):
        ax.hist(df[feat], bins=30, color="#B5D4F4",
                edgecolor="white", alpha=0.7, density=True)
        df[feat].plot.kde(ax=ax, color="#185FA5", linewidth=2.0)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_facecolor("#FAFAFA")
        ax.grid(True, alpha=0.2)

        # Mark mean and median
        ax.axvline(df[feat].mean(), color="#D85A30", linewidth=1.2,
                   linestyle="--", alpha=0.8, label=f"Mean={df[feat].mean():.2f}")
        ax.axvline(df[feat].median(), color="#1D9E75", linewidth=1.2,
                   linestyle=":", alpha=0.8, label=f"Median={df[feat].median():.2f}")
        ax.legend(fontsize=7)

    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_correlation_heatmap(df, output_path):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.4, ax=ax, annot_kws={"size": 8})
    labels = FEATURE_LABELS + ["Quality"]
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, rotation=0, fontsize=9)
    ax.set_title("Feature Correlation Matrix — Wine Quality Dataset\n"
                 "Bottom row shows correlation with quality target",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_feature_vs_quality(df, output_path):
    """Box plots of top 6 features vs quality score."""
    top_feats = ['alcohol', 'volatile_acidity', 'sulphates',
                 'citric_acid', 'total_sulfur_dioxide', 'chlorides']
    top_labels = ['Alcohol', 'Volatile Acidity', 'Sulphates',
                  'Citric Acid', 'Total SO₂', 'Chlorides']
    qualities  = sorted(df.quality.unique())
    colours    = [QUALITY_COLOURS[q] for q in qualities]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Top 6 Feature Distributions by Quality Score\n"
                 "Features ranked by absolute correlation with quality",
                 fontsize=12, fontweight="bold")

    for ax, feat, label in zip(axes.flat, top_feats, top_labels):
        data = [df[df.quality == q][feat].values for q in qualities]
        bp   = ax.boxplot(data, patch_artist=True,
                          medianprops=dict(color="black", linewidth=1.5))
        for patch, col in zip(bp["boxes"], colours):
            patch.set_facecolor(col); patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(qualities)+1))
        ax.set_xticklabels([str(q) for q in qualities], fontsize=9)
        ax.set_xlabel("Quality Score", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label} vs Quality", fontsize=11, fontweight="bold")
        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="y", alpha=0.25)

        # Trend line through medians
        medians = [df[df.quality==q][feat].median() for q in qualities]
        ax.plot(range(1, len(qualities)+1), medians, "k--",
                linewidth=1.2, alpha=0.5, label="Median trend")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def outlier_analysis(df):
    print("\n  Outlier analysis (IQR method):")
    print(f"  {'Feature':<26} {'Q1':>8} {'Q3':>8} "
          f"{'IQR':>8} {'Outliers':>9} {'%':>6}")
    print("  " + "─" * 68)
    for feat, label in zip(FEATURES, FEATURE_LABELS):
        q1  = df[feat].quantile(0.25)
        q3  = df[feat].quantile(0.75)
        iqr = q3 - q1
        n_out = ((df[feat] < q1-1.5*iqr) | (df[feat] > q3+1.5*iqr)).sum()
        print(f"  {label:<26} {q1:>8.3f} {q3:>8.3f} "
              f"{iqr:>8.3f} {n_out:>9} {n_out/len(df)*100:>5.1f}%")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    print("Wine Quality — Exploratory Data Analysis")
    print(f"  Samples   : {len(df)}")
    print(f"  Features  : {len(FEATURES)}")
    print(f"  Quality range : {df.quality.min()}–{df.quality.max()}")
    print(f"  Mean quality  : {df.quality.mean():.3f}")

    print(f"\n  Quality distribution:")
    for q, cnt in df.quality.value_counts().sort_index().items():
        bar = "█" * (cnt // 30)
        print(f"    {q}: {cnt:>4}  {bar}")

    print(f"\n  Top feature correlations with quality:")
    corr = df.corr()["quality"].drop("quality").sort_values(
        key=abs, ascending=False)
    for feat, val in corr.head(6).items():
        print(f"    {feat:<28} {val:>+.4f}")

    outlier_analysis(df)

    print("\nGenerating plots...")
    plot_quality_distribution(df,
        os.path.join(RESULTS_DIR, "quality_distribution.png"))
    plot_feature_distributions(df,
        os.path.join(RESULTS_DIR, "feature_distributions.png"))
    plot_correlation_heatmap(df,
        os.path.join(RESULTS_DIR, "correlation_heatmap.png"))
    plot_feature_vs_quality(df,
        os.path.join(RESULTS_DIR, "feature_vs_quality_boxplots.png"))

    print("\nData exploration complete. 4 plots saved.")


if __name__ == "__main__":
    main()
