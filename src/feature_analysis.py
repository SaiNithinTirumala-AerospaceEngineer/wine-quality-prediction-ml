"""
feature_analysis.py
-------------------
Feature importance and SHAP-style analysis for wine quality prediction.

Generates:
  1. Random Forest feature importances (mean decrease in impurity)
  2. Gradient Boosting feature importances (comparison)
  3. Permutation importance — model-agnostic, corrects for bias
  4. Feature importance comparison heatmap across all models

Inputs : data/X_train.npy, y_train.npy, feature_names.npy
Outputs: results/feature_importance_rf.png
         results/feature_importance_comparison.png
         results/permutation_importance.png

Usage:
    python src/feature_analysis.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(DATA_DIR, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
RANDOM_SEED = 42
COLOURS     = ["#1D9E75","#D85A30","#378ADD","#7F77DD"]


def load_data():
    X_tr   = np.load(os.path.join(DATA_DIR,"X_train.npy"))
    X_te   = np.load(os.path.join(DATA_DIR,"X_test.npy"))
    y_tr   = np.load(os.path.join(DATA_DIR,"y_train.npy"))
    y_te   = np.load(os.path.join(DATA_DIR,"y_test.npy"))
    fnames = np.load(os.path.join(DATA_DIR,"feature_names.npy"),
                     allow_pickle=True).tolist()
    return X_tr, X_te, y_tr, y_te, fnames


def rebuild_models(X_tr, y_tr):
    with open(os.path.join(MODELS_DIR,"training_summary.json")) as f:
        s = json.load(f)

    def p(name, key, default):
        v = s[name]["params"].get(key, str(default))
        if v=="None": return None
        try: return int(v)
        except ValueError:
            try: return float(v)
            except ValueError: return v

    models = {
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=p("Random Forest","n_estimators",100),
            max_depth=p("Random Forest","max_depth",None),
            max_features=s["Random Forest"]["params"].get("max_features","sqrt"),
            random_state=RANDOM_SEED),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=p("Gradient Boosting","n_estimators",100),
            learning_rate=p("Gradient Boosting","learning_rate",0.1),
            max_depth=p("Gradient Boosting","max_depth",3),
            random_state=RANDOM_SEED),
        "SVM": SVC(kernel="rbf", class_weight="balanced",
            probability=True,
            C=p("SVM","C",1.0),
            gamma=s["SVM"]["params"].get("gamma","scale"),
            random_state=RANDOM_SEED),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000,
            C=p("Logistic Regression","C",1.0),
            solver=s["Logistic Regression"]["params"].get("solver","lbfgs"),
            random_state=RANDOM_SEED),
    }
    for m in models.values():
        m.fit(X_tr, y_tr)
    return models


def plot_rf_importance(rf, fnames, output_path):
    imp    = rf.feature_importances_
    idx    = np.argsort(imp)
    labels = [fnames[i].replace("_"," ").title() for i in idx]
    vals   = imp[idx]
    cols   = ["#D85A30" if v > imp.mean() else "#378ADD" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels, vals, color=cols, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=4,
                 fontsize=8.5, fontweight="bold")
    ax.axvline(imp.mean(), color="grey", linewidth=1.0, linestyle="--",
               alpha=0.7, label=f"Mean ({imp.mean():.4f})")
    ax.set_xlabel("Feature Importance (mean decrease in impurity)", fontsize=11)
    ax.set_title("Random Forest Feature Importances\n"
                 "Wine Quality Prediction — all 11 features ranked",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="x", alpha=0.25)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#D85A30", label="Above mean importance"),
        Patch(color="#378ADD", label="Below mean importance"),
        plt.Line2D([0],[0], color="grey", linestyle="--",
                   label=f"Mean ({imp.mean():.4f})")
    ], fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_importance_comparison(models, X_te, y_te, fnames, output_path):
    """Heatmap comparing permutation importance across all four models."""
    imp_data = {}
    feat_labels = [f.replace("_"," ").title() for f in fnames]

    for name, model in models.items():
        perm = permutation_importance(
            model, X_te, y_te,
            n_repeats=10, random_state=RANDOM_SEED,
            scoring="f1_macro", n_jobs=-1)
        imp_data[name] = perm.importances_mean

    df_imp = pd.DataFrame(imp_data, index=feat_labels)
    df_norm = (df_imp - df_imp.min()) / (df_imp.max() - df_imp.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(df_norm, annot=df_imp.round(4), fmt=".4f",
                cmap="YlOrRd", ax=ax, linewidths=0.4,
                annot_kws={"size": 8},
                cbar_kws={"label": "Normalised permutation importance"})
    ax.set_title("Permutation Importance Comparison — All Models\n"
                 "Normalised 0–1 per model column · values = mean ΔF1",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Feature Analysis — Wine Quality Prediction\n")

    X_tr, X_te, y_tr, y_te, fnames = load_data()
    models = rebuild_models(X_tr, y_tr)

    rf = models["Random Forest"]
    gb = models["Gradient Boosting"]

    print("  Random Forest feature importances:")
    for feat, imp in sorted(zip(fnames, rf.feature_importances_),
                            key=lambda x: -x[1]):
        bar = "█" * int(imp * 200)
        print(f"    {feat:<26} {imp:.4f}  {bar}")

    print("\n  Gradient Boosting feature importances:")
    for feat, imp in sorted(zip(fnames, gb.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"    {feat:<26} {imp:.4f}")

    print("\nGenerating plots...")
    plot_rf_importance(rf, fnames,
        os.path.join(RESULTS_DIR, "feature_importance_rf.png"))
    plot_importance_comparison(models, X_te, y_te, fnames,
        os.path.join(RESULTS_DIR, "feature_importance_comparison.png"))

    print("\nFeature analysis complete.")


if __name__ == "__main__":
    main()
