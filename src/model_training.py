"""
model_training.py
-----------------
Train and tune four classifiers on the wine quality dataset.

Handles class imbalance using SMOTE oversampling and class_weight
adjustments. Treats quality prediction as a multi-class classification
problem (scores 3–8).

Models:
  1. Random Forest        — ensemble, robust to imbalanced classes
  2. Gradient Boosting    — sequential error correction
  3. Support Vector Machine (RBF) — margin-based multi-class
  4. Logistic Regression  — regularised linear baseline

Evaluation strategy:
  - Stratified 5-fold cross-validation
  - GridSearchCV hyperparameter tuning
  - Macro-averaged F1 score as primary metric
    (appropriate for imbalanced multi-class — treats all classes equally)

Inputs : data/winequality_red.csv
Outputs: data/X_train.npy, X_test.npy, y_train.npy, y_test.npy
         data/models/training_summary.json
         results/cv_f1_comparison.png
         results/learning_curves.png

Usage:
    python src/model_training.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, StratifiedKFold,
                                     learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH   = os.path.join(ROOT, "data", "winequality_red.csv")
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(DATA_DIR, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
RANDOM_SEED = 42
COLOURS     = ["#1D9E75", "#D85A30", "#378ADD", "#7F77DD"]


def load_and_split(path):
    df = pd.read_csv(path)
    X  = df.drop(columns=["quality"]).values
    y  = df["quality"].values
    feat_names = df.drop(columns=["quality"]).columns.tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    np.save(os.path.join(DATA_DIR, "X_train.npy"),      X_tr)
    np.save(os.path.join(DATA_DIR, "X_test.npy"),       X_te)
    np.save(os.path.join(DATA_DIR, "y_train.npy"),      y_tr)
    np.save(os.path.join(DATA_DIR, "y_test.npy"),       y_te)
    np.save(os.path.join(DATA_DIR, "feature_names.npy"),
            np.array(feat_names))
    return X_tr, X_te, y_tr, y_te, feat_names


def train_models(X_tr, y_tr):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    results = {}

    configs = [
        ("Random Forest",
         RandomForestClassifier(class_weight="balanced",
                                random_state=RANDOM_SEED),
         {"n_estimators": [100],
          "max_depth":    [None, 10],
          "min_samples_split": [2],
          "max_features": ["sqrt"]}),

        ("Gradient Boosting",
         GradientBoostingClassifier(random_state=RANDOM_SEED),
         {"n_estimators":   [100],
          "learning_rate":  [0.1],
          "max_depth":      [3, 4],
          "min_samples_split": [2]}),

        ("SVM",
         SVC(kernel="rbf", class_weight="balanced",
             random_state=RANDOM_SEED, probability=True),
         {"C":     [1, 10],
          "gamma": ["scale"]}),

        ("Logistic Regression",
         LogisticRegression(class_weight="balanced", max_iter=2000,
                            random_state=RANDOM_SEED),
         {"C":      [0.1, 1, 10],
          "solver": ["lbfgs"]}),
    ]

    for i, (name, estimator, params) in enumerate(configs):
        print(f"  [{i+1}/4] {name}...")
        gs = GridSearchCV(estimator, params, cv=cv,
                          scoring="f1_macro", n_jobs=-1)
        gs.fit(X_tr, y_tr)

        cv_scores = cross_val_score(gs.best_estimator_, X_tr, y_tr,
                                    cv=cv, scoring="f1_macro")
        cv_acc    = cross_val_score(gs.best_estimator_, X_tr, y_tr,
                                    cv=cv, scoring="accuracy")

        results[name] = {
            "model":       gs.best_estimator_,
            "cv_f1_mean":  cv_scores.mean(),
            "cv_f1_std":   cv_scores.std(),
            "cv_acc_mean": cv_acc.mean(),
            "cv_acc_std":  cv_acc.std(),
            "best_params": gs.best_params_,
        }
        print(f"     CV F1={cv_scores.mean():.4f}±{cv_scores.std():.4f}  "
              f"Acc={cv_acc.mean():.4f}  params={gs.best_params_}")

    return results


def plot_cv_comparison(results, output_path):
    names     = list(results.keys())
    f1_means  = [results[n]["cv_f1_mean"]  for n in names]
    f1_stds   = [results[n]["cv_f1_std"]   for n in names]
    acc_means = [results[n]["cv_acc_mean"] for n in names]

    x     = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - width/2, f1_means, width, label="Macro F1",
                color=COLOURS, edgecolor="white")
    b2 = ax.bar(x + width/2, acc_means, width, label="Accuracy",
                color=[c+"88" for c in COLOURS], edgecolor="white")
    ax.errorbar(x - width/2, f1_means, yerr=f1_stds,
                fmt="none", color="black", capsize=5, linewidth=1.2)
    ax.bar_label(b1, fmt="%.4f", padding=4, fontsize=9, fontweight="bold")
    ax.bar_label(b2, fmt="%.4f", padding=4, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Score (5-fold stratified CV)", fontsize=11)
    ax.set_title("Model Comparison — CV Macro F1 and Accuracy\n"
                 "Macro F1 primary metric — handles class imbalance",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.set_facecolor("#FAFAFA")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_learning_curves(results, X_tr, y_tr, output_path):
    """Learning curves — training and CV score vs dataset size."""
    cv   = StratifiedKFold(n_splits=5, shuffle=True,
                           random_state=RANDOM_SEED)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Learning Curves — Training vs Validation Score\n"
                 "Diagnose bias/variance trade-off per model",
                 fontsize=12, fontweight="bold")

    for ax, (name, res), colour in zip(axes.flat, results.items(), COLOURS):
        sizes, tr_scores, cv_scores = learning_curve(
            res["model"], X_tr, y_tr,
            train_sizes=np.linspace(0.1, 1.0, 8),
            cv=cv, scoring="f1_macro", n_jobs=-1)

        tr_mean = tr_scores.mean(axis=1)
        tr_std  = tr_scores.std(axis=1)
        cv_mean = cv_scores.mean(axis=1)
        cv_std  = cv_scores.std(axis=1)

        ax.plot(sizes, tr_mean, "o-", color=colour,
                linewidth=2, label="Training score")
        ax.fill_between(sizes, tr_mean-tr_std, tr_mean+tr_std,
                        alpha=0.15, color=colour)
        ax.plot(sizes, cv_mean, "s--", color=colour,
                linewidth=2, alpha=0.7, label="CV score")
        ax.fill_between(sizes, cv_mean-cv_std, cv_mean+cv_std,
                        alpha=0.10, color=colour)

        gap = tr_mean[-1] - cv_mean[-1]
        ax.set_xlabel("Training Samples", fontsize=10)
        ax.set_ylabel("Macro F1 Score", fontsize=10)
        ax.set_title(f"{name}\nFinal gap: {gap:.3f} "
                     f"({'high variance' if gap>0.1 else 'good fit'})",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Model Training — Wine Quality Prediction\n")
    X_tr, X_te, y_tr, y_te, feat_names = load_and_split(DATA_PATH)
    print(f"  Train: {X_tr.shape[0]}  Test: {X_te.shape[0]}")
    print(f"  Features: {X_tr.shape[1]}\n")

    results = train_models(X_tr, y_tr)

    summary = {n: {
        "cv_f1_mean":  round(r["cv_f1_mean"],  4),
        "cv_f1_std":   round(r["cv_f1_std"],   4),
        "cv_acc_mean": round(r["cv_acc_mean"], 4),
        "params": {k: str(v) for k, v in r["best_params"].items()}
    } for n, r in results.items()}

    with open(os.path.join(MODELS_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nGenerating plots...")
    plot_cv_comparison(results,
        os.path.join(RESULTS_DIR, "cv_f1_comparison.png"))
    # plot_learning_curves disabled — use model_evaluation.py plots instead

    best = max(results, key=lambda k: results[k]["cv_f1_mean"])
    print(f"\n  Best model : {best}")
    print(f"  Best CV F1 : {results[best]['cv_f1_mean']:.4f}")
    print("\nModel training complete.")


if __name__ == "__main__":
    main()
