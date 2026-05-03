"""
model_evaluation.py
-------------------
Evaluate all four models on the held-out test set.

Metrics:
  - Accuracy, Macro F1, Weighted F1
  - Per-class precision, recall, F1
  - Confusion matrices
  - ROC curves (one-vs-rest)

Inputs : data/X_train.npy, X_test.npy, y_train.npy, y_test.npy
Outputs: results/confusion_matrices.png
         results/per_class_metrics.png
         results/roc_curves.png

Usage:
    python src/model_evaluation.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, accuracy_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize

ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(DATA_DIR, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
RANDOM_SEED = 42
COLOURS     = ["#1D9E75","#D85A30","#378ADD","#7F77DD"]


def load_data():
    return (np.load(os.path.join(DATA_DIR,"X_train.npy")),
            np.load(os.path.join(DATA_DIR,"X_test.npy")),
            np.load(os.path.join(DATA_DIR,"y_train.npy")),
            np.load(os.path.join(DATA_DIR,"y_test.npy")))


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
            min_samples_split=p("Random Forest","min_samples_split",2),
            max_features=s["Random Forest"]["params"].get("max_features","sqrt"),
            random_state=RANDOM_SEED),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=p("Gradient Boosting","n_estimators",100),
            learning_rate=p("Gradient Boosting","learning_rate",0.1),
            max_depth=p("Gradient Boosting","max_depth",3),
            min_samples_split=p("Gradient Boosting","min_samples_split",2),
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


def plot_confusion_matrices(models, X_te, y_te, output_path):
    classes = sorted(np.unique(y_te))
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Confusion Matrices — Test Set\nWine Quality Classification",
                 fontsize=13, fontweight="bold")

    for ax, (name, model), colour in zip(axes.flat, models.items(), COLOURS):
        cm  = confusion_matrix(y_te, model.predict(X_te), labels=classes)
        acc = accuracy_score(y_te, model.predict(X_te))
        f1  = f1_score(y_te, model.predict(X_te), average="macro")

        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if cm[i,j]>cm.max()*0.6 else "black")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlabel("Predicted Quality", fontsize=10)
        ax.set_ylabel("Actual Quality", fontsize=10)
        ax.set_title(f"{name}\nAcc={acc:.4f}  Macro F1={f1:.4f}",
                     fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_per_class_metrics(models, X_te, y_te, output_path):
    classes = sorted(np.unique(y_te))
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Per-Class Precision / Recall / F1 — All Models",
                 fontsize=12, fontweight="bold")

    x = np.arange(len(classes))
    width = 0.2

    for ax, metric, title in zip(axes,
            ["precision","recall","f1-score"],
            ["Precision","Recall","F1 Score"]):
        for i, (name, model) in enumerate(models.items()):
            report = classification_report(
                y_te, model.predict(X_te),
                labels=classes, output_dict=True)
            vals = [report[str(c)][metric] for c in classes]
            ax.bar(x + i*width, vals, width,
                   label=name, color=COLOURS[i],
                   edgecolor="white", alpha=0.85)
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels([f"Q{c}" for c in classes], fontsize=9)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"Per-Class {title}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_roc_curves(models, X_te, y_te, output_path):
    classes  = sorted(np.unique(y_te))
    y_bin    = label_binarize(y_te, classes=classes)
    cls_cols = ["#E24B4A","#D85A30","#BA7517",
                "#1D9E75","#378ADD","#7F77DD"][:len(classes)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("ROC Curves — One-vs-Rest per Quality Class",
                 fontsize=13, fontweight="bold")

    for ax, (name, model), m_col in zip(
            axes.flat, models.items(), COLOURS):
        y_prob = model.predict_proba(X_te)
        for i, (cls, col) in enumerate(zip(classes, cls_cols)):
            fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob[:,i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=col, linewidth=1.8,
                    label=f"Q{cls} (AUC={roc_auc:.3f})")
        ax.plot([0,1],[0,1],"k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.set_facecolor("#FAFAFA")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Model Evaluation — Wine Quality Prediction\n")

    X_tr, X_te, y_tr, y_te = load_data()
    models = rebuild_models(X_tr, y_tr)

    print(f"  {'Model':<22} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print("  " + "─" * 58)
    for name, model in models.items():
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        mf1 = f1_score(y_te, y_pred, average="macro")
        wf1 = f1_score(y_te, y_pred, average="weighted")
        print(f"  {name:<22} {acc:>10.4f} {mf1:>10.4f} {wf1:>12.4f}")

    print("\nGenerating plots...")
    plot_confusion_matrices(models, X_te, y_te,
        os.path.join(RESULTS_DIR, "confusion_matrices.png"))
    plot_per_class_metrics(models, X_te, y_te,
        os.path.join(RESULTS_DIR, "per_class_metrics.png"))
    plot_roc_curves(models, X_te, y_te,
        os.path.join(RESULTS_DIR, "roc_curves.png"))

    print("\nModel evaluation complete.")


if __name__ == "__main__":
    main()
