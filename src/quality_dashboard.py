"""
quality_dashboard.py
--------------------
Six-panel wine quality prediction dashboard — the hero README figure.

Panels:
  1. CV F1 + Accuracy comparison — four models
  2. Test metrics table
  3. Random Forest confusion matrix (best model)
  4. Feature importance — RF top 11
  5. Quality distribution — actual vs RF predicted
  6. Pipeline summary

Inputs : data/ (all processed arrays)
Outputs: results/quality_dashboard.png

Usage:
    python src/quality_dashboard.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             f1_score)

ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(DATA_DIR, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
RANDOM_SEED = 42
COLOURS     = ["#1D9E75","#D85A30","#378ADD","#7F77DD"]


def load_and_rebuild():
    X_tr   = np.load(os.path.join(DATA_DIR,"X_train.npy"))
    X_te   = np.load(os.path.join(DATA_DIR,"X_test.npy"))
    y_tr   = np.load(os.path.join(DATA_DIR,"y_train.npy"))
    y_te   = np.load(os.path.join(DATA_DIR,"y_test.npy"))
    fnames = np.load(os.path.join(DATA_DIR,"feature_names.npy"),
                     allow_pickle=True).tolist()

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
        "SVM": SVC(kernel="rbf", class_weight="balanced", probability=True,
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

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_te)
        results[name] = {
            "y_pred":  y_pred,
            "acc":     accuracy_score(y_te, y_pred),
            "f1_mac":  f1_score(y_te, y_pred, average="macro"),
            "f1_wt":   f1_score(y_te, y_pred, average="weighted"),
            "cv_f1":   s[name]["cv_f1_mean"],
            "cv_std":  s[name]["cv_f1_std"],
            "cv_acc":  s[name]["cv_acc_mean"],
        }

    return models, results, X_tr, X_te, y_tr, y_te, fnames, s


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Generating wine quality dashboard...")

    models, results, X_tr, X_te, y_tr, y_te, fnames, s = load_and_rebuild()
    names   = list(results.keys())
    classes = sorted(np.unique(y_te))

    fig = plt.figure(figsize=(19, 12))
    fig.suptitle(
        "Wine Quality Prediction Dashboard — Four ML Model Comparison\n"
        "Bharat Intern Machine Learning Internship  ·  Aug–Sep 2023  ·  "
        "UCI Red Wine Quality Dataset (1599 samples, 11 features)",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, wspace=0.38, hspace=0.45)

    # ── Panel 1: CV F1 + Accuracy ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0,0])
    x   = np.arange(len(names))
    w   = 0.38
    b1  = ax1.bar(x-w/2, [results[n]["cv_f1"]  for n in names], w,
                  label="CV Macro F1", color=COLOURS, edgecolor="white")
    b2  = ax1.bar(x+w/2, [results[n]["cv_acc"] for n in names], w,
                  label="CV Accuracy",
                  color=[c+"88" for c in COLOURS], edgecolor="white")
    ax1.errorbar(x-w/2, [results[n]["cv_f1"] for n in names],
                 yerr=[results[n]["cv_std"] for n in names],
                 fmt="none", color="black", capsize=4, linewidth=1.2)
    ax1.bar_label(b1, fmt="%.3f", padding=3, fontsize=8, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace(" ","\n") for n in names], fontsize=7.5)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_title("CV Macro F1 vs Accuracy", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8); ax1.set_ylim(0,0.95)
    ax1.grid(axis="y",alpha=0.25); ax1.set_facecolor("#FAFAFA")

    # ── Panel 2: Test metrics table ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0,1])
    ax2.axis("off")
    td = [[n, f"{results[n]['acc']:.4f}",
           f"{results[n]['f1_mac']:.4f}",
           f"{results[n]['f1_wt']:.4f}"] for n in names]
    tbl = ax2.table(cellText=td,
                    colLabels=["Model","Acc","Macro F1","Wtd F1"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1,1.9)
    for (r,c), cell in tbl.get_celld().items():
        if r==0:
            cell.set_facecolor("#1D9E75")
            cell.set_text_props(color="white", fontweight="bold")
        elif r%2==0: cell.set_facecolor("#EDF7F3")
    ax2.set_title("Test Set Metrics", fontsize=11, fontweight="bold", pad=60)

    # ── Panel 3: RF confusion matrix ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[0,2])
    best = max(results, key=lambda k: results[k]["f1_mac"])
    cm = confusion_matrix(y_te, results[best]["y_pred"], labels=classes)
    im = ax3.imshow(cm, cmap="Greens")
    plt.colorbar(im, ax=ax3, shrink=0.85)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax3.text(j,i,str(cm[i,j]),ha="center",va="center",
                     fontsize=8,fontweight="bold",
                     color="white" if cm[i,j]>cm.max()*0.6 else "black")
    ax3.set_xticks(range(len(classes)))
    ax3.set_yticks(range(len(classes)))
    ax3.set_xticklabels(classes, fontsize=9)
    ax3.set_yticklabels(classes, fontsize=9)
    ax3.set_xlabel("Predicted", fontsize=10)
    ax3.set_ylabel("Actual", fontsize=10)
    ax3.set_title(f"Confusion Matrix — {best}\nMacro F1={results[best]['f1_mac']:.4f}",
                  fontsize=11, fontweight="bold")

    # ── Panel 4: RF feature importance ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1,0])
    rf  = models["Random Forest"]
    imp = rf.feature_importances_
    idx = np.argsort(imp)
    feat_labels = [fnames[i].replace("_"," ").title() for i in idx]
    ax4.barh(feat_labels, imp[idx], color="#1D9E75", edgecolor="white")
    ax4.set_xlabel("Importance", fontsize=10)
    ax4.set_title("RF Feature Importances", fontsize=11, fontweight="bold")
    ax4.set_facecolor("#FAFAFA"); ax4.grid(axis="x",alpha=0.25)

    # ── Panel 5: Quality distribution actual vs predicted ─────────────────
    ax5 = fig.add_subplot(gs[1,1])
    q_colours = {3:"#E24B4A",4:"#D85A30",5:"#BA7517",
                 6:"#1D9E75",7:"#378ADD",8:"#7F77DD"}
    actual_counts = pd.Series(y_te).value_counts().sort_index()
    pred_counts   = pd.Series(results[best]["y_pred"]).value_counts().sort_index()
    all_q = sorted(set(list(actual_counts.index)+list(pred_counts.index)))
    xq = np.arange(len(all_q)); wq=0.38
    ax5.bar(xq-wq/2, [actual_counts.get(q,0) for q in all_q], wq,
            label="Actual", color=[q_colours.get(q,"#888780") for q in all_q],
            edgecolor="white")
    ax5.bar(xq+wq/2, [pred_counts.get(q,0) for q in all_q], wq,
            label="RF Predicted",
            color=[q_colours.get(q,"#888780")+"88" for q in all_q],
            edgecolor="white")
    ax5.set_xticks(xq)
    ax5.set_xticklabels([f"Q{q}" for q in all_q], fontsize=9)
    ax5.set_ylabel("Count", fontsize=10)
    ax5.set_title("Quality Distribution\nActual vs Predicted",
                  fontsize=11, fontweight="bold")
    ax5.legend(fontsize=9); ax5.grid(axis="y",alpha=0.25)
    ax5.set_facecolor("#FAFAFA")

    # ── Panel 6: Summary ──────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1,2]); ax6.axis("off")
    metrics_list = [
        ("Dataset",          "UCI Red Wine Quality"),
        ("Samples",          "1599 (1279 train / 320 test)"),
        ("Features",         "11 physicochemical"),
        ("Classes",          "6 (quality scores 3–8)"),
        ("Class imbalance",  "Yes — balanced weighting used"),
        ("Models",           "4"),
        ("Best model",       best),
        ("Best Macro F1",    f"{results[best]['f1_mac']:.4f}"),
        ("Best Accuracy",    f"{results[best]['acc']:.4f}"),
        ("CV strategy",      "5-fold stratified"),
        ("Key feature",      "Alcohol (top importance)"),
        ("Internship",       "Bharat Intern ML · 2023"),
    ]
    for i,(label,value) in enumerate(metrics_list):
        y_pos = 0.95 - i*0.077
        ax6.text(0.02,y_pos,label+":",transform=ax6.transAxes,
                 fontsize=9,color="#555555")
        ax6.text(0.52,y_pos,value,transform=ax6.transAxes,
                 fontsize=9,fontweight="bold",color="#1A1A1A")
    ax6.set_title("Pipeline Summary",fontsize=11,fontweight="bold")
    ax6.add_patch(plt.Rectangle((0,0),1,1,fill=False,
                                 edgecolor="#CCCCCC",linewidth=1,
                                 transform=ax6.transAxes))

    out = os.path.join(RESULTS_DIR,"quality_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    print("\nQuality dashboard complete.")


if __name__ == "__main__":
    main()
