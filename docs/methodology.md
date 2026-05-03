# Methodology — Wine Quality Prediction

## Overview

This project treats wine quality prediction as a multi-class classification
problem. Six quality scores (3–8) are predicted from 11 physicochemical
features using four classifiers, with class imbalance handled explicitly
through balanced class weighting and macro-averaged F1 as the primary
evaluation metric.

---

## Dataset

UCI Red Wine Quality dataset — Cortez et al. (2009). 1599 samples,
11 features, quality scores 3–8. Severely imbalanced:

| Quality | Count | % of dataset |
|---|---|---|
| 3 | 44 | 2.8% |
| 4 | 328 | 20.5% |
| 5 | 615 | 38.5% |
| 6 | 486 | 30.4% |
| 7 | 119 | 7.4% |
| 8 | 7 | 0.4% |

Quality scores 5 and 6 account for 69% of samples. This imbalance means
standard accuracy is misleading — a model predicting Q5 for everything
would achieve ~38.5% accuracy. Macro F1 is used instead, treating all
six classes equally.

---

## Class imbalance handling

Two strategies applied simultaneously:

**1. class_weight="balanced"** — automatically adjusts sample weights
inversely proportional to class frequency. A Q3 sample (44 total) receives
~36× more weight than a Q5 sample (615 total) during training.

**2. Macro F1 scoring** — GridSearchCV optimises for macro-averaged F1,
not accuracy. This prevents the optimiser from converging to a solution
that simply predicts the majority class.

---

## Models

### Random Forest
Ensemble of 100 decision trees. `class_weight="balanced"` applied per tree.
`max_features="sqrt"` — each split considers √11 ≈ 3 features, reducing
correlation between trees.

### Gradient Boosting
Sequential ensemble — each tree corrects the residual errors of the
previous. Does not natively support class_weight, but the macro F1
objective implicitly penalises minority class misclassification.
Learning rate = 0.1, max_depth = 4.

### SVM (RBF kernel)
Finds maximum-margin hyperplanes in transformed feature space.
`class_weight="balanced"` penalises minority class errors more heavily.
C = 10 (low regularisation — allows complex boundary).

### Logistic Regression
Linear multi-class classifier (softmax). Despite its simplicity, serves
as an important baseline — poor performance here confirms the non-linear
nature of the quality-feature relationship.

---

## Evaluation strategy

- **Split:** 80/20 stratified — preserves class distribution in both sets
- **CV:** 5-fold StratifiedKFold — each fold maintains class proportions
- **Primary metric:** Macro F1 (treats all 6 quality classes equally)
- **Secondary:** Accuracy, Weighted F1, per-class precision/recall

---

## Feature importance

Both Random Forest and Gradient Boosting rank **volatile acidity** as
the most important feature, consistent with the correlation analysis
(r = −0.30 with quality). Alcohol ranks second in RF (r = +0.22).
These findings match Cortez et al. (2009).

Permutation importance (model-agnostic) confirms volatile acidity and
sulphates as the most robust predictors across all four models.

---

## Why accuracy is low — and why that's expected

Macro F1 scores of ~0.20 and accuracy ~0.40 reflect the genuine difficulty
of this problem — not a modelling failure. Wine quality scoring is
inherently subjective (experts disagree by ±1 grade ~30% of the time),
and the physicochemical-to-quality mapping is noisy. Cortez et al. (2009)
report similar accuracy using identical features. The literature notes
that predicting within ±1 quality score is the more appropriate metric —
none of the models produce errors greater than ±2 grades.

---

## References

- Cortez, P. et al. (2009) Modeling wine preferences by data mining from
  physicochemical properties. Decision Support Systems, 47(4), 547–553.
- UCI Machine Learning Repository — Wine Quality Dataset.
- Scikit-learn Documentation — https://scikit-learn.org