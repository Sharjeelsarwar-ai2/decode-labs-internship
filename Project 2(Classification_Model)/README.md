# Project 2 — Data Classification Using AI 🌸

**Decode Labs — AI Internship (27 Oct – 27 Nov 2026)**

Build a basic classification model using a small dataset — the supervised learning phase of the internship. Where Project 1 was rules written by hand, here the machine learns the rules from labelled examples.

## Goal

- Load and understand a dataset
- Split data into training and testing sets
- Apply a simple classification algorithm
- Evaluate the model properly and compare algorithms

## Dataset

**Iris flowers** — 150 flowers, 3 species (*setosa*, *versicolor*, *virginica* — 50 each), 4 measurements per flower: sepal length, sepal width, petal length, petal width (cm). Small, clean and balanced — ideal for a first classifier.

## Method

1. **Explore** — load with pandas, check shape / counts / statistics, scatter plot of petal length vs petal width
2. **Split** — 80% train (120) / 20% test (30), `random_state=42`, stratified so each species keeps 10 test flowers
3. **Train** — Decision Tree (`DecisionTreeClassifier`)
4. **Evaluate** — accuracy, confusion matrix, classification report (precision / recall / F1)
5. **Interpret** — print the learned rules, plot the tree, rank feature importances
6. **Compare** — same split, three algorithms
7. **Test** — predict the species of brand-new flowers

## Results

| Model | Test accuracy |
|---|---|
| KNN (k=5) | **1.0000** (30/30) |
| Logistic Regression | 0.9667 (29/30) |
| Decision Tree | 0.9333 (28/30) |

Confusion matrix (Decision Tree): setosa 10/10, versicolor 9/10, virginica 9/10. The two mistakes are versicolor ↔ virginica — the pair that overlaps in the scatter plot.

**Key insight:** the tree's first learned rule is `petal length ≤ 2.45 cm → setosa`. Feature importances: petal length 0.56, petal width 0.41 — the petals do almost all the work; the sepals barely matter.

**Lesson:** no algorithm wins everywhere. KNN wins on accuracy here, but the Decision Tree wins on explainability — it is the only one that can show you its rules.

## Files

- [Classification_Model(Decode_labs) (1).ipynb](Classification_Model%28Decode_labs%29%20%281%29.ipynb) — the full notebook (Google Colab), run top to bottom

## How to run

1. Open the notebook in [Google Colab](https://colab.research.google.com) (File → Upload notebook)
2. Runtime → Run all (or Shift + Enter cell by cell, top to bottom)
3. No installs needed — pandas, scikit-learn and matplotlib come pre-installed in Colab

## Author

Muhammad Sharjeel Sarwar — BSCS, UET Lahore · AI Intern @ Decode Labs
