# classification-exp.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

from tree.base import DecisionTree
from metrics import accuracy, precision, recall

np.random.seed(42)

# ---------- Part (a): Single train-test split ----------

# Generate dataset
X_np, y_np = make_classification(
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=2,
    class_sep=0.5
)

# Plot the data (optional, for report)
plt.figure()
plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np)
plt.title("Synthetic classification dataset")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# Convert to pandas
X = pd.DataFrame(X_np, columns=["X1", "X2"])
y = pd.Series(y_np)

# Train-test split: first 70% train, remaining 30% test
N = X.shape[0]
train_size = int(0.7 * N)

X_train = X.iloc[:train_size].reset_index(drop=True)
y_train = y.iloc[:train_size].reset_index(drop=True)

X_test = X.iloc[train_size:].reset_index(drop=True)
y_test = y.iloc[train_size:].reset_index(drop=True)

# Train your decision tree
tree = DecisionTree(criterion="information_gain", max_depth=5)
tree.fit(X_train, y_train)

# Predict on test set
y_hat_test = tree.predict(X_test)

# Convert to Series if not already
y_hat_test = pd.Series(y_hat_test)

# Compute metrics
acc = accuracy(y_hat_test, y_test)
print("=== Part (a): Single Train/Test Split ===")
print("Accuracy on test set:", acc)

for cls in sorted(y_test.unique()):
    prec = precision(y_hat_test, y_test, cls)
    rec = recall(y_hat_test, y_test, cls)
    print(f"Class {cls}: Precision = {prec}, Recall = {rec}")


# ---------- Part (b): Nested 5-fold Cross-Validation ----------

print("\n=== Part (b): Nested 5-fold Cross-Validation ===")

X_all = X.reset_index(drop=True)
y_all = y.reset_index(drop=True)

outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
inner_kf = KFold(n_splits=5, shuffle=True, random_state=123)

depth_candidates = [1, 2, 3, 4, 5, 6, 7, 8]

outer_fold_idx = 1
best_depths = []
outer_accuracies = []

for outer_train_idx, outer_val_idx in outer_kf.split(X_all):
    X_outer_train = X_all.iloc[outer_train_idx].reset_index(drop=True)
    y_outer_train = y_all.iloc[outer_train_idx].reset_index(drop=True)

    X_outer_val = X_all.iloc[outer_val_idx].reset_index(drop=True)
    y_outer_val = y_all.iloc[outer_val_idx].reset_index(drop=True)

    # ----- Inner loop: choose best depth -----
    best_depth = None
    best_inner_score = -np.inf

    for depth in depth_candidates:
        inner_scores = []

        for inner_train_idx, inner_val_idx in inner_kf.split(X_outer_train):
            X_inner_train = X_outer_train.iloc[inner_train_idx].reset_index(drop=True)
            y_inner_train = y_outer_train.iloc[inner_train_idx].reset_index(drop=True)

            X_inner_val = X_outer_train.iloc[inner_val_idx].reset_index(drop=True)
            y_inner_val = y_outer_train.iloc[inner_val_idx].reset_index(drop=True)

            tree_inner = DecisionTree(criterion="information_gain", max_depth=depth)
            tree_inner.fit(X_inner_train, y_inner_train)
            y_inner_pred = tree_inner.predict(X_inner_val)
            y_inner_pred = pd.Series(y_inner_pred)

            inner_acc = accuracy(y_inner_pred, y_inner_val)
            inner_scores.append(inner_acc)

        mean_inner_acc = np.mean(inner_scores)
        if mean_inner_acc > best_inner_score:
            best_inner_score = mean_inner_acc
            best_depth = depth

    best_depths.append(best_depth)

    # ----- Train final model on outer train using best depth -----
    final_tree = DecisionTree(criterion="information_gain", max_depth=best_depth)
    final_tree.fit(X_outer_train, y_outer_train)
    y_outer_pred = final_tree.predict(X_outer_val)
    y_outer_pred = pd.Series(y_outer_pred)

    outer_acc = accuracy(y_outer_pred, y_outer_val)
    outer_accuracies.append(outer_acc)

    print(f"Outer fold {outer_fold_idx}: best_depth = {best_depth}, accuracy = {outer_acc}")
    outer_fold_idx += 1

print("\nAverage outer accuracy:", np.mean(outer_accuracies))
print("Chosen depths per outer fold:", best_depths)
print("Most frequently chosen depth (for report):", pd.Series(best_depths).mode().iloc[0])
