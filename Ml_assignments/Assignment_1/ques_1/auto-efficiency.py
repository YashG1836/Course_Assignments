# classification-exp.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from tree.base import DecisionTree          # your implementation
from metrics import accuracy, precision, recall

np.random.seed(42)

# ---------- Part (a): 70/30 train-test split ----------

X_np, y_np = make_classification(
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=2,
    class_sep=0.5
)

# For plotting
plt.figure()
plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np)
plt.title("Synthetic classification dataset")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

X = pd.DataFrame(X_np, columns=["X1", "X2"])
y = pd.Series(y_np)

# 70/30 split (first 70% train, rest test, as assignment says)
N = X.shape[0]
train_size = int(0.7 * N)

X_train = X.iloc[:train_size].reset_index(drop=True)
y_train = y.iloc[:train_size].reset_index(drop=True)

X_test = X.iloc[train_size:].reset_index(drop=True)
y_test = y.iloc[train_size:].reset_index(drop=True)

# ---- Your manual decision tree ----
my_tree = DecisionTree(criterion="information_gain", max_depth=5)
my_tree.fit(X_train, y_train)
y_hat_my = my_tree.predict(X_test)
y_hat_my = pd.Series(y_hat_my, index=y_test.index)

print("=== My Decision Tree (70/30 split) ===")
print("Accuracy:", accuracy(y_hat_my, y_test))
for cls in sorted(y_test.unique()):
    print(f"Class {cls}: Precision = {precision(y_hat_my, y_test, cls)}, "
          f"Recall = {recall(y_hat_my, y_test, cls)}")

# ---- sklearn DecisionTreeClassifier on same split ----
sk_tree = DecisionTreeClassifier(
    criterion="gini",      # or "entropy"
    max_depth=5,
    random_state=42
)
sk_tree.fit(X_train.values, y_train.values)
y_hat_sk = sk_tree.predict(X_test.values)
y_hat_sk = pd.Series(y_hat_sk, index=y_test.index)

print("\n=== sklearn DecisionTreeClassifier (70/30 split) ===")
print("Accuracy:", accuracy(y_hat_sk, y_test))
for cls in sorted(y_test.unique()):
    print(f"Class {cls}: Precision = {precision(y_hat_sk, y_test, cls)}, "
          f"Recall = {recall(y_hat_sk, y_test, cls)}")
