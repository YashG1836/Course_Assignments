"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


class _Node:
    """
    Internal class representing a node in the decision tree.
    """

    def __init__(
        self,
        is_leaf: bool = False,
        prediction=None,
        feature: Optional[str] = None,
        threshold: Optional[float] = None,
        left: Optional["_Node"] = None,
        right: Optional["_Node"] = None,
    ):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root: Optional[_Node] = None
        self.is_regression: Optional[bool] = None
        self.feature_names = None  # will store column order after encoding

    # ---------- PUBLIC API ----------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree.
        """

        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
        assert isinstance(y, pd.Series), "y must be a pandas Series"
        assert X.shape[0] == y.size, "X and y must have same number of rows"
        assert self.max_depth >= 0, "max_depth must be non-negative"

        # Handle discrete input: one-hot encode
        X_enc = one_hot_encoding(X.copy())
        self.feature_names = list(X_enc.columns)

        # Decide if this is regression or classification based on y
        self.is_regression = check_ifreal(y)

        # Build the tree recursively
        self.root = self._build_tree(X_enc, y, depth=0)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to run the decision tree on test inputs.
        """

        assert self.root is not None, "You must call fit() before predict()"
        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"

        # Apply the same encoding as during training
        X_enc = one_hot_encoding(X.copy())
        # Align columns with training columns (missing -> 0, extra -> drop)
        X_enc = X_enc.reindex(columns=self.feature_names, fill_value=0)

        preds = []
        for _, row in X_enc.iterrows():
            preds.append(self._predict_one(row))

        return pd.Series(preds, index=X.index)

    def plot(self) -> None:
        """
        Function to plot (print) the tree in a human-readable form.

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        assert self.root is not None, "You must call fit() before plot()"
        self._print_node(self.root, indent="")

    # ---------- INTERNAL METHODS ----------

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> _Node:
        """
        Recursively build the decision tree.
        """

        # Stopping conditions
        # 1. max depth reached
        if depth >= self.max_depth:
            return self._create_leaf(y)

        # 2. all labels same (pure node)
        if y.nunique() == 1:
            return self._create_leaf(y)

        # 3. no features left or no samples
        if X.shape[0] == 0 or X.shape[1] == 0:
            return self._create_leaf(y)

        # Choose criterion for utils (entropy/gini/mse)
        if self.is_regression:
            local_criterion = "mse"
        else:
            if self.criterion == "information_gain":
                local_criterion = "entropy"
            elif self.criterion == "gini_index":
                local_criterion = "gini"
            else:
                raise ValueError(f"Unknown criterion: {self.criterion}")

        # Candidate features to split on
        features = pd.Series(X.columns)

        # Find best attribute and threshold
        best_attr, best_thresh = opt_split_attribute(X, y, local_criterion, features)

        # If no useful split found, make a leaf
        if best_attr is None or best_thresh is None:
            return self._create_leaf(y)

        # Split the data
        X_left, y_left, X_right, y_right = split_data(X, y, best_attr, best_thresh)

        # If split fails (one side empty), make a leaf
        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            return self._create_leaf(y)

        # Recursively build children
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        # Create internal node
        return _Node(
            is_leaf=False,
            prediction=None,
            feature=best_attr,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
        )

    def _create_leaf(self, y: pd.Series) -> _Node:
        """
        Create a leaf node with the appropriate prediction.
        """
        if self.is_regression:
            # Regression -> predict mean
            pred = y.mean()
        else:
            # Classification -> predict majority class
            pred = y.value_counts().idxmax()

        return _Node(is_leaf=True, prediction=pred)

    def _predict_one(self, row: pd.Series):
        """
        Traverse the tree for a single example (one row) and return prediction.
        """
        node = self.root
        while not node.is_leaf:
            val = row[node.feature]
            if val <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def _print_node(self, node: _Node, indent: str) -> None:
        """
        Recursively print the tree.
        """
        if node.is_leaf:
            print(indent + f"Leaf: {node.prediction}")
            return

        # Internal node
        print(indent + f"?({node.feature} <= {node.threshold:.3f})")
        print(indent + "Y:")
        self._print_node(node.left, indent + "    ")
        print(indent + "N:")
        self._print_node(node.right, indent + "    ")
