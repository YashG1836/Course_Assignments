import pandas as pd
import math


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data.
    Converts categorical (object/category) columns to 0/1 columns.
    Numeric columns are left as-is.
    """
    assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
    # drop_first=False -> keep all categories (no need to drop because trees don't care about collinearity)
    X_enc = pd.get_dummies(X, drop_first=False)
    return X_enc


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real (continuous) or discrete values.
    Here we treat float-dtype targets as real-valued.
    """
    assert isinstance(y, pd.Series), "y must be a pandas Series"
    return pd.api.types.is_float_dtype(y)



def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy of a discrete label Series Y.
    """
    assert isinstance(Y, pd.Series), "Y must be a pandas Series"
    assert Y.size > 0, "Y must be non-empty"

    # Probability of each class
    probs = Y.value_counts(normalize=True)

    ent = 0.0
    for p in probs:
        if p > 0:
            ent -= p * math.log2(p)
    return ent



def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index of a discrete label Series Y.
    """
    assert isinstance(Y, pd.Series), "Y must be a pandas Series"
    assert Y.size > 0, "Y must be non-empty"

    probs = Y.value_counts(normalize=True)
    return 1.0 - (probs ** 2).sum()


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion 
    (entropy, gini index, or MSE) when splitting Y using attribute 'attr'.
    """
    assert isinstance(Y, pd.Series)
    assert isinstance(attr, pd.Series)
    assert Y.size == attr.size, "Y and attr must have same length"
    assert Y.size > 0, "Y must be non-empty"

    N = Y.size

    def mse(values: pd.Series) -> float:
        mean_val = values.mean()
        return ((values - mean_val) ** 2).mean()

    # parent impurity
    if criterion == "entropy":
        parent_imp = entropy(Y)
        impurity_func = entropy
    elif criterion == "gini":
        parent_imp = gini_index(Y)
        impurity_func = gini_index
    elif criterion.lower() == "mse":
        parent_imp = mse(Y)
        impurity_func = mse
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    # weighted impurity of children
    df = pd.DataFrame({"Y": Y, "A": attr})
    weighted_impurity = 0.0

    for _, group in df.groupby("A"):
        w = group.shape[0] / N
        weighted_impurity += w * impurity_func(group["Y"])

    return parent_imp - weighted_impurity


# def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
#     """
#     Function to find the optimal attribute to split about.

#     features: pd.Series or list-like of column names we have to split upon

#     return: attribute (column name) to split upon
#     """

#     assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
#     assert isinstance(y, pd.Series), "y must be a pandas Series"
#     assert X.shape[0] == y.size, "X and y must have the same number of rows"
#     assert len(features) > 0, "features must be non-empty"

#     best_attr = None
#     best_gain = -float("inf")

#     # Loop over all candidate features
#     for attr in features:
#         # attr is the column name
#         col = X[attr]

#         # Compute information gain for this feature
#         gain = information_gain(y, col, criterion)

#         # Keep track of the best one
#         if gain > best_gain:
#             best_gain = gain
#             best_attr = attr

#     return best_attr
def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute and threshold to split about.

    features: pd.Series or list-like of column names we consider for splitting.

    return: (best_attribute_name, best_threshold)
    """
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.size, "X and y must have same number of rows"
    assert len(features) > 0, "features must be non-empty"

    best_attr = None
    best_thresh = None
    best_gain = -float("inf")

    for attr in features:
        col = X[attr]
        # unique sorted values
        unique_vals = sorted(col.unique())
        if len(unique_vals) <= 1:
            continue  # can't split on constant feature

        # candidate thresholds = midpoints between consecutive unique values
        thresholds = [
            0.5 * (unique_vals[i] + unique_vals[i + 1])
            for i in range(len(unique_vals) - 1)
        ]

        for t in thresholds:
            # binary attribute: True if <= t, False otherwise
            attr_binary = col <= t

            gain = information_gain(y, attr_binary, criterion)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
                best_thresh = t

    return best_attr, best_thresh


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Function to split the data according to an attribute and a threshold.

    attribute: column name to split upon
    value: threshold value (for real features) or split value

    return: (X_left, y_left, X_right, y_right)
    """
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert attribute in X.columns, "attribute must be a column in X"
    assert X.shape[0] == y.size, "X and y must have same number of rows"

    col = X[attribute]
    left_mask = col <= value
    right_mask = ~left_mask

    X_left = X[left_mask]
    y_left = y[left_mask]

    X_right = X[right_mask]
    y_right = y[right_mask]

    return X_left, y_left, X_right, y_right
