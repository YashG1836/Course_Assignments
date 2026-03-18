from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    assert isinstance(y_hat, pd.Series)
    assert isinstance(y, pd.Series)
    assert y_hat.size == y.size, "y_hat and y must be of same length"
    assert y_hat.size > 0, "y_hat and y must be non-empty"

    return (y_hat == y).sum() / y_hat.size

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision for a given class `cls`
    """
    assert isinstance(y_hat, pd.Series)
    assert isinstance(y, pd.Series)
    assert y_hat.size == y.size, "y_hat and y must be of same length"
    assert y_hat.size > 0, "y_hat and y must be non-empty"

    # Positive predictions = predicted cls
    pred_pos = (y_hat == cls)
    true_pos = (y == cls)

    TP = (pred_pos & true_pos).sum()
    FP = (pred_pos & ~true_pos).sum()

    if TP + FP == 0:
        # No predictions for this class
        return 0.0

    return TP / (TP + FP)


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall for a given class `cls`
    """
    assert isinstance(y_hat, pd.Series)
    assert isinstance(y, pd.Series)
    assert y_hat.size == y.size, "y_hat and y must be of same length"
    assert y_hat.size > 0, "y_hat and y must be non-empty"

    pred_pos = (y_hat == cls)
    true_pos = (y == cls)

    TP = (pred_pos & true_pos).sum()
    FN = (~pred_pos & true_pos).sum()

    if TP + FN == 0:
        # Class `cls` never appears in true labels
        return 0.0

    return TP / (TP + FN)



def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert isinstance(y_hat, pd.Series)
    assert isinstance(y, pd.Series)
    assert y_hat.size == y.size, "y_hat and y must be of same length"
    assert y_hat.size > 0, "y_hat and y must be non-empty"

    diff = y_hat - y
    mse = (diff ** 2).mean()
    return mse ** 0.5


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert isinstance(y_hat, pd.Series)
    assert isinstance(y, pd.Series)
    assert y_hat.size == y.size, "y_hat and y must be of same length"
    assert y_hat.size > 0, "y_hat and y must be non-empty"

    return (y_hat - y).abs().mean()
