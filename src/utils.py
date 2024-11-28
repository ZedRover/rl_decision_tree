import os
import numpy as np


def read_data(data_name):
    if os.path.exists(f"data/{data_name}"):
        data = np.loadtxt(f"data/{data_name}", delimiter=",")
    elif os.path.exists(f"data/{data_name}.csv"):
        data = np.loadtxt(f"data/{data_name}.csv", delimiter=",")
    x, y = data[:, :-1], data[:, -1].astype(int)
    return x, y


def cart_split(X, y):
    """
    Perform a CART-based split for the given data.
    Finds the best feature and threshold that minimizes the Gini impurity.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target vector of shape (n_samples,).

    Returns:
        best_feature (int): Index of the best feature to split on.
        best_threshold (float): Threshold for the best split.
    """
    n_samples, n_features = X.shape
    best_gini = float("inf")
    best_feature = -1
    best_threshold = None

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])  # Unique thresholds for this feature
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            left_y = y[left_mask]
            right_y = y[right_mask]

            # Calculate Gini impurity for the split
            left_gini = (
                1.0
                - sum((np.sum(left_y == c) / len(left_y)) ** 2 for c in np.unique(y))
                if len(left_y) > 0
                else 0
            )
            right_gini = (
                1.0
                - sum((np.sum(right_y == c) / len(right_y)) ** 2 for c in np.unique(y))
                if len(right_y) > 0
                else 0
            )

            # Weighted average of the Gini impurity
            gini = (len(left_y) * left_gini + len(right_y) * right_gini) / len(y)

            # Update the best split if the Gini impurity is lower
            if gini < best_gini:
                best_gini = gini
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold
