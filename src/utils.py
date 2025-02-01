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


def calculate_feature_gaps(X):
    """
    Calculate gaps for each feature in the dataset X.

    Parameters:
    X (numpy.ndarray): The dataset with shape (n_samples, n_features).

    Returns:
    list: A list of dictionaries, where each dictionary contains:
        - 'feature_index': The index of the feature.
        - 'gaps': The calculated gap values for that feature.
    """
    feature_gaps = []

    for feature_idx in range(X.shape[1]):
        # Extract and sort the feature values
        feature_values = np.sort(X[:, feature_idx])
        feature_values = np.unique(feature_values)

        # Calculate gaps as the average of consecutive values
        gaps = (feature_values[1:] + feature_values[:-1]) / 2

        # Store the results
        feature_gaps.append({"feature_index": feature_idx, "gaps": gaps})

    return feature_gaps


# Example usage
if __name__ == "__main__":
    # Sample dataset
    X_sample = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 5.0],
            [4.0, 1.0, 2.0],
        ]
    )

    gaps = calculate_feature_gaps(X_sample)
    for gap in gaps:
        print(f"Feature {gap['feature_index']} gaps: {gap['gaps']}")
