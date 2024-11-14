import numpy as np


class DecisionTree:
    def __init__(self, max_depth, n_classes):
        self.max_depth = max_depth
        self.n_classes = n_classes
        self.nodes = {}
        self.n_features = None

    def add_node(self, node_id, feature=None, threshold=None):
        if self.n_features is None and feature is not None:
            self.n_features = feature + 1
        self.nodes[node_id] = {
            "feature": feature,
            "threshold": threshold,
            "leaf_class": None,
        }

    def set_leaf_class(self, node_id, leaf_class):
        if node_id in self.nodes:
            self.nodes[node_id]["leaf_class"] = leaf_class

    def predict(self, X):
        if len(X.shape) == 1:
            return self._predict_single(X)
        else:
            return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        node_id = 0
        while True:
            node = self.nodes.get(node_id)
            if node is None or node["leaf_class"] is not None:
                return node["leaf_class"] if node is not None else 0

            if node["feature"] is None or node["threshold"] is None:
                return 0

            if x[node["feature"]] <= node["threshold"]:
                node_id = 2 * node_id + 1
            else:
                node_id = 2 * node_id + 2

    def get_state_representation(self):
        max_nodes = 2 ** (self.max_depth) - 1
        state = np.full(max_nodes * 2, -1, dtype=float)
        for node_id, node in self.nodes.items():
            if node_id >= max_nodes:
                break
            state[node_id * 2] = node["feature"] if node["feature"] is not None else -1
            state[node_id * 2 + 1] = (
                node["threshold"] if node["threshold"] is not None else -1
            )
        return state

    def export_mytree(self, node_id=0, depth=0):
        """Recursively export the tree structure in a human-readable format."""
        tree_structure = ""

        node = self.nodes.get(node_id, None)
        if node is None:
            return tree_structure

        indent = "|   " * depth
        if node["leaf_class"] is not None:
            # Leaf node
            tree_structure += f"{indent}|--- class: {node['leaf_class']}\n"
        else:
            # Decision node
            feature = node["feature"]
            threshold = node["threshold"]

            # Left branch
            tree_structure += f"{indent}|--- Feature {feature} <= {threshold:.2f}\n"
            tree_structure += self.export_mytree(
                node_id=2 * node_id + 1, depth=depth + 1
            )

            # Right branch
            tree_structure += f"{indent}|--- Feature {feature} >  {threshold:.2f}\n"
            tree_structure += self.export_mytree(
                node_id=2 * node_id + 2, depth=depth + 1
            )

        return tree_structure
