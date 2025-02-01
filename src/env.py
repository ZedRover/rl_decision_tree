import numpy as np
import gymnasium as gym
from gymnasium import spaces
from copy import deepcopy
from .tree import DecisionTree
import wandb
from .utils import calculate_feature_gaps


class DecisionTreeEnv(gym.Env):
    def __init__(self, X, y, max_depth, n_classes, action_logger):
        super().__init__()
        self.X = X
        self.y = y
        self.labels = np.unique(y)
        self.m, self.d = X.shape
        self.max_depth = max_depth
        self.n_classes = n_classes
        self.action_logger = action_logger

        # Calculate feature gaps
        self.feature_gaps = calculate_feature_gaps(self.X)

        # Flatten all gap values into a single action space
        self.all_gaps = []
        self.gap_indices = []  # Keep track of which feature and gap
        for feature in self.feature_gaps:
            for gap in feature['gaps']:
                self.all_gaps.append(gap)
                self.gap_indices.append((feature['feature_index'], gap))

        # Define the action space as the number of valid gaps
        self.action_space = spaces.Discrete(len(self.all_gaps))

        # Define observation space
        max_nodes = 2 ** (self.max_depth) - 1
        self.observation_space = spaces.Box(
            low=-1, high=self.d - 1, shape=(max_nodes * 2,), dtype=np.float32
        )

        # Initialize tree
        self.tree = DecisionTree(max_depth, n_classes)
        self.current_node = 0
        self.step_count = 0
        self.cur_acc = 0
        self.best_tree = DecisionTree(max_depth, n_classes)
        self.best_tree_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_node = 0
        self.tree = DecisionTree(self.max_depth, self.n_classes)
        return self.tree.get_state_representation(), {}

    def step(self, action):
        done = False

        # Map action to feature and gap value
        feature_index, gap_value = self.gap_indices[action]

        if self.current_node < 2**self.max_depth - 1:
            self.tree.add_node(self.current_node, feature=feature_index, threshold=gap_value)
            reward = self._calculate_reward(self.current_node)

        else:
            done = True

        if done:
            self._assign_leaf_classes()
            total_accuracy = np.mean(self.tree.predict(self.X) == self.y)
            self.action_logger.accuracy[self.step_count] = total_accuracy
            random_accuracy = np.mean(np.bincount(self.y).max() == self.y)

            # Log accuracy to WandB
            wandb.log({"accuracy": total_accuracy, "step": self.step_count})

            reward = (total_accuracy - random_accuracy) * 0
            if total_accuracy > self.cur_acc:
                self.cur_acc = total_accuracy
                self.best_tree = deepcopy(self.tree)
                self.best_tree_step = self.step_count

        self.action_logger.reward[self.step_count] = reward

        self.action_logger.log(
            self.current_node, feature_index, gap_value, self.step_count, reward
        )
        if not done:
            wandb.log(
                {
                    "step": self.step_count,
                    f"node_{self.current_node} | threshold": feature_index + gap_value,
                    "reward": reward,
                }
            )

        self.current_node += 1
        self.step_count += 1

        return self.tree.get_state_representation(), reward, done, False, {}

    def _assign_leaf_classes(self):
        global_label_counts = {label: 0 for label in self.labels}
        for y_value in self.y:
            global_label_counts[y_value] += 1
        global_most_common_class = max(global_label_counts, key=global_label_counts.get)

        for node_id in range(2**self.max_depth - 1, 2 ** (self.max_depth + 1) - 1):
            if node_id not in self.tree.nodes:
                self.tree.add_node(node_id)
            if self.tree.nodes[node_id]["leaf_class"] is None:
                samples_at_node = self._get_samples_for_node(node_id)
                if samples_at_node.size > 0:
                    label_counts = {label: 0 for label in self.labels}
                    for sample_index in samples_at_node:
                        label_counts[self.y[sample_index]] += 1

                    if sum(label_counts.values()) == 0:
                        self.tree.set_leaf_class(node_id, global_most_common_class)
                    else:
                        leaf_class = max(label_counts, key=label_counts.get)
                        self.tree.set_leaf_class(node_id, leaf_class)
                else:
                    self.tree.set_leaf_class(node_id, global_most_common_class)

    def _get_samples_for_node(self, node_id):
        path_conditions = []
        current_node = node_id

        while current_node > 0:
            parent_node = (
                (current_node - 1) // 2
                if current_node % 2 == 1
                else (current_node - 2) // 2
            )
            node = self.tree.nodes.get(parent_node, None)
            if node and "feature" in node and "threshold" in node:
                is_left_child = current_node % 2 == 1
                path_conditions.append(
                    (node["feature"], node["threshold"], is_left_child)
                )
            current_node = parent_node

        indices = np.arange(self.m)
        for feature, threshold, is_left_child in reversed(path_conditions):
            if is_left_child:
                indices = indices[self.X[indices][:, feature] <= threshold]
            else:
                indices = indices[self.X[indices][:, feature] > threshold]

        return indices

    def _calculate_reward(self, node_id):
        samples_at_node = self._get_samples_for_node(node_id)
        default_label = (
            np.bincount(self.y[samples_at_node]).argmax()
            if samples_at_node.size > 0
            else 0
        )
        default_accuracy = (
            np.mean(self.y[samples_at_node] == default_label)
            if samples_at_node.size > 0
            else 0
        )

        feature = self.tree.nodes[node_id]["feature"]
        threshold = self.tree.nodes[node_id]["threshold"]

        if feature is not None and threshold is not None:
            left_indices = samples_at_node[
                self.X[samples_at_node][:, feature] <= threshold
            ]
            right_indices = samples_at_node[
                self.X[samples_at_node][:, feature] > threshold
            ]
            if len(left_indices) == 0 or len(right_indices) == 0:
                return -1

            left_accuracy = (
                np.mean(
                    self.y[left_indices] == np.bincount(self.y[left_indices]).argmax()
                )
                if left_indices.size > 0
                else 0
            )
            right_accuracy = (
                np.mean(
                    self.y[right_indices] == np.bincount(self.y[right_indices]).argmax()
                )
                if right_indices.size > 0
                else 0
            )

            split_accuracy = (
                len(left_indices) * left_accuracy + len(right_indices) * right_accuracy
            ) / max(len(samples_at_node), 1)
        else:
            split_accuracy = default_accuracy

        reward = split_accuracy - default_accuracy
        return reward
