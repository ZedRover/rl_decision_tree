import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from copy import deepcopy


MAX_DEPTH = 2
TOY_DATA = np.loadtxt("data/body", delimiter=",")
D = N_CLASSES = len(np.unique(TOY_DATA[:, -1]))
GLOBAL_X, GLOBAL_Y = TOY_DATA[:, :-1], TOY_DATA[:, -1].astype(int)
print(GLOBAL_X)

class ActionLogger:
    def __init__(self, max_depth, d):
        self.logs = {}
        self.d = d
        self.max_depth = max_depth
        self.max_nodes = 2 ** (max_depth + 1) - 1
        self.reward = {}
        self.accuracy = {}

        for node_id in range(self.max_nodes):
            self.logs[node_id] = {feature: [] for feature in range(d)}

    def log(self, node_id, feature, threshold, step):
        if feature in self.logs[node_id]:
            self.logs[node_id][feature].append((step, threshold))

    def plot_node_feature_thresholds(self):
        num_nodes = len(self.logs)
        num_features = max(len(features) for features in self.logs.values())

        fig, axes = plt.subplots(
            num_nodes, num_features, figsize=(5 * num_features, 5 * num_nodes)
        )

        for i, (node_id, feature_data) in enumerate(self.logs.items()):
            for j, (feature, values) in enumerate(feature_data.items()):
                ax = (
                    axes[i, j]
                    if num_nodes > 1 and num_features > 1
                    else (axes[i] if num_nodes > 1 else axes[j])
                )
                if values:
                    steps, thresholds = zip(*values)
                    ax.scatter(steps, thresholds, alpha=0.5)
                    ax.set_xlabel("Steps")
                    ax.set_ylabel("Threshold")
                    ax.set_title(f"Node {node_id} - Feature {feature}")
                else:
                    ax.axis("off")

        plt.tight_layout()
        plt.savefig("feature_threshold_exploration.png")
        plt.show()
        plt.close()

    def plot_rewards(self):
        plt.scatter(self.reward.keys(), self.reward.values(), alpha=0.5)
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Rewards")
        plt.savefig("Rewards.png")
        plt.show()
        plt.close()

    def plot_accuracy(self):
        plt.scatter(self.accuracy.keys(), self.accuracy.values(), alpha=0.5)
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy")
        plt.savefig("model_accuracy.png")
        plt.show()
        plt.close()


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
                # 返回节点类或默认类（这里选择返回 0）
                return node["leaf_class"] if node is not None else 0

            # 确保 feature 和 threshold 不为空
            if node["feature"] is None or node["threshold"] is None:
                return 0  # 或者选择一个默认的类别

            # 基于阈值选择下一个节点
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


class DecisionTreeEnv(gym.Env):
    def __init__(self, X, y, max_depth, n_classes, action_logger):
        super().__init__()
        self.X = X
        self.y = y
        self.m, self.d = X.shape
        self.max_depth = max_depth
        self.n_classes = n_classes
        self.action_logger = action_logger

        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([self.d - 1, 1]), dtype=np.float32
        )

        max_nodes = 2 ** (self.max_depth) - 1
        self.observation_space = spaces.Box(
            low=-1, high=self.d - 1, shape=(max_nodes * 2,), dtype=np.float32
        )

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
        feature = int(action[0])
        threshold = action[1]
        # print(
        #     f"Node {self.current_node}: Split on Feature {feature} at Threshold {threshold:.4f}"
        # )

        self.action_logger.log(self.current_node, feature, threshold, self.step_count)

        if self.current_node < 2**self.max_depth - 1:
            self.tree.add_node(self.current_node, feature=feature, threshold=threshold)

        done = self.current_node >= 2**self.max_depth - 2
        if done:
            self._assign_leaf_classes()

        reward = self._calculate_reward(self.current_node)
        self.current_node += 1
        self.step_count += 1

        if done:
            try:
                total_accuracy = np.mean(self.tree.predict(self.X) == self.y)
                self.action_logger.accuracy[self.step_count] = total_accuracy
                if total_accuracy > self.cur_acc:
                    print(f"Total accuracy: {total_accuracy:.4f}")
                    self.cur_acc = total_accuracy
                    self.best_tree = deepcopy(self.tree)
                    self.best_tree_step = self.step_count
            except Exception as e:
                print(f"Error during accuracy calculation or tree copying: {e}")
                reward = -0.1

        self.action_logger.reward[self.step_count] = reward
        return self.tree.get_state_representation(), reward, done, False, {}

    def _assign_leaf_classes(self):
        for node_id in range(2**self.max_depth - 1, 2 ** (self.max_depth + 1) - 1):
            if node_id not in self.tree.nodes:
                self.tree.add_node(node_id)
            if self.tree.nodes[node_id]["leaf_class"] is None:
                samples_at_node = self._get_samples_for_node(node_id)
                if samples_at_node.size > 0:
                    leaf_class = np.bincount(self.y[samples_at_node]).argmax()
                    self.tree.set_leaf_class(node_id, leaf_class)

    def _get_samples_for_node(self, node_id):
        indices = np.arange(self.m)
        for depth in range(self.max_depth):
            if node_id < 2**depth:
                break
            parent_node = node_id // 2
            node = self.tree.nodes.get(parent_node, {})
            feature = node.get("feature")
            threshold = node.get("threshold")

            if feature is None or threshold is None:
                break

            if node_id % 2 == 1:
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
            # if len(left_indices) == 0 or len(right_indices) == 0:
            #     return -0.2
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
        return reward * 10


action_logger = ActionLogger(MAX_DEPTH, D)

env = DecisionTreeEnv(GLOBAL_X, GLOBAL_Y, MAX_DEPTH, N_CLASSES, action_logger)
env = DummyVecEnv([lambda: env])

policy_kwargs = dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
    policy_kwargs=policy_kwargs,
    # learning_rate=1e-5,
)
model.learn(total_timesteps=1_000_00)

action_logger.plot_node_feature_thresholds()
action_logger.plot_rewards()
action_logger.plot_accuracy()

final_tree = env.get_attr("best_tree")[0]

print(f"Final Decision Tree @{env.get_attr("best_tree_step")[0]} Structure:")
for node_id, node in final_tree.nodes.items():
    if node["leaf_class"] is not None:
        print(f"Node {node_id}: Leaf Class = {node['leaf_class']}")
    else:
        print(
            f"Node {node_id}: Feature = {node['feature']}, Threshold = {node['threshold']:.4f}"
        )

predictions = final_tree.predict(GLOBAL_X)
accuracy = np.mean(predictions == GLOBAL_Y)
print(f"Final model accuracy: {accuracy:.4f}")

cart_tree = DecisionTreeClassifier(max_depth=MAX_DEPTH)
cart_tree.fit(GLOBAL_X, GLOBAL_Y)
cart_predictions = cart_tree.predict(GLOBAL_X)
cart_accuracy = np.mean(cart_predictions == GLOBAL_Y)
print(f"Sklearn DecisionTree accuracy: {cart_accuracy:.4f}")

tree_rules = export_text(cart_tree, feature_names=[f"Feature {i}" for i in range(D)])
print(tree_rules)
