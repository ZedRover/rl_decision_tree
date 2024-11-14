import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from .utils import ActionLogger


# 全局数据生成
def generate_global_data(m, d, n_classes):
    X = np.random.rand(m, d)
    y = np.random.randint(0, n_classes, size=m)
    return X, y


# 全局参数
# M = 10  # 样本数量
# D = 2    # 特征维度
# MAX_DEPTH = 2
# N_CLASSES = 2


# 生成全局数据
# GLOBAL_X, GLOBAL_Y = generate_global_data(M, D, N_CLASSES)
M = 10
D = 2
MAX_DEPTH = 2
N_CLASSES = 2
TOY_DATA = np.loadtxt("toy_data.csv", delimiter=",")
GLOBAL_X, GLOBAL_Y = TOY_DATA[:, :-1], TOY_DATA[:, -1].astype(int)


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
        # print(f"Added node {node_id} with feature {feature} and threshold {threshold}")

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
                return (
                    node["leaf_class"]
                    if node is not None
                    else np.random.randint(self.n_classes)
                )

            if x[node["feature"]] <= node["threshold"]:
                node_id = 2 * node_id + 1
            else:
                node_id = 2 * node_id + 2

    def get_state_representation(self):
        max_nodes = 2 ** (self.max_depth + 1) - 1
        state = np.full(max_nodes * 2, -1, dtype=float)
        for node_id, node in self.nodes.items():
            state[node_id * 2] = node["feature"] if node["feature"] is not None else -1
            state[node_id * 2 + 1] = (
                node["threshold"] if node["threshold"] is not None else -1
            )
        return state


class DecisionTreeEnv(gym.Env):
    def __init__(self, X, y, max_depth, n_classes):
        super().__init__()
        self.X = X
        self.y = y
        self.m, self.d = X.shape
        self.max_depth = max_depth
        self.n_classes = n_classes

        # 动作空间: [feature, threshold]
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([self.d - 1, 1]), dtype=np.float32
        )

        # 状态空间: 仅包含特征和阈值信息
        max_nodes = 2 ** (self.max_depth + 1) - 1
        self.observation_space = spaces.Box(
            low=-1, high=self.d - 1, shape=(max_nodes * 2,), dtype=np.float32
        )

        self.tree = DecisionTree(max_depth, n_classes)
        self.current_node = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Only reset `current_node` to keep the tree?
        self.current_node = 0
        self.tree = DecisionTree(self.max_depth, self.n_classes)
        return self.tree.get_state_representation(), {}

    def step(self, action):
        feature = int(action[0])
        threshold = action[1]

        if self.current_node < 2**self.max_depth - 1:
            self.tree.add_node(self.current_node, feature=feature, threshold=threshold)
        else:
            self._assign_leaf_classes()

        reward = self._calculate_reward(self.current_node)
        self.current_node += 1
        done = self.current_node >= 2 ** (self.max_depth + 1) - 1
        # if done:
        #     reward += np.mean(self.tree.predict(self.X) == self.y)- np.mean(self.y == np.bincount(self.y).argmax())

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
        if node_id == 0:
            samples_at_node = np.arange(self.m)
            default_accuracy = np.mean(self.y == np.bincount(self.y).argmax())

        else:
            samples_at_node = self._get_samples_for_node(node_id)
            if samples_at_node.size > 0:
                default_label = np.bincount(self.y[samples_at_node]).argmax()
                default_accuracy = np.mean(self.y[samples_at_node] == default_label)
            else:
                return 0
                # default_accuracy = 0

        feature = self.tree.nodes[node_id]["feature"]
        threshold = self.tree.nodes[node_id]["threshold"]

        if feature is not None and threshold is not None:
            left_indices = samples_at_node[
                self.X[samples_at_node][:, feature] <= threshold
            ]
            right_indices = samples_at_node[
                self.X[samples_at_node][:, feature] > threshold
            ]

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
            # split_accuracy = (left_accuracy+right_accuracy)/2
        else:
            split_accuracy = (
                default_accuracy  # If no split is possible, use default accuracy.
            )

        reward = split_accuracy - default_accuracy
        print(f"Node {node_id} Reward: {reward:.4f}")
        return reward


# 创建环境
env = DecisionTreeEnv(GLOBAL_X, GLOBAL_Y, MAX_DEPTH, N_CLASSES)
env = DummyVecEnv([lambda: env])

# 创建并训练模型
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=1_000_000)

# 使用训练好的模型
obs = env.reset()
for _ in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, dones, infos = env.step(action)
    if dones.any():
        break

# 输出最终的决策树结构
final_tree = env.get_attr("tree")[0]
print("Final Decision Tree Structure:")
for node_id, node in final_tree.nodes.items():
    if node["leaf_class"] is not None:
        print(f"Node {node_id}: Leaf Class = {node['leaf_class']}")
    else:
        print(
            f"Node {node_id}: Feature = {node['feature']}, Threshold = {node['threshold']:.4f}"
        )

# 评估最终模型
predictions = final_tree.predict(GLOBAL_X)
accuracy = np.mean(predictions == GLOBAL_Y)
print(f"Final model accuracy: {accuracy:.4f}")

cart_tree = DecisionTreeClassifier(max_depth=MAX_DEPTH)
cart_tree.fit(GLOBAL_X, GLOBAL_Y)
cart_predictions = cart_tree.predict(GLOBAL_X)
cart_accuracy = np.mean(cart_predictions == GLOBAL_Y)
print(f"Sklearn DecisionTree accuracy: {cart_accuracy:.4f}")
# print decision tree structure

tree_rules = export_text(cart_tree, feature_names=[f"Feature {i}" for i in range(D)])
print(tree_rules)
