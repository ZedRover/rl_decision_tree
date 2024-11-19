import wandb
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from src.utils import read_data
from src.tree import DecisionTree
from src.env import DecisionTreeEnv
import argparse

args = argparse.ArgumentParser()
args.add_argument("--max_depth", type=int, default=2)
args.add_argument("--data_name", type=str, default="glass")
args = args.parse_args()

MAX_DEPTH = args.max_depth
DATA_NAME = args.data_name

TOY_DATA = np.loadtxt(f"data/{DATA_NAME}", delimiter=",")
GLOBAL_X, GLOBAL_Y = read_data(DATA_NAME)
N_CLASSES = len(np.unique(GLOBAL_Y))
M, D = GLOBAL_X.shape

GLOBAL_X = MinMaxScaler().fit_transform(GLOBAL_X)

wandb.init(
    project="rl_decision_tree_training",
    config={
        "data_name": DATA_NAME,
        "max_depth": MAX_DEPTH,
    },
    name=f"{DATA_NAME}_depth{MAX_DEPTH}",
)


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

    def log(self, node_id, feature, threshold, step, reward):
        if feature in self.logs[node_id]:
            self.logs[node_id][feature].append((step, threshold))
            # Log to WandB
            wandb.log(
                {
                    "step": step,
                    "node_id": node_id,
                    "feature": feature,
                    "threshold": threshold,
                    "reward": reward,
                }
            )

    def plot_node_feature_thresholds(self):
        # This function remains the same for generating plots
        # but we can also log them to WandB
        # Save and log plot to WandB
        plt.savefig(f"pic/{DATA_NAME}_feature_threshold_exploration.png")
        wandb.log(
            {
                "feature_threshold_exploration": wandb.Image(
                    f"pic/{DATA_NAME}_feature_threshold_exploration.png"
                )
            }
        )
        plt.show()
        plt.close()

    def plot_rewards(self):
        plt.scatter(self.reward.keys(), self.reward.values(), alpha=0.5)
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Rewards")
        plt.savefig(f"pic/{DATA_NAME}_rewards.png")
        # Log rewards plot to WandB
        wandb.log({"rewards_plot": wandb.Image(f"pic/{DATA_NAME}_rewards.png")})
        plt.show()
        plt.close()

    def plot_accuracy(self):
        plt.scatter(self.accuracy.keys(), self.accuracy.values(), alpha=0.5)
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy")
        plt.savefig(f"pic/{DATA_NAME}_model_accuracy.png")
        # Log accuracy plot to WandB
        wandb.log({"accuracy_plot": wandb.Image(f"pic/{DATA_NAME}_model_accuracy.png")})
        plt.show()
        plt.close()


if __name__ == "__main__":
    action_logger = ActionLogger(MAX_DEPTH, D)
    env = DummyVecEnv(
        [
            lambda: DecisionTreeEnv(
                GLOBAL_X, GLOBAL_Y, MAX_DEPTH, N_CLASSES, action_logger
            )
        ]
    )
    policy_kwargs = dict(net_arch=dict(pi=[100, 32], vf=[100, 32]))
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        policy_kwargs=policy_kwargs,
        learning_rate=lambda x: 1e-4 * (1 - x),
    )
    total_steps = 1_000_000
    model.learn(total_timesteps=total_steps)

    action_logger.plot_node_feature_thresholds()
    action_logger.plot_rewards()
    action_logger.plot_accuracy()

    # Extract the tree structure from the model
    final_tree = env.get_attr("best_tree")[0]
    print(f"Final Decision Tree @{env.get_attr("best_tree_step")[0]} Structure:")
    print(final_tree.nodes)
    print("Final Decision Tree Structure:")
    tree_structure = final_tree.export_mytree()
    print(tree_structure)

    predictions = final_tree.predict(GLOBAL_X)
    accuracy = np.mean(predictions == GLOBAL_Y)
    print(f"Final model accuracy: {accuracy:.4f}")

    # Train and evaluate sklearn DecisionTreeClassifier
    cart_tree = DecisionTreeClassifier(max_depth=MAX_DEPTH)
    cart_tree.fit(GLOBAL_X, GLOBAL_Y)
    cart_predictions = cart_tree.predict(GLOBAL_X)
    cart_accuracy = np.mean(cart_predictions == GLOBAL_Y)
    print(f"Sklearn DecisionTree accuracy: {cart_accuracy:.4f}")

    # Export sklearn tree structure
    tree_rules = export_text(
        cart_tree, feature_names=[f"Feature {i}" for i in range(D)]
    )
    print(tree_rules)

    # Save all information to a text file
    wandb.log(
        {
            "final_model_accuracy": accuracy,
            "cart_accuracy": cart_accuracy,
            "best_tree_step": np.round(
                env.get_attr("best_tree_step")[0] / total_steps, 2
            ),
        }
    )

    with open(f"logs/{DATA_NAME}_tree_info_d{MAX_DEPTH}_s{total_steps}.txt", "w") as f:
        f.write("Final Decision Tree Structure:\n")
        f.write(tree_structure)
        f.write(f"\nFinal model accuracy: {accuracy:.4f}\n\n")

        f.write("Sklearn Decision Tree Structure:\n")
        f.write(tree_rules)
        f.write(f"\nSklearn model accuracy: {cart_accuracy:.4f}\n\n")

        f.write(f"Best Tree Step: {env.get_attr('best_tree_step')[0]}\n")
    wandb.save(f"logs/{DATA_NAME}_tree_info_d{MAX_DEPTH}_s{total_steps}.txt")
