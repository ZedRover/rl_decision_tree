import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.tree import DecisionTree

# Generate synthetic data for validation
np.random.seed(42)
X = np.random.rand(1000, 20)  # 100 samples, 5 features
y = np.random.randint(0, 2, size=1000)  # Binary classification

# Train a CART decision tree
cart_tree = DecisionTreeClassifier(max_depth=3)
cart_tree.fit(X, y)

# Extract parameters from the CART decision tree
my_tree = DecisionTree(max_depth=3, n_classes=2)

# Recursive function to add nodes to my_tree
def add_cart_nodes(my_tree, cart_tree, node_id=0):
    tree = cart_tree.tree_
    if node_id >= tree.node_count:
        return

    feature = tree.feature[node_id]
    threshold = tree.threshold[node_id]

    if tree.children_left[node_id] == tree.children_right[node_id]:
        # Leaf node
        value = tree.value[node_id]
        leaf_class = np.argmax(value)
        my_tree.add_node(node_id)
        my_tree.set_leaf_class(node_id, leaf_class)
    else:
        # Decision node
        my_tree.add_node(node_id, feature=feature, threshold=threshold)
        add_cart_nodes(my_tree, cart_tree, tree.children_left[node_id])
        add_cart_nodes(my_tree, cart_tree, tree.children_right[node_id])

# Add all nodes to my_tree
add_cart_nodes(my_tree, cart_tree)

# Compare predictions
cart_predictions = cart_tree.predict(X)
my_tree_predictions = my_tree.predict(X)

# Check if predictions match
accuracy_match = np.mean(cart_predictions == my_tree_predictions)
print(f"Accuracy match between CART and MyTree: {accuracy_match * 100:.2f}%")

# Optional: Print the tree structure for debugging
print("\nCART Tree Structure:")
print(cart_tree)

print("\nMy Tree Structure:")
print(my_tree.export_mytree())