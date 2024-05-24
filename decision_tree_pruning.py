import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

def plot_decision_tree(clf, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names, ax=ax)
    st.pyplot(fig)

def main():
    st.title("Decision Tree Pruning Interactive Tool")

    # Explanation of Decision Tree Pruning
    st.header("What is Decision Tree Pruning?")
    st.markdown("""
    Decision Tree Pruning is a technique used to simplify and optimize decision trees by removing unnecessary branches or nodes. The goal of pruning is to reduce overfitting and improve the generalization performance of the tree.

    There are two main approaches to pruning:
    1. **Pre-pruning**: Stopping the tree-growing process early based on certain criteria, such as maximum depth or minimum samples per leaf.
    2. **Post-pruning**: Growing a full tree and then removing or collapsing branches based on their impact on validation or test set performance.

    Pruning helps to find the right balance between model complexity and generalization ability. It reduces the size of the tree, making it more interpretable and less prone to overfitting.

    In this interactive tool, you can explore the effect of different pruning parameters on the structure and performance of a Decision Tree.
    """)

    # Sidebar layout
    st.sidebar.title("Options")

    # Dataset selection
    dataset_options = ['Iris', 'Wine', 'Breast Cancer']
    selected_dataset = st.sidebar.selectbox("Select a dataset", dataset_options)

    # Load the selected dataset
    if selected_dataset == 'Iris':
        data = load_iris()
    elif selected_dataset == 'Wine':
        data = load_wine()
    else:
        data = load_breast_cancer()

    X = data.data
    y = data.target
    feature_names = data.feature_names
    class_names = data.target_names

    # Pruning parameters
    criterion = st.sidebar.selectbox("Splitting criterion", ['gini', 'entropy'])
    max_depth = st.sidebar.slider("Maximum depth", min_value=1, max_value=10, step=1, value=3)
    min_samples_split = st.sidebar.slider("Minimum samples split", min_value=2, max_value=20, step=1, value=2)
    min_samples_leaf = st.sidebar.slider("Minimum samples leaf", min_value=1, max_value=20, step=1, value=1)
    ccp_alpha = st.sidebar.slider("Complexity parameter (ccp_alpha)", min_value=0.0, max_value=0.1, step=0.001, value=0.0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Decision Tree model
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)

    # Plot the Decision Tree
    st.subheader("Pruned Decision Tree Visualization")
    plot_decision_tree(clf, feature_names, class_names)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Test Accuracy: {accuracy:.2f}")

    # Explanation of the pruning parameters and their effects
    st.header("Understanding Pruning Parameters")
    st.markdown("""
    The pruning parameters control the complexity and generalization of the Decision Tree:

    - **Splitting criterion**: The function used to measure the quality of a split. Gini impurity and entropy are common criteria.
    - **Maximum depth**: The maximum depth allowed for the Decision Tree. Limiting the depth helps to prevent overfitting.
    - **Minimum samples split**: The minimum number of samples required to split an internal node. Higher values prevent overfitting by requiring more samples to make a split.
    - **Minimum samples leaf**: The minimum number of samples required to be at a leaf node. Higher values create simpler trees by forcing more samples into each leaf.
    - **Complexity parameter (ccp_alpha)**: The threshold used for post-pruning. Nodes with a cost complexity lower than this value are pruned. Higher values result in more aggressive pruning.

    Experiment with different pruning parameters to observe their impact on the tree structure and performance. Finding the right balance is key to achieving a model that generalizes well to unseen data.
    """)

if __name__ == '__main__':
    main()