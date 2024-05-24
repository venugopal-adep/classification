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
    st.title("Decision Tree Interactive Tool")

    # Explanation of Decision Trees
    st.header("What are Decision Trees?")
    st.markdown("""
    Decision Trees are a popular machine learning algorithm used for both classification and regression tasks. They are tree-like models that make decisions based on a series of rules learned from the input features.

    The key concepts in Decision Trees are:
    - **Nodes**: The points in the tree where a decision is made based on a feature.
    - **Edges**: The branches connecting the nodes, representing the decision paths.
    - **Leaves**: The final nodes in the tree, representing the predicted class or value.

    Decision Trees learn the optimal rules by recursively splitting the data based on the feature that provides the most information gain or reduces the impurity the most. The splitting process continues until a stopping criterion is met, such as reaching a maximum depth or having a minimum number of samples in a leaf.

    In this interactive tool, you can explore how Decision Trees work on different datasets and see the impact of various hyperparameters on the model's performance.
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

    # Hyperparameters
    max_depth = st.sidebar.slider("Maximum depth", min_value=1, max_value=10, step=1, value=3)
    min_samples_split = st.sidebar.slider("Minimum samples split", min_value=2, max_value=20, step=1, value=2)
    min_samples_leaf = st.sidebar.slider("Minimum samples leaf", min_value=1, max_value=20, step=1, value=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Decision Tree model
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    clf.fit(X_train, y_train)

    # Plot the Decision Tree
    st.subheader("Decision Tree Visualization")
    plot_decision_tree(clf, feature_names, class_names)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Test Accuracy: {accuracy:.2f}")

    # Explanation of the plot
    st.header("Interpreting the Decision Tree")
    st.markdown("""
    The Decision Tree plot visualizes the learned rules and decision paths. Each node in the tree represents a decision based on a feature, and the edges represent the decision paths. The leaves represent the predicted class or value.

    - The nodes are colored based on the majority class of the samples reaching that node.
    - The shade of the node's color indicates the purity of the samples. Darker shades represent higher purity.
    - The text inside each node shows the feature and threshold used for splitting, along with the Gini impurity and the number of samples reaching that node.
    - The leaves show the predicted class and the number of samples in that leaf.

    You can explore the impact of different hyperparameters, such as the maximum depth and minimum samples split, on the structure and complexity of the Decision Tree.
    """)

if __name__ == '__main__':
    main()