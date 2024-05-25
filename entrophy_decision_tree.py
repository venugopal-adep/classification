import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Set page title
st.set_page_config(page_title="Entropy Demonstration")

# Title
st.title("Entropy Demonstration in Decision Trees")
st.write('**Developed by : Venugopal Adep**')

# Description
st.write("This interactive program demonstrates how entropy is used in creating a decision tree.")

# Sidebar for user inputs
st.sidebar.title("Parameters")
n_samples = st.sidebar.slider("Number of samples", 50, 500, 200, 50)
n_features = st.sidebar.slider("Number of features", 2, 10, 5, 1)
max_depth = st.sidebar.slider("Maximum depth of the decision tree", 1, 10, 3, 1)

# Generate random data
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=2, n_redundant=0, random_state=42)

# Create decision tree classifier
dt = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
dt.fit(X, y)

# Plot the decision tree
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(dt, filled=True, rounded=True, class_names=["Class 0", "Class 1"], feature_names=[f"Feature {i}" for i in range(n_features)])
st.pyplot(fig)

# Explain entropy
st.header("Entropy")
st.write("Entropy is a measure of impurity used in decision trees to determine the best split at each node. It quantifies the amount of uncertainty or randomness in the data.")
st.latex(r"Entropy(t) = -\sum_{i=1}^{c} p_i \log_2(p_i)")
st.write("Where:")
st.write("- $t$ is the current node")
st.write("- $c$ is the number of classes")
st.write("- $p_i$ is the proportion of samples belonging to class $i$ at node $t$")

st.write("Entropy ranges from 0 to 1, where 0 indicates a perfectly pure node (all samples belong to the same class) and 1 indicates a maximally impure node (samples are evenly distributed among classes).")

st.write("At each node, the decision tree algorithm selects the feature and split point that maximizes the information gain, which is the difference between the entropy of the parent node and the weighted average of the entropies of the child nodes. This process is repeated recursively until a stopping criterion is met, such as reaching the maximum depth or having a minimum number of samples in a leaf node.")

# Interpretation of the decision tree
st.header("Interpretation")
st.write("In the decision tree plot above, each node shows the feature and threshold used for splitting, along with the entropy at that node. The colors of the nodes represent the majority class at each node.")
st.write("The decision tree splits the data based on the features and thresholds that maximize the information gain, leading to a hierarchical structure that can be used for classification.")

# Conclusion
st.header("Conclusion")
st.write("Entropy is a fundamental concept in decision tree learning, allowing the algorithm to determine the best splits based on the impurity of the nodes. By minimizing the entropy at each split, the decision tree aims to create pure leaf nodes, leading to accurate classifications.")
