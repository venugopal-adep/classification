import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Set page title
st.set_page_config(page_title="Information Gain Demonstration")

# Title
st.title("Information Gain Demonstration in Decision Trees")

# Description
st.write("This interactive program demonstrates how information gain is used in creating a decision tree.")

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

# Explain information gain
st.header("Information Gain")
st.write("Information gain is a measure used in decision trees to determine the best split at each node. It quantifies the reduction in entropy achieved by splitting the data based on a particular feature.")
st.latex(r"IG(t, a) = Entropy(t) - \sum_{v \in Values(a)} \frac{|t_v|}{|t|} Entropy(t_v)")
st.write("Where:")
st.write("- $t$ is the current node")
st.write("- $a$ is the feature being considered for splitting")
st.write("- $Values(a)$ is the set of possible values for feature $a$")
st.write("- $t_v$ is the subset of samples at node $t$ where feature $a$ has value $v$")
st.write("- $|t|$ is the number of samples at node $t$")
st.write("- $|t_v|$ is the number of samples in subset $t_v$")

st.write("The feature with the highest information gain is chosen as the splitting feature at each node. This process is repeated recursively until a stopping criterion is met, such as reaching the maximum depth or having a minimum number of samples in a leaf node.")

# Interpretation of the decision tree
st.header("Interpretation")
st.write("In the decision tree plot above, each node shows the feature and threshold used for splitting, along with the entropy at that node. The colors of the nodes represent the majority class at each node.")
st.write("The decision tree splits the data based on the features and thresholds that maximize the information gain, leading to a hierarchical structure that can be used for classification.")

# Conclusion
st.header("Conclusion")
st.write("Information gain is a key concept in decision tree learning, allowing the algorithm to determine the best splits based on the reduction in entropy. By maximizing the information gain at each split, the decision tree aims to create pure leaf nodes, leading to accurate classifications.")