import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import re

# Set page config
st.set_page_config(layout="wide", page_title="Information Gain Explorer", page_icon="🌳")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px !important;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .tab-subheader {
        font-size: 28px !important;
        font-weight: bold;
        color: #4169E1;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .content-text {
        font-size: 18px !important;
        line-height: 1.6;
    }
    .stButton>button {
        background-color: #4682B4;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1E90FF;
        transform: scale(1.05);
    }
    .highlight {
        background-color: #E6F2FF;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .quiz-question {
        background-color: #F0F8FF;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #4169E1;
    }
    .explanation {
        background-color: #F0FFFF;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🌳 Information Gain Interactive Explorer 🌳</h1>", unsafe_allow_html=True)
st.markdown("<p class='content-text'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Dataset Parameters")
n_samples = st.sidebar.slider("Number of samples", min_value=50, max_value=500, value=200, step=50)
n_features = st.sidebar.slider("Number of features", min_value=2, max_value=10, value=5, step=1)
n_classes = st.sidebar.slider("Number of classes", min_value=2, max_value=5, value=2, step=1)
max_depth = st.sidebar.slider("Maximum depth of the decision tree", min_value=1, max_value=10, value=3, step=1)

# Generate random data
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, 
                           n_informative=n_features, n_redundant=0, random_state=42)

# Calculate Entropy
def calculate_entropy(y):
    class_counts = np.bincount(y)
    class_probabilities = class_counts / len(y)
    return -np.sum(class_probabilities * np.log2(class_probabilities + 1e-10))

# Calculate Information Gain
def calculate_information_gain(y, y_left, y_right):
    entropy_parent = calculate_entropy(y)
    weight_left = len(y_left) / len(y)
    weight_right = len(y_right) / len(y)
    entropy_children = weight_left * calculate_entropy(y_left) + weight_right * calculate_entropy(y_right)
    return entropy_parent - entropy_children

# Normalize Information Gain to [0, 1]
def normalize_information_gain(ig):
    return 1 - 2 ** (-ig)

# Create decision tree classifier
dt = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
dt.fit(X, y)

# Calculate information gain for each node
def calculate_node_information_gain(tree, node_id, X, y):
    if tree.feature[node_id] != -2:  # Not a leaf node
        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        y_left = y[left_mask]
        y_right = y[right_mask]
        
        ig = calculate_information_gain(y, y_left, y_right)
        return normalize_information_gain(ig)
    return 0

# Calculate information gain for all nodes
node_information_gain = [calculate_node_information_gain(dt.tree_, i, X, y) for i in range(dt.tree_.node_count)]

# Custom node_dict function
def node_to_str(tree, node_id, criterion):
    if tree.feature[node_id] != -2:
        return f"Feature {tree.feature[node_id]} <= {tree.threshold[node_id]:.3f}\n" \
               f"entropy = {tree.impurity[node_id]:.3f}\n" \
               f"samples = {tree.n_node_samples[node_id]}\n" \
               f"value = {tree.value[node_id][0]}\n" \
               f"class = Class {np.argmax(tree.value[node_id])}\n" \
               f"info gain = {node_information_gain[node_id]:.3f}"
    else:
        return f"entropy = {tree.impurity[node_id]:.3f}\n" \
               f"samples = {tree.n_node_samples[node_id]}\n" \
               f"value = {tree.value[node_id][0]}\n" \
               f"class = Class {np.argmax(tree.value[node_id])}"

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎓 Learn", "🔍 Information Gain Explorer & Decision Tree", "🧠 Quiz"])

with tab1:
    st.markdown("<h2 class='tab-subheader'>Learn About Information Gain</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    Information Gain is a measure used in decision trees to determine the best split at each node. It quantifies the reduction in entropy (or increase in purity) achieved by splitting the data on a particular feature.

    <b>Key points about Information Gain:</b>
    1. It ranges from 0 to 1 (after normalization).
    2. A value of 0 indicates no reduction in entropy (no information gained).
    3. A value close to 1 indicates a significant reduction in entropy (high information gained).
    4. The decision tree algorithm aims to maximize the Information Gain at each split.

    <b>Information Gain Formula:</b>
    </p>
    """, unsafe_allow_html=True)

    st.latex(r"IG(S, F) = H(S) - \sum_{v \in values(F)} \frac{|S_v|}{|S|} H(S_v)")

    st.markdown("""
    <p class='content-text'>
    Where:
    - S is the current dataset or node
    - F is the feature used for splitting
    - H(S) is the entropy of the current dataset
    - S_v is the subset of S where feature F has value v
    - H(S_v) is the entropy of subset S_v

    The decision tree algorithm selects the feature and split point that maximizes the Information Gain. This process is repeated recursively until a stopping criterion is met, such as reaching the maximum depth or having a minimum number of samples in a leaf node.

    By maximizing Information Gain at each split, the decision tree aims to create pure leaf nodes, leading to accurate classifications.
    </p>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("<h2 class='tab-subheader'>Decision Tree Visualization</h2>", unsafe_allow_html=True)
    
    # Plot the decision tree
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(dt, filled=True, rounded=True, 
            class_names=[f"Class {i}" for i in range(n_classes)], 
            feature_names=[f"Feature {i}" for i in range(n_features)],
            node_ids=True, ax=ax)

    # Modify the text in each node to include information gain
    for text in ax.texts:
        bbox = text.get_bbox_patch()
        if bbox is not None:
            # Extract node id from the text
            node_text = text.get_text()
            node_id_match = re.search(r"node #(\d+)", node_text)
            if node_id_match:
                node_id = int(node_id_match.group(1))
                text.set_text(node_to_str(dt.tree_, node_id, 'entropy'))

    st.pyplot(fig)

    st.markdown("<p class='content-text'>This decision tree visualization shows how Information Gain is used to make splits at each node. The entropy values and normalized information gain are displayed in each node, helping you understand the decision-making process.</p>", unsafe_allow_html=True)

with tab3:
    st.markdown("<h2 class='tab-subheader'>Test Your Information Gain Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What does Information Gain measure in decision trees?",
            "options": [
                "The depth of the tree",
                "The reduction in entropy achieved by a split",
                "The number of features",
                "The accuracy of the model"
            ],
            "correct": 1,
            "explanation": "Information Gain measures the reduction in entropy (or increase in purity) achieved by splitting the data on a particular feature."
        },
        {
            "question": "What does an Information Gain of 0 indicate?",
            "options": [
                "Perfect purity",
                "No reduction in entropy",
                "Equal distribution of classes",
                "Insufficient data"
            ],
            "correct": 1,
            "explanation": "An Information Gain of 0 indicates no reduction in entropy, meaning the split doesn't provide any useful information for classification."
        },
        {
            "question": "How does the decision tree algorithm use Information Gain?",
            "options": [
                "To minimize purity at each split",
                "To maximize the reduction in entropy at each split",
                "To determine the depth of the tree",
                "To select the number of features"
            ],
            "correct": 1,
            "explanation": "The decision tree algorithm uses Information Gain to maximize the reduction in entropy at each split, aiming to create the purest possible child nodes."
        }
    ]

    score = 0
    for i, q in enumerate(questions):
        st.markdown(f"<div class='quiz-question'>", unsafe_allow_html=True)
        st.markdown(f"<p class='content-text'><strong>Question {i+1}:</strong> {q['question']}</p>", unsafe_allow_html=True)
        user_answer = st.radio("Select your answer:", q['options'], key=f"q{i}")
        
        if st.button("Check Answer", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! 🎉")
                score += 1
            else:
                st.error("Incorrect. Try again! 🤔")
            st.markdown(f"<div class='explanation'><p>{q['explanation']}</p></div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

    if st.button("Show Final Score"):
        st.markdown(f"<p class='tab-subheader'>Your score: {score}/{len(questions)}</p>", unsafe_allow_html=True)
        if score == len(questions):
            st.balloons()
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're an Information Gain expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering Information Gain. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)
