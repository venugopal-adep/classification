import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd

# Set page config
st.set_page_config(layout="wide", page_title="Gini Index Explorer", page_icon="🌳")

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
st.markdown("<h1 class='main-header'>🌳 Gini Index Interactive Explorer 🌳</h1>", unsafe_allow_html=True)
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

# Calculate Gini index for the entire dataset
class_counts = np.bincount(y)
class_probabilities = class_counts / n_samples
gini_index = 1 - np.sum(class_probabilities ** 2)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎓 Learn", "🔍 Gini Index Explorer & Decision Tree", "🧠 Quiz"])

with tab1:
    st.markdown("<h2 class='tab-subheader'>Learn About Gini Index</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    The Gini index is a measure of impurity used in decision trees to determine the best split at each node. It quantifies the probability of incorrectly classifying a randomly chosen element in the dataset if it were randomly labeled according to the distribution of classes in the subset.

    <b>Key points about the Gini Index:</b>
    1. It ranges from 0 to 1-1/k, where k is the number of classes.
    2. A value of 0 indicates perfect purity (all samples belong to the same class).
    3. A value close to 1-1/k indicates high impurity (samples are evenly distributed among classes).
    4. The decision tree algorithm aims to minimize the Gini index at each split.

    <b>Gini Index Formula:</b>
    </p>
    """, unsafe_allow_html=True)

    st.latex(r"Gini(t) = 1 - \sum_{i=1}^{c} p_i^2")

    st.markdown("""
    <p class='content-text'>
    Where:
    - t is the current node
    - c is the number of classes
    - p_i is the proportion of samples belonging to class i at node t

    The decision tree algorithm selects the feature and split point that minimizes the weighted average of the Gini index of the child nodes. This process is repeated recursively until a stopping criterion is met, such as reaching the maximum depth or having a minimum number of samples in a leaf node.

    By minimizing the Gini index at each split, the decision tree aims to create pure leaf nodes, leading to accurate classifications.
    </p>
    """, unsafe_allow_html=True)

    # Conclusion
    st.markdown("<h2 class='tab-subheader'>Explore and Learn! 🚀</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='content-text'>
    You've explored the world of Gini Index in Decision Trees! Remember:

    1. The Gini index measures the impurity of nodes in a decision tree.
    2. It ranges from 0 (perfect purity) to 1-1/k (maximum impurity), where k is the number of classes.
    3. Decision trees use the Gini index to find the best splits at each node.
    4. Minimizing the Gini index helps create more accurate and efficient decision trees.
    5. Understanding the Gini index is crucial for interpreting and optimizing decision tree models.

    Keep exploring and applying these concepts to build more robust and efficient Decision Tree models!
    </p>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("<h2 class='tab-subheader'>Decision Tree Visualization</h2>", unsafe_allow_html=True)
    
    # Create decision tree classifier
    dt = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, random_state=42)
    dt.fit(X, y)

    # Plot the decision tree
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(dt, filled=True, rounded=True, class_names=[f"Class {i}" for i in range(n_classes)], 
              feature_names=[f"Feature {i}" for i in range(n_features)])
    st.pyplot(fig)

    st.markdown("<p class='content-text'>This decision tree visualization shows how the Gini index is used to make splits at each node. The Gini index values are displayed in each node, helping you understand the decision-making process.</p>", unsafe_allow_html=True)

with tab3:
    st.markdown("<h2 class='tab-subheader'>Test Your Gini Index Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What does the Gini index measure in decision trees?",
            "options": [
                "The depth of the tree",
                "The impurity of a node",
                "The number of features",
                "The accuracy of the model"
            ],
            "correct": 1,
            "explanation": "The Gini index measures the impurity of a node in a decision tree. It quantifies how mixed the classes are within a node."
        },
        {
            "question": "What does a Gini index of 0 indicate?",
            "options": [
                "Perfect impurity",
                "Perfect purity",
                "Equal distribution of classes",
                "Insufficient data"
            ],
            "correct": 1,
            "explanation": "A Gini index of 0 indicates perfect purity, meaning all samples in the node belong to the same class."
        },
        {
            "question": "How does the decision tree algorithm use the Gini index?",
            "options": [
                "To maximize impurity at each split",
                "To minimize impurity at each split",
                "To determine the depth of the tree",
                "To select the number of features"
            ],
            "correct": 1,
            "explanation": "The decision tree algorithm uses the Gini index to minimize impurity at each split, aiming to create the purest possible child nodes."
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
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're a Gini Index expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering the Gini Index. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)


