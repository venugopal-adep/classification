import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd

# Set page config
st.set_page_config(layout="wide", page_title="Decision Tree Explorer", page_icon="🌳")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px !important;
        font-weight: bold;
        color: #4B0082;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .tab-subheader {
        font-size: 28px !important;
        font-weight: bold;
        color: #8A2BE2;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .content-text {
        font-size: 18px !important;
        line-height: 1.6;
    }
    .stButton>button {
        background-color: #9370DB;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #8A2BE2;
        transform: scale(1.05);
    }
    .highlight {
        background-color: #E6E6FA;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .quiz-question {
        background-color: #F0E6FA;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #8A2BE2;
    }
    .explanation {
        background-color: #E6F3FF;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🌳 Decision Tree Interactive Tool 🌳</h1>", unsafe_allow_html=True)
st.markdown("<p class='content-text'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)

def plot_decision_tree(clf, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names, ax=ax)
    st.pyplot(fig)

# Sidebar
st.sidebar.markdown("<h3 class='content-text'>Dataset and Parameters</h3>", unsafe_allow_html=True)

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

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎓 Learn", "🔍 Decision Tree Explorer", "🧠 Quiz"])

with tab1:
    st.markdown("<h2 class='tab-subheader'>Learn About Decision Trees</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    Decision Trees are a popular machine learning algorithm used for both classification and regression tasks. They are tree-like models that make decisions based on a series of rules learned from the input features.

    Key concepts in Decision Trees:
    1. <b>Nodes:</b> The points in the tree where a decision is made based on a feature.
    2. <b>Edges:</b> The branches connecting the nodes, representing the decision paths.
    3. <b>Leaves:</b> The final nodes in the tree, representing the predicted class or value.

    Decision Trees learn the optimal rules by recursively splitting the data based on the feature that provides the most information gain or reduces the impurity the most. The splitting process continues until a stopping criterion is met, such as reaching a maximum depth or having a minimum number of samples in a leaf.

    <b>Interpreting the Decision Tree Plot:</b>
    - Each node shows the feature and threshold used for splitting, along with the Gini impurity and the number of samples reaching that node.
    - The color of the node represents the majority class, with darker shades indicating higher purity.
    - The leaves show the predicted class and the number of samples in that leaf.

    Advantages of Decision Trees:
    - Easy to understand and interpret
    - Can handle both numerical and categorical data
    - Requires little data preprocessing
    - Can capture non-linear relationships

    Disadvantages:
    - Prone to overfitting, especially with deep trees
    - Can be unstable (small changes in data can result in very different trees)
    - Biased towards features with more levels in categorical variables

    Decision Trees are widely used in various applications, including finance, healthcare, and marketing, due to their interpretability and ability to handle complex decision-making processes.
    </p>
    """, unsafe_allow_html=True)

    # Conclusion
    st.markdown("<h2 class='tab-subheader'>Explore and Learn! 🚀</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='content-text'>
    You've explored the world of Decision Trees! Remember:

    1. Decision Trees are powerful and interpretable models for classification and regression.
    2. They make decisions based on a series of rules learned from the input features.
    3. The tree structure allows for easy visualization and understanding of the decision-making process.
    4. Hyperparameters like maximum depth and minimum samples split can help control the model's complexity.
    5. While powerful, Decision Trees can be prone to overfitting, so it's important to tune them carefully.

    Keep exploring and applying these concepts in your machine learning journey!
    </p>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("<h2 class='tab-subheader'>Decision Tree Visualization and Configuration</h2>", unsafe_allow_html=True)

    st.markdown("<h3 class='content-text'>Model Performance</h3>", unsafe_allow_html=True)
    st.markdown(f"<p class='content-text'><strong>Accuracy:</strong> {accuracy:.2f} | <strong>Precision:</strong> {precision:.2f} | <strong>Recall:</strong> {recall:.2f} | <strong>F1-score:</strong> {f1:.2f}</p>", unsafe_allow_html=True)
    
    st.markdown("<h3 class='content-text'>Decision Tree Visualization</h3>", unsafe_allow_html=True)
    plot_decision_tree(clf, feature_names, class_names)

    # Feature importance
    feature_importance = clf.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    fig = px.bar(feature_importance_df, x='importance', y='feature', orientation='h',
                 title='Feature Importance', labels={'importance': 'Importance', 'feature': 'Feature'})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("<h2 class='tab-subheader'>Test Your Decision Tree Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What is the main advantage of Decision Trees?",
            "options": [
                "They always produce the highest accuracy",
                "They are easy to understand and interpret",
                "They can only handle numerical data",
                "They are immune to overfitting"
            ],
            "correct": 1,
            "explanation": "Decision Trees are popular because they are easy to understand and interpret, making them valuable for decision-making processes in various fields."
        },
        {
            "question": "What does the 'maximum depth' parameter control in a Decision Tree?",
            "options": [
                "The number of features used",
                "The number of samples in each leaf",
                "The maximum number of levels in the tree",
                "The minimum number of samples required to split a node"
            ],
            "correct": 2,
            "explanation": "The 'maximum depth' parameter controls the maximum number of levels allowed in the tree, which helps prevent overfitting by limiting the tree's complexity."
        },
        {
            "question": "What is a leaf node in a Decision Tree?",
            "options": [
                "The root node of the tree",
                "A node that splits the data",
                "A node with no children, representing a final decision or prediction",
                "A node that contains only one sample"
            ],
            "correct": 2,
            "explanation": "A leaf node is a terminal node in the Decision Tree that has no children. It represents a final decision or prediction for the samples that reach that node."
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
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're a Decision Tree expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering Decision Trees. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)

