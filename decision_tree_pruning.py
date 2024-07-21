import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd

# Set page config
st.set_page_config(layout="wide", page_title="Decision Tree Pruning Explorer", page_icon="✂️")

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
st.markdown("<h1 class='main-header'>✂️ Decision Tree Pruning Interactive Tool ✂️</h1>", unsafe_allow_html=True)
st.markdown("<p class='content-text'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)

def plot_decision_tree(clf, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names, ax=ax)
    st.pyplot(fig)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🌳 Tree Pruning Explorer", "📊 Model Performance", "🎓 Learn More", "🧠 Quiz"])

with tab1:
    st.markdown("<h2 class='tab-subheader'>Decision Tree Pruning Visualization and Configuration</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("<h3 class='content-text'>Dataset and Pruning Parameters</h3>", unsafe_allow_html=True)
        
        # Dataset selection
        dataset_options = ['Iris', 'Wine', 'Breast Cancer']
        selected_dataset = st.selectbox("Select a dataset", dataset_options)

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
        criterion = st.selectbox("Splitting criterion", ['gini', 'entropy'])
        max_depth = st.slider("Maximum depth", min_value=1, max_value=10, step=1, value=3)
        min_samples_split = st.slider("Minimum samples split", min_value=2, max_value=20, step=1, value=2)
        min_samples_leaf = st.slider("Minimum samples leaf", min_value=1, max_value=20, step=1, value=1)
        ccp_alpha = st.slider("Complexity parameter (ccp_alpha)", min_value=0.0, max_value=0.1, step=0.001, value=0.0)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Decision Tree model
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)

    with col2:
        st.markdown("<h3 class='content-text'>Pruned Decision Tree Visualization</h3>", unsafe_allow_html=True)
        plot_decision_tree(clf, feature_names, class_names)

with tab2:
    st.markdown("<h2 class='tab-subheader'>Model Performance</h2>", unsafe_allow_html=True)
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.markdown(f"<h3 class='content-text'>Test Accuracy: {accuracy:.2f}</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Feature importance
    feature_importance = clf.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    fig = px.bar(feature_importance_df, x='importance', y='feature', orientation='h',
                 title='Feature Importance', labels={'importance': 'Importance', 'feature': 'Feature'})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("<h2 class='tab-subheader'>Learn More About Decision Tree Pruning</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    Decision Tree Pruning is a technique used to simplify and optimize decision trees by removing unnecessary branches or nodes. The goal of pruning is to reduce overfitting and improve the generalization performance of the tree.

    There are two main approaches to pruning:
    1. <b>Pre-pruning</b>: Stopping the tree-growing process early based on certain criteria, such as maximum depth or minimum samples per leaf.
    2. <b>Post-pruning</b>: Growing a full tree and then removing or collapsing branches based on their impact on validation or test set performance.

    Pruning helps to find the right balance between model complexity and generalization ability. It reduces the size of the tree, making it more interpretable and less prone to overfitting.

    <b>Understanding Pruning Parameters:</b>
    - <b>Splitting criterion</b>: The function used to measure the quality of a split. Gini impurity and entropy are common criteria.
    - <b>Maximum depth</b>: The maximum depth allowed for the Decision Tree. Limiting the depth helps to prevent overfitting.
    - <b>Minimum samples split</b>: The minimum number of samples required to split an internal node. Higher values prevent overfitting by requiring more samples to make a split.
    - <b>Minimum samples leaf</b>: The minimum number of samples required to be at a leaf node. Higher values create simpler trees by forcing more samples into each leaf.
    - <b>Complexity parameter (ccp_alpha)</b>: The threshold used for post-pruning. Nodes with a cost complexity lower than this value are pruned. Higher values result in more aggressive pruning.

    Experiment with different pruning parameters to observe their impact on the tree structure and performance. Finding the right balance is key to achieving a model that generalizes well to unseen data.
    </p>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown("<h2 class='tab-subheader'>Test Your Decision Tree Pruning Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What is the main goal of Decision Tree Pruning?",
            "options": [
                "To increase the depth of the tree",
                "To reduce overfitting and improve generalization",
                "To increase the number of leaf nodes",
                "To make the tree more complex"
            ],
            "correct": 1,
            "explanation": "The main goal of Decision Tree Pruning is to reduce overfitting and improve the tree's ability to generalize to unseen data by simplifying the tree structure."
        },
        {
            "question": "Which of the following is NOT a pruning parameter in this tool?",
            "options": [
                "Maximum depth",
                "Minimum samples split",
                "Complexity parameter (ccp_alpha)",
                "Learning rate"
            ],
            "correct": 3,
            "explanation": "Learning rate is not a pruning parameter for Decision Trees. It is commonly used in other algorithms like Gradient Boosting."
        },
        {
            "question": "What does the 'ccp_alpha' parameter control in Decision Tree pruning?",
            "options": [
                "The maximum depth of the tree",
                "The minimum number of samples required to split a node",
                "The threshold for cost-complexity pruning",
                "The splitting criterion"
            ],
            "correct": 2,
            "explanation": "The 'ccp_alpha' parameter controls the threshold for cost-complexity pruning. Higher values result in more aggressive pruning."
        },
        {
            "question": "Which approach to pruning stops the tree-growing process early based on certain criteria?",
            "options": [
                "Pre-pruning",
                "Post-pruning",
                "Mid-pruning",
                "Adaptive pruning"
            ],
            "correct": 0,
            "explanation": "Pre-pruning stops the tree-growing process early based on certain criteria, such as maximum depth or minimum samples per leaf."
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
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're a Decision Tree Pruning expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering Decision Tree Pruning. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)

# Conclusion
st.markdown("<h2 class='tab-subheader'>Explore and Learn! 🚀</h2>", unsafe_allow_html=True)
st.markdown("""
<p class='content-text'>
You've explored the world of Decision Tree Pruning! Remember:

1. Pruning helps reduce overfitting and improve generalization in Decision Trees.
2. Pre-pruning and post-pruning are two main approaches to tree pruning.
3. Parameters like maximum depth, minimum samples split, and ccp_alpha control the pruning process.
4. Experiment with different pruning parameters to find the right balance between model complexity and performance.
5. Visualizing the pruned tree helps in understanding the impact of different pruning strategies.

Keep exploring and applying these concepts to build more robust and efficient Decision Tree models!
</p>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0E1117;
    color: #FAFAFA;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>
<div class='footer'>
    Created with ❤️ by Venugopal Adep | © 2023 All Rights Reserved
</div>
""", unsafe_allow_html=True)
