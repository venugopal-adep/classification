import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd

# Set page config
st.set_page_config(layout="wide", page_title="Entropy Explorer", page_icon="🌳")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px !important;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .tab-subheader {
        font-size: 28px !important;
        font-weight: bold;
        color: #3CB371;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .content-text {
        font-size: 18px !important;
        line-height: 1.6;
    }
    .stButton>button {
        background-color: #66CDAA;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #3CB371;
        transform: scale(1.05);
    }
    .highlight {
        background-color: #E0FFF0;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .quiz-question {
        background-color: #F0FFF0;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #3CB371;
    }
    .explanation {
        background-color: #F0FFF0;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🌳 Entropy Interactive Explorer 🌳</h1>", unsafe_allow_html=True)
st.markdown("<p class='content-text'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Entropy Explorer", "📊 Decision Tree Visualization", "🎓 Learn More", "🧠 Quiz"])

with tab1:
    st.markdown("<h2 class='tab-subheader'>Entropy Calculation and Visualization</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("<h3 class='content-text'>Dataset Parameters</h3>", unsafe_allow_html=True)
        
        # Dataset parameters
        n_samples = st.slider("Number of samples", min_value=50, max_value=500, value=200, step=50)
        n_features = st.slider("Number of features", min_value=2, max_value=10, value=5, step=1)
        n_classes = st.slider("Number of classes", min_value=2, max_value=5, value=2, step=1)

        # Generate random data
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, 
                                   n_informative=n_features, n_redundant=0, random_state=42)

        # Calculate entropy for the entire dataset
        class_counts = np.bincount(y)
        class_probabilities = class_counts / n_samples
        entropy = -np.sum(class_probabilities * np.log2(class_probabilities + 1e-10))

        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='content-text'>Dataset Entropy: {entropy:.4f}</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 class='content-text'>Class Distribution</h3>", unsafe_allow_html=True)
        
        class_distribution = pd.DataFrame({'Class': range(n_classes), 'Count': class_counts})
        fig = px.bar(class_distribution, x='Class', y='Count', 
                     title='Class Distribution in the Dataset',
                     labels={'Count': 'Number of Samples', 'Class': 'Class Label'})
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("<h2 class='tab-subheader'>Decision Tree Visualization</h2>", unsafe_allow_html=True)
    
    max_depth = st.slider("Maximum depth of the decision tree", min_value=1, max_value=10, value=3, step=1)

    # Create decision tree classifier
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
    dt.fit(X, y)

    # Plot the decision tree
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(dt, filled=True, rounded=True, class_names=[f"Class {i}" for i in range(n_classes)], 
              feature_names=[f"Feature {i}" for i in range(n_features)])
    st.pyplot(fig)

    st.markdown("<p class='content-text'>This decision tree visualization shows how entropy is used to make splits at each node. The entropy values are displayed in each node, helping you understand the decision-making process.</p>", unsafe_allow_html=True)

with tab3:
    st.markdown("<h2 class='tab-subheader'>Learn More About Entropy</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    Entropy is a measure of impurity or uncertainty used in decision trees to determine the best split at each node. It quantifies the amount of information contained in a set of class labels.

    <b>Key points about Entropy:</b>
    1. It ranges from 0 to log2(k), where k is the number of classes.
    2. A value of 0 indicates perfect purity (all samples belong to the same class).
    3. A value of log2(k) indicates maximum impurity (samples are evenly distributed among classes).
    4. The decision tree algorithm aims to minimize entropy at each split.

    <b>Entropy Formula:</b>
    </p>
    """, unsafe_allow_html=True)

    st.latex(r"Entropy(t) = -\sum_{i=1}^{c} p_i \log_2(p_i)")

    st.markdown("""
    <p class='content-text'>
    Where:
    - t is the current node
    - c is the number of classes
    - p_i is the proportion of samples belonging to class i at node t

    The decision tree algorithm selects the feature and split point that maximizes the information gain, which is the difference between the entropy of the parent node and the weighted average of the entropies of the child nodes. This process is repeated recursively until a stopping criterion is met, such as reaching the maximum depth or having a minimum number of samples in a leaf node.

    By minimizing entropy at each split, the decision tree aims to create pure leaf nodes, leading to accurate classifications.
    </p>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown("<h2 class='tab-subheader'>Test Your Entropy Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What does entropy measure in decision trees?",
            "options": [
                "The depth of the tree",
                "The impurity or uncertainty of a node",
                "The number of features",
                "The accuracy of the model"
            ],
            "correct": 1,
            "explanation": "Entropy measures the impurity or uncertainty of a node in a decision tree. It quantifies the amount of information contained in a set of class labels."
        },
        {
            "question": "What is the range of entropy for binary classification?",
            "options": [
                "0 to 1",
                "-1 to 1",
                "0 to log2(2)",
                "0 to infinity"
            ],
            "correct": 0,
            "explanation": "For binary classification, entropy ranges from 0 to 1. In general, it ranges from 0 to log2(k), where k is the number of classes."
        },
        {
            "question": "What does an entropy of 0 indicate?",
            "options": [
                "Maximum impurity",
                "Perfect purity",
                "Equal distribution of classes",
                "Insufficient data"
            ],
            "correct": 1,
            "explanation": "An entropy of 0 indicates perfect purity, meaning all samples in the node belong to the same class."
        },
        {
            "question": "How does the decision tree algorithm use entropy?",
            "options": [
                "To maximize impurity at each split",
                "To minimize impurity at each split",
                "To determine the depth of the tree",
                "To select the number of features"
            ],
            "correct": 1,
            "explanation": "The decision tree algorithm uses entropy to minimize impurity at each split by maximizing the information gain, aiming to create the purest possible child nodes."
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
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're an Entropy expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering Entropy. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)

# Conclusion
st.markdown("<h2 class='tab-subheader'>Explore and Learn! 🚀</h2>", unsafe_allow_html=True)
st.markdown("""
<p class='content-text'>
You've explored the world of Entropy in Decision Trees! Remember:

1. Entropy measures the impurity or uncertainty of nodes in a decision tree.
2. It ranges from 0 (perfect purity) to log2(k) (maximum impurity), where k is the number of classes.
3. Decision trees use entropy to find the best splits at each node by maximizing information gain.
4. Minimizing entropy helps create more accurate and efficient decision trees.
5. Understanding entropy is crucial for interpreting and optimizing decision tree models.

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
