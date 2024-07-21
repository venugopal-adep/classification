import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Set page config
st.set_page_config(layout="wide", page_title="Decision Tree vs Random Forest Explorer", page_icon="🌳")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px !important;
        font-weight: bold;
        color: #FF6347;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .tab-subheader {
        font-size: 28px !important;
        font-weight: bold;
        color: #FF7F50;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .content-text {
        font-size: 18px !important;
        line-height: 1.6;
    }
    .stButton>button {
        background-color: #FFA07A;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF6347;
        transform: scale(1.05);
    }
    .highlight {
        background-color: #FFE4E1;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .quiz-question {
        background-color: #FFF0F5;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #FF6347;
    }
    .explanation {
        background-color: #FFF5EE;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.title("Model Parameters")
dataset = st.sidebar.selectbox("Select dataset", ("Moons", "Circles", "Linearly Separable"))
n_samples = st.sidebar.slider("Number of samples", 50, 500, 200, 50)
noise = st.sidebar.slider("Noise", 0.0, 0.5, 0.2, 0.1)
max_depth = st.sidebar.slider("Maximum depth", 1, 10, 3, 1)
n_estimators = st.sidebar.slider("Number of trees (Random Forest)", 1, 100, 10, 1)

# Generate dataset
if dataset == "Moons":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
elif dataset == "Circles":
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
else:
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2,
                               random_state=42, n_clusters_per_class=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
rf.fit(X_train, y_train)

# Title
st.markdown("<h1 class='main-header'>🌳 Decision Tree vs Random Forest Interactive Explorer 🌳</h1>", unsafe_allow_html=True)
st.markdown("<p class='content-text'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Model Explorer", "📊 Performance Comparison", "🎓 Learn More", "🧠 Quiz"])

with tab1:
    st.markdown("<h2 class='tab-subheader'>Model Configuration and Visualization</h2>", unsafe_allow_html=True)

    st.markdown("<h3 class='content-text'>Decision Boundaries Visualization</h3>", unsafe_allow_html=True)
    
    # Plot decision boundaries
    fig = go.Figure()

    # Add scatter points for the data
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', 
                             marker=dict(color=y, colorscale='Viridis'), 
                             name='Data Points'))

    # Create mesh grid for decision boundary visualization
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Make predictions on the mesh grid
    Z_dt = dt.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    Z_rf = rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Add contour plots for decision boundaries
    fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02), 
                             z=Z_dt, name='Decision Tree', showscale=False, 
                             colorscale='RdBu', opacity=0.5))
    fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02), 
                             z=Z_rf, name='Random Forest', showscale=False, 
                             colorscale='RdYlBu', opacity=0.5))

    fig.update_layout(title=f'Decision Boundaries - {dataset} Dataset',
                      xaxis_title='Feature 1', yaxis_title='Feature 2')

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("<h2 class='tab-subheader'>Model Performance Comparison</h2>", unsafe_allow_html=True)
    
    # Calculate accuracies
    dt_pred = dt.predict(X_test)
    rf_pred = rf.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='content-text'>Decision Tree Accuracy: {dt_accuracy:.3f}</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='content-text'>Random Forest Accuracy: {rf_accuracy:.3f}</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Accuracy comparison bar chart
    accuracy_data = pd.DataFrame({
        'Model': ['Decision Tree', 'Random Forest'],
        'Accuracy': [dt_accuracy, rf_accuracy]
    })
    fig = px.bar(accuracy_data, x='Model', y='Accuracy', 
                 title='Model Accuracy Comparison', 
                 color='Model', color_discrete_map={'Decision Tree': '#FF6347', 'Random Forest': '#4682B4'})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("<h2 class='tab-subheader'>Learn More About Decision Trees and Random Forests</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    <b>Decision Trees:</b>
    - Single tree-like model that makes decisions based on a series of rules.
    - Recursively splits the data based on the most informative features.
    - Easy to interpret and visualize.
    - Prone to overfitting, especially with deep trees.

    <b>Random Forests:</b>
    - Ensemble of multiple decision trees.
    - Combines predictions from multiple trees to make the final prediction.
    - Introduces randomness by using random subsets of features and samples for each tree.
    - More robust and accurate than individual decision trees.
    - Less interpretable than single decision trees.

    <b>Key Differences:</b>
    1. Complexity: Decision trees are simpler, while random forests are more complex.
    2. Performance: Random forests generally outperform decision trees, especially on complex datasets.
    3. Overfitting: Random forests are less prone to overfitting compared to deep decision trees.
    4. Interpretability: Decision trees are more easily interpretable than random forests.
    5. Computational resources: Random forests require more computational power and memory.

    Random forests tend to perform better than decision trees in most scenarios, but decision trees are still valuable for their simplicity and interpretability.
    </p>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown("<h2 class='tab-subheader'>Test Your Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "Which model is an ensemble of multiple decision trees?",
            "options": [
                "Decision Tree",
                "Random Forest",
                "Logistic Regression",
                "Support Vector Machine"
            ],
            "correct": 1,
            "explanation": "Random Forest is an ensemble model that consists of multiple decision trees."
        },
        {
            "question": "Which model is generally more prone to overfitting?",
            "options": [
                "Decision Tree",
                "Random Forest",
                "Both equally",
                "Neither"
            ],
            "correct": 0,
            "explanation": "Decision Trees, especially deep ones, are more prone to overfitting compared to Random Forests."
        },
        {
            "question": "What does the 'n_estimators' parameter control in a Random Forest?",
            "options": [
                "The maximum depth of each tree",
                "The number of trees in the forest",
                "The number of features to consider for each split",
                "The minimum number of samples required to split a node"
            ],
            "correct": 1,
            "explanation": "The 'n_estimators' parameter controls the number of trees in the Random Forest."
        },
        {
            "question": "Which model is generally more interpretable?",
            "options": [
                "Decision Tree",
                "Random Forest",
                "Both equally",
                "Neither"
            ],
            "correct": 0,
            "explanation": "Decision Trees are generally more interpretable than Random Forests due to their single tree structure."
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
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're a Decision Tree and Random Forest expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering Decision Trees and Random Forests. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)

# Conclusion
st.markdown("<h2 class='tab-subheader'>Explore and Learn! 🚀</h2>", unsafe_allow_html=True)
st.markdown("""
<p class='content-text'>
You've explored the world of Decision Trees and Random Forests! Remember:

1. Decision Trees are simple and interpretable, while Random Forests are more complex and powerful.
2. Random Forests generally outperform Decision Trees, especially on complex datasets.
3. The choice between Decision Trees and Random Forests depends on your specific needs for interpretability vs. performance.
4. Experiment with different parameters to see how they affect model performance and decision boundaries.
5. Both models have their place in machine learning, and understanding their strengths and weaknesses is key to applying them effectively.

Keep exploring and applying these concepts to build more robust and efficient machine learning models!
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
