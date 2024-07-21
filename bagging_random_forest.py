import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Set page config
st.set_page_config(layout="wide", page_title="Bagging vs Random Forest Explorer", page_icon="🌳")

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
    .interpretation {
        background-color: #F0E6FF;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #8A2BE2;
    }
    .quiz-question {
        background-color: #F0E6FF;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #8A2BE2;
    }
    .explanation {
        background-color: #E6E6FA;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🌳 Bagging vs Random Forest Interactive Explorer 🌳</h1>", unsafe_allow_html=True)
st.markdown("<p class='content-text'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.title("Model Parameters")
n_estimators = st.sidebar.slider("Number of Estimators", min_value=1, max_value=100, value=10, step=1)
n_samples = st.sidebar.slider("Number of samples", 100, 1000, 500, 50)
noise = st.sidebar.slider("Noise", 0.0, 0.5, 0.3, 0.1)

# Generate dataset
X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
bagging = BaggingClassifier(n_estimators=n_estimators, random_state=42)
bagging.fit(X_train, y_train)
bagging_accuracy = accuracy_score(y_test, bagging.predict(X_test))

rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
rf.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf.predict(X_test))

def plot_decision_boundary(clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    
    # Add decision boundary contour
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z,
        colorscale='RdBu',
        opacity=0.5,
        showscale=False,
        contours=dict(start=0, end=1, size=0.5)
    ))
    
    # Add scatter plot for data points
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode='markers',
        marker=dict(
            size=10,
            color=y,
            colorscale='RdBu',
            line=dict(color='Black', width=1)
        ),
        showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        width=700,
        height=600,
        autosize=False,
        margin=dict(l=50, r=50, b=50, t=50, pad=4)
    )
    
    return fig

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Model Explorer", "📊 Performance Comparison", "🎓 Learn More", "🧠 Quiz"])

with tab1:
    st.markdown("<h2 class='tab-subheader'>Model Visualization</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_decision_boundary(bagging, X, y, f"Bagging (Accuracy: {bagging_accuracy:.2f})"), use_container_width=True)
    with col2:
        st.plotly_chart(plot_decision_boundary(rf, X, y, f"Random Forest (Accuracy: {rf_accuracy:.2f})"), use_container_width=True)

    st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
    st.markdown("""
    <p class='content-text'>
    <strong>Interpretation:</strong>
    - The decision boundaries (colored regions) show how the classifiers separate the two classes in the feature space.
    - Data points are represented by circles, with colors indicating their true class.
    - A more complex and irregular decision boundary indicates a more flexible model.
    - As you increase the number of estimators, the decision boundaries typically become smoother and more stable.
    - Random Forest often performs better than Bagging due to the additional randomness introduced by feature subsampling.
    - The accuracy scores provide a quantitative measure of how well the classifiers perform on the test set.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<h2 class='tab-subheader'>Model Performance Comparison</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='content-text'>Bagging Accuracy: {bagging_accuracy:.3f}</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='content-text'>Random Forest Accuracy: {rf_accuracy:.3f}</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Accuracy comparison bar chart
    accuracy_data = pd.DataFrame({
        'Model': ['Bagging', 'Random Forest'],
        'Accuracy': [bagging_accuracy, rf_accuracy]
    })
    fig = px.bar(accuracy_data, x='Model', y='Accuracy', 
                 title='Model Accuracy Comparison', 
                 color='Model', color_discrete_map={'Bagging': '#9370DB', 'Random Forest': '#8A2BE2'})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("<h2 class='tab-subheader'>Learn More About Bagging and Random Forest</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    Bagging (Bootstrap Aggregating) and Random Forest are ensemble learning methods that combine multiple decision trees to improve prediction accuracy and reduce overfitting.

    <b>Bagging:</b>
    - Bagging creates multiple subsets of the training data by randomly sampling with replacement (bootstrap samples).
    - It trains a decision tree on each subset independently.
    - The final prediction is obtained by aggregating the predictions of all the trees (majority voting for classification, averaging for regression).

    <b>Random Forest:</b>
    - Random Forest is an extension of the bagging technique.
    - In addition to bootstrapping the data, it also selects a random subset of features at each split in the decision trees.
    - This further reduces the correlation between the trees and improves the diversity of the ensemble.

    <b>Key Differences:</b>
    1. Feature Selection: Bagging uses all features, while Random Forest selects a random subset of features for each split.
    2. Diversity: Random Forest typically achieves greater diversity among its trees due to the feature subsampling.
    3. Performance: Random Forest often outperforms Bagging, especially on high-dimensional datasets.
    4. Interpretability: Both methods are less interpretable than single decision trees, but Random Forest provides feature importance measures.

    In this demo, you can observe the decision boundaries of Bagging and Random Forest classifiers on a toy dataset. Adjust the number of estimators (trees) and see how the decision boundary changes.
    </p>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown("<h2 class='tab-subheader'>Test Your Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What is the main difference between Bagging and Random Forest?",
            "options": [
                "Bagging uses decision trees, while Random Forest uses linear models",
                "Random Forest introduces additional randomness through feature subsampling",
                "Bagging can only be used for classification, while Random Forest can be used for both classification and regression",
                "Random Forest uses boosting, while Bagging does not"
            ],
            "correct": 1,
            "explanation": "The main difference is that Random Forest introduces additional randomness by selecting a random subset of features at each split in the decision trees, while Bagging uses all features."
        },
        {
            "question": "What does the term 'bootstrap' refer to in Bagging?",
            "options": [
                "A method for initializing model parameters",
                "A technique for feature scaling",
                "Random sampling with replacement to create subsets of the training data",
                "A way to prune decision trees"
            ],
            "correct": 2,
            "explanation": "In Bagging, 'bootstrap' refers to the process of creating multiple subsets of the training data by randomly sampling with replacement."
        },
        {
            "question": "How do Bagging and Random Forest typically compare in terms of performance?",
            "options": [
                "Bagging always outperforms Random Forest",
                "Random Forest often outperforms Bagging, especially on high-dimensional datasets",
                "They always perform exactly the same",
                "Bagging is better for small datasets, Random Forest for large datasets"
            ],
            "correct": 1,
            "explanation": "Random Forest often outperforms Bagging, especially on high-dimensional datasets, due to the additional randomness introduced by feature subsampling."
        },
        {
            "question": "What happens to the decision boundary as you increase the number of estimators (trees) in Bagging or Random Forest?",
            "options": [
                "It becomes more complex and irregular",
                "It becomes smoother and more stable",
                "It doesn't change at all",
                "It always becomes a straight line"
            ],
            "correct": 1,
            "explanation": "As you increase the number of estimators, the decision boundaries typically become smoother and more stable due to the aggregation of multiple trees' predictions."
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
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're a Bagging and Random Forest expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering Bagging and Random Forest. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)

# Conclusion
st.markdown("<h2 class='tab-subheader'>Explore and Learn! 🚀</h2>", unsafe_allow_html=True)
st.markdown("""
<p class='content-text'>
You've explored the world of Bagging and Random Forest! Remember:

1. Both Bagging and Random Forest are ensemble methods that combine multiple decision trees.
2. Random Forest introduces additional randomness through feature subsampling, often leading to better performance.
3. Increasing the number of estimators generally improves model stability and performance, but with diminishing returns.
4. The choice between Bagging and Random Forest depends on your specific dataset and problem.
5. Experiment with different parameters to see how they affect model performance and decision boundaries.

Keep exploring and applying these concepts to build more robust and efficient machine learning models!
</p>
""", unsafe_allow_html=True)

