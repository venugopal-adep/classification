import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

def main():
    st.set_page_config(page_title="Feature Importance Analysis", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
    .stButton>button {
        background-color: #ff6e40;
        color: white;
    }
    .stButton>button:hover {
        background-color: #ff9e80;
    }
    .quiz-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔍 Feature Importance Analysis")
    
    tabs = st.tabs(["📚 Learn", "🧪 Experiment", "📊 Real Data Example", "🧠 Quiz"])
    
    with tabs[0]:
        learn_section()
    
    with tabs[1]:
        experiment_section()
    
    with tabs[2]:
        real_data_section()
    
    with tabs[3]:
        quiz_section()

def learn_section():
    st.header("What is Feature Importance?")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        Feature importance refers to techniques that assign a score to input features based on how useful they are at predicting a target variable.

        Feature importance is a powerful tool that can be used for:
        - Understanding your data
        - Feature selection
        - Model interpretation
        """)
    
    with col2:
        st.image("https://machinelearningmastery.com/wp-content/uploads/2019/11/Plot-of-Feature-Importance-Scores-for-the-Diabetes-Dataset.png", caption="Example of Feature Importance Visualization")
    
    st.subheader("Common Methods for Calculating Feature Importance")
    st.write("""
    1. **Built-in Feature Importance**: Many tree-based models (like Random Forests) provide built-in feature importance scores.
    
    2. **Permutation Importance**: Measures the increase in the model's prediction error after permuting the feature's values.
    
    3. **SHAP (SHapley Additive exPlanations)**: A game theoretic approach to explain the output of any machine learning model.
    
    4. **Correlation Coefficient**: Measures the linear correlation between features and the target variable.
    
    5. **Recursive Feature Elimination**: Recursively removes features and builds a model on those features that remain.
    """)

def experiment_section():
    st.header("🧪 Interactive Feature Importance Experiment")
    
    st.write("""
    In this experiment, we'll simulate a dataset with controllable feature importances and visualize how changing these importances affects the model's feature importance scores.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Adjust Feature Importances")
        feature1_imp = st.slider("Feature 1 Importance", 0.0, 1.0, 0.5, 0.1)
        feature2_imp = st.slider("Feature 2 Importance", 0.0, 1.0, 0.3, 0.1)
        feature3_imp = st.slider("Feature 3 Importance", 0.0, 1.0, 0.1, 0.1)
        noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1, 0.1)
        
        n_samples = st.number_input("Number of Samples", 100, 10000, 1000, 100)
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(n_samples, 3)
    y = (feature1_imp * X[:, 0] + 
         feature2_imp * X[:, 1] + 
         feature3_imp * X[:, 2] + 
         noise_level * np.random.randn(n_samples))
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    feature_names = ['Feature 1', 'Feature 2', 'Feature 3']
    
    with col2:
        fig = px.bar(x=feature_names, y=importances, 
                     labels={'x': 'Features', 'y': 'Importance'},
                     title='Feature Importance Scores')
        fig.update_traces(marker_color='#ff6e40')
        st.plotly_chart(fig)
    
    st.write("""
    💡 **Interpretation:**
    - The bars show the importance of each feature as determined by the Random Forest model.
    - Higher bars indicate more important features.
    - Try adjusting the sliders to see how changing the true feature importances and noise level affects the model's feature importance scores.
    """)

def real_data_section():
    st.header("📊 Real Data Example: California Housing Dataset")
    
    # Load California Housing dataset
    california = fetch_california_housing()
    X, y = california.data, california.target
    feature_names = california.feature_names
    
    # Train a Random Forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort features by importance
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Visualize feature importances
    fig = px.bar(feature_importance, x='feature', y='importance', 
                 labels={'importance': 'Importance', 'feature': 'Feature'},
                 title='Feature Importance in California Housing Dataset')
    fig.update_traces(marker_color='#1e3d59')
    st.plotly_chart(fig)
    
    st.write("""
    This example shows the feature importances for the California Housing dataset, which contains information about housing prices in California.
    
    💡 **Interpretation:**
    - The most important features for predicting housing prices in this dataset are median income and house age.
    - Geographic features (latitude and longitude) also play a significant role.
    - This aligns with common knowledge about real estate: location and local economic conditions are often key determinants of housing prices.
    """)
    
    st.subheader("Explore the Dataset")
    if st.checkbox("Show California Housing Dataset"):
        st.write(pd.DataFrame(X, columns=feature_names))

def quiz_section():
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main purpose of feature importance analysis?",
            "options": [
                "To improve model accuracy",
                "To understand which features are most influential in making predictions",
                "To reduce the number of features in a dataset",
                "To visualize the relationship between features"
            ],
            "correct": "To understand which features are most influential in making predictions",
            "explanation": "The main purpose of feature importance analysis is to determine which features in a dataset have the most impact on the model's predictions. This can help in understanding the data, interpreting the model, and potentially in feature selection."
        },
        {
            "question": "Which of the following is NOT a common method for calculating feature importance?",
            "options": [
                "Built-in importance in Random Forests",
                "Permutation Importance",
                "SHAP values",
                "K-means clustering"
            ],
            "correct": "K-means clustering",
            "explanation": "K-means clustering is not a method for calculating feature importance. It's an unsupervised learning algorithm used for clustering data points. The other options (Built-in importance in Random Forests, Permutation Importance, and SHAP values) are all common methods for assessing feature importance."
        },
        {
            "question": "If a feature has a very low importance score, what might this indicate?",
            "options": [
                "The feature is crucial for the model's predictions",
                "The feature might be redundant or not relevant for the prediction task",
                "The model is overfitting",
                "The dataset is too small"
            ],
            "correct": "The feature might be redundant or not relevant for the prediction task",
            "explanation": "A very low importance score typically suggests that the feature has little impact on the model's predictions. This could mean that the feature is not relevant for the task at hand, or that it's redundant with other features. However, be cautious: in some cases, a feature might be important but have complex interactions that aren't captured by simple importance measures."
        }
    ]
    
    for i, q in enumerate(questions, 1):
        st.subheader(f"Question {i}")
        with st.container():
            st.write(q["question"])
            answer = st.radio("Select your answer:", q["options"], key=f"q{i}")
            if st.button("Check Answer", key=f"check{i}"):
                if answer == q["correct"]:
                    st.success("Correct! 🎉")
                else:
                    st.error(f"Incorrect. The correct answer is: {q['correct']}")
                st.info(f"Explanation: {q['explanation']}")
            st.write("---")

if __name__ == "__main__":
    main()