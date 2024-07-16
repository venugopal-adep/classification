import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache_data
def load_data():
    iris = load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])
    return data

def main():
    st.set_page_config(page_title="Cross-Validation Analysis Demo", layout="wide")
    
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
        color: white !important;
    }
    .stButton>button:hover {
        background-color: #ff9e80;
        color: white !important;
    }
    .info-box {
        background-color: #e6f3ff;
        border-left: 5px solid #3366cc;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff6e40;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔀 Cross-Validation Analysis")
    
    tabs = st.tabs(["📚 Introduction", "🔍 Data Exploration", "📊 K-Fold CV", "🔁 LOOCV", "🧠 Quiz"])
    
    with tabs[0]:
        introduction_section()
    
    with tabs[1]:
        data_exploration_section()
    
    with tabs[2]:
        kfold_cv_section()
    
    with tabs[3]:
        loocv_section()
    
    with tabs[4]:
        quiz_section()

def introduction_section():
    st.header("Introduction to Cross-Validation")
    
    st.markdown("""
    <div class="info-box">
    Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. 
    The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into. 
    As such, the procedure is often called k-fold cross-validation.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Key Concepts")
    concepts = {
        "K-Fold Cross-Validation": "The data set is divided into k subsets, and the holdout method is repeated k times",
        "Leave-One-Out Cross-Validation (LOOCV)": "K is set to the number of observations in the dataset",
        "Stratified K-Fold": "The folds are selected so that the mean response value is approximately equal in all the folds",
        "Repeated K-Fold": "The k-fold CV process is repeated multiple times",
        "Nested Cross-Validation": "An inner CV loop is used for model selection, and an outer CV for error estimation"
    }

    for concept, description in concepts.items():
        st.markdown(f"**{concept}**: {description}")

    st.subheader("Importance of Cross-Validation")
    st.markdown("""
    - Provides a more accurate measure of model prediction performance
    - Helps to detect overfitting
    - Useful for model selection
    - Helps in hyperparameter tuning
    - Provides a less biased or less optimistic estimate of the model skill than other methods, such as a simple train/test split
    """)

def data_exploration_section():
    st.header("Data Exploration")

    data = load_data()

    st.subheader("Iris Dataset Overview")
    st.write(data.head())
    st.write(f"Dataset shape: {data.shape}")

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Pairplot
    st.subheader("Pairplot of Iris Features")
    fig = sns.pairplot(data, hue='target')
    st.pyplot(fig)

    st.markdown("""
    The Iris dataset contains 150 samples with 4 features each:
    1. Sepal length
    2. Sepal width
    3. Petal length
    4. Petal width
    
    The target variable is the Iris species, which has 3 classes:
    - 0: Setosa
    - 1: Versicolor
    - 2: Virginica
    
    This dataset is often used for classification tasks and is a good example for demonstrating cross-validation techniques.
    """)

def kfold_cv_section():
    st.header("K-Fold Cross-Validation")

    data = load_data()
    X = data.drop('target', axis=1)
    y = data['target']

    st.markdown("""
    <div class="info-box">
    K-Fold Cross-Validation involves randomly dividing the set of observations into k groups, or folds, of approximately equal size. 
    The first fold is treated as a validation set, and the model is fit on the remaining k-1 folds.
    </div>
    """, unsafe_allow_html=True)

    # Model selection
    model_name = st.selectbox("Select a model", ["Logistic Regression", "SVM", "Decision Tree", "Random Forest"])
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "SVM":
        model = SVC()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    else:
        model = RandomForestClassifier()

    # Number of folds
    k = st.slider("Number of folds", 2, 10, 5)

    if st.button("Run K-Fold Cross-Validation"):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf)

        st.subheader("Cross-Validation Results")
        st.write(f"Mean Accuracy: {scores.mean():.4f}")
        st.write(f"Standard Deviation: {scores.std():.4f}")

        # Visualize results
        fig = go.Figure(data=[go.Bar(y=scores, x=[f"Fold {i+1}" for i in range(k)])])
        fig.update_layout(title=f"{k}-Fold Cross-Validation Scores", 
                          xaxis_title="Fold", 
                          yaxis_title="Accuracy")
        st.plotly_chart(fig)

        st.markdown("""
        The bar plot shows the accuracy for each fold. The variation in these scores gives us an idea of how stable 
        our model's performance is across different subsets of the data.
        """)

def loocv_section():
    st.header("Leave-One-Out Cross-Validation (LOOCV)")

    data = load_data()
    X = data.drop('target', axis=1)
    y = data['target']

    st.markdown("""
    <div class="info-box">
    Leave-One-Out Cross-Validation (LOOCV) is a special case of K-Fold Cross-Validation where K equals the number of observations in the dataset. 
    In each iteration, it uses a single observation from the original sample as the validation data, and the remaining observations as the training data.
    </div>
    """, unsafe_allow_html=True)

    # Model selection
    model_name = st.selectbox("Select a model", ["Logistic Regression", "SVM", "Decision Tree", "Random Forest"], key="loocv_model")
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "SVM":
        model = SVC()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    else:
        model = RandomForestClassifier()

    if st.button("Run Leave-One-Out Cross-Validation"):
        loo = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=loo)

        st.subheader("LOOCV Results")
        st.write(f"Mean Accuracy: {scores.mean():.4f}")
        st.write(f"Standard Deviation: {scores.std():.4f}")

        # Visualize results
        fig = go.Figure(data=[go.Histogram(x=scores)])
        fig.update_layout(title="Distribution of LOOCV Scores", 
                          xaxis_title="Accuracy", 
                          yaxis_title="Count")
        st.plotly_chart(fig)

        st.markdown("""
        The histogram shows the distribution of accuracy scores across all LOOCV iterations. 
        This gives us a detailed view of our model's performance across all possible train-test splits.
        """)

def quiz_section():
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main purpose of cross-validation?",
            "options": [
                "To increase the size of the dataset",
                "To evaluate model performance on unseen data",
                "To speed up model training",
                "To visualize the data"
            ],
            "correct": "To evaluate model performance on unseen data",
            "explanation": "Cross-validation is primarily used to assess how the results of a statistical analysis will generalize to an independent data set. It helps in estimating how accurately a predictive model will perform in practice."
        },
        {
            "question": "In k-fold cross-validation, what does 'k' represent?",
            "options": [
                "The number of features in the dataset",
                "The number of classes in the target variable",
                "The number of groups the data is split into",
                "The number of iterations the model is trained"
            ],
            "correct": "The number of groups the data is split into",
            "explanation": "In k-fold cross-validation, 'k' represents the number of groups (or folds) that the data is split into. The model is then trained on k-1 folds and validated on the remaining fold, repeating this process k times."
        },
        {
            "question": "What is Leave-One-Out Cross-Validation (LOOCV)?",
            "options": [
                "A type of cross-validation where one feature is left out",
                "A type of cross-validation where k equals the number of observations",
                "A method to remove outliers from the dataset",
                "A technique to select the best features"
            ],
            "correct": "A type of cross-validation where k equals the number of observations",
            "explanation": "LOOCV is a special case of k-fold cross-validation where k is equal to the number of observations. In each iteration, it uses a single observation as the validation data and the remaining observations as the training data."
        },
        {
            "question": "What is an advantage of k-fold cross-validation over a simple train-test split?",
            "options": [
                "It's faster to compute",
                "It uses less memory",
                "It provides a more robust estimate of model performance",
                "It always improves model accuracy"
            ],
            "correct": "It provides a more robust estimate of model performance",
            "explanation": "K-fold cross-validation provides a more robust estimate of model performance because it tests the model on multiple train-test splits, reducing the impact of how the data is divided."
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