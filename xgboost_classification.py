import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.datasets import load_breast_cancer, load_iris
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(layout="wide", page_title="XGBoost Classification Explorer", page_icon="🌳")

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size: 20px !important;
        font-weight: bold;
    }
    .small-font {
        font-size: 16px !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🌳 XGBoost Classification Explorer 🌳</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("<p class='big-font'>Welcome to the XGBoost Classification Explorer!</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>Let's explore the power of XGBoost for classification tasks.</p>", unsafe_allow_html=True)

# Explanation
st.markdown("<p class='medium-font'>What is XGBoost?</p>", unsafe_allow_html=True)
st.markdown("""
<p class='small-font'>
XGBoost (eXtreme Gradient Boosting) is an optimized distributed gradient boosting library. Key points:

- Efficient implementation of gradient boosting decision trees
- Designed for speed and performance
- Built-in regularization to prevent overfitting
- Handles missing values automatically
- Allows for parallel and distributed computing
- Widely used in machine learning competitions and industry

XGBoost is known for its high performance and is often a go-to algorithm for many data scientists.
</p>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Model Training", "🎛️ Hyperparameter Tuning", "📊 Feature Importance", "🧠 Quiz"])

# Load data
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Breast Cancer":
        data = load_breast_cancer()
    elif dataset_name == "Iris":
        data = load_iris()
    else:
        raise ValueError("Unknown dataset")
    
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y

with tab1:
    st.markdown("<p class='medium-font'>XGBoost Model Training</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's train an XGBoost model on a selected dataset and evaluate its performance.
        </p>
        """, unsafe_allow_html=True)

        dataset = st.selectbox("Select dataset", ["Breast Cancer", "Iris"])
        X, y = load_data(dataset)
        
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", 0, 100, 42)
        
        n_estimators = st.slider("Number of estimators", 10, 500, 100, 10)
        max_depth = st.slider("Max depth", 1, 10, 3, 1)
        learning_rate = st.number_input("Learning rate", 0.01, 1.0, 0.1, 0.01)
        
        if st.button("Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state
            )
            
            model.fit(X_train, y_train)
            
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, train_preds)
            test_accuracy = accuracy_score(y_test, test_preds)
            
            st.markdown(f"""
            <p class='small-font'>
            Train Accuracy: {train_accuracy:.4f}<br>
            Test Accuracy: {test_accuracy:.4f}
            </p>
            """, unsafe_allow_html=True)

    with col2:
        if 'model' in locals():
            # Confusion Matrix
            cm = confusion_matrix(y_test, test_preds)
            fig = ff.create_annotated_heatmap(cm, x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'])
            fig.update_layout(title='Confusion Matrix')
            st.plotly_chart(fig)
            
            # ROC Curve
            if len(np.unique(y)) == 2:  # Binary classification
                y_score = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.2f})'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier'))
                fig.update_layout(
                    title='Receiver Operating Characteristic (ROC) Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate'
                )
                st.plotly_chart(fig)

with tab2:
    st.markdown("<p class='medium-font'>Hyperparameter Tuning</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's explore how different hyperparameters affect the model's performance.
        </p>
        """, unsafe_allow_html=True)

        dataset = st.selectbox("Select dataset", ["Breast Cancer", "Iris"], key="hp_dataset")
        X, y = load_data(dataset)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        param_to_tune = st.selectbox("Parameter to tune", ["n_estimators", "max_depth", "learning_rate"])
        
        if st.button("Tune Hyperparameter"):
            results = []
            if param_to_tune == "n_estimators":
                for n_est in range(10, 210, 20):
                    model = xgb.XGBClassifier(n_estimators=n_est, random_state=42)
                    model.fit(X_train, y_train)
                    test_accuracy = accuracy_score(y_test, model.predict(X_test))
                    results.append((n_est, test_accuracy))
            elif param_to_tune == "max_depth":
                for depth in range(1, 11):
                    model = xgb.XGBClassifier(max_depth=depth, random_state=42)
                    model.fit(X_train, y_train)
                    test_accuracy = accuracy_score(y_test, model.predict(X_test))
                    results.append((depth, test_accuracy))
            else:  # learning_rate
                for lr in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
                    model = xgb.XGBClassifier(learning_rate=lr, random_state=42)
                    model.fit(X_train, y_train)
                    test_accuracy = accuracy_score(y_test, model.predict(X_test))
                    results.append((lr, test_accuracy))
            
            results_df = pd.DataFrame(results, columns=[param_to_tune, 'Test Accuracy'])
            st.dataframe(results_df)

    with col2:
        if 'results_df' in locals():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results_df[param_to_tune], y=results_df['Test Accuracy'], mode='lines+markers'))
            fig.update_layout(
                title=f'Model Performance vs {param_to_tune}',
                xaxis_title=param_to_tune,
                yaxis_title='Test Accuracy'
            )
            st.plotly_chart(fig)

with tab3:
    st.markdown("<p class='medium-font'>Feature Importance</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's examine which features are most important for our XGBoost model.
        </p>
        """, unsafe_allow_html=True)

        dataset = st.selectbox("Select dataset", ["Breast Cancer", "Iris"], key="fi_dataset")
        X, y = load_data(dataset)
        
        n_estimators = st.slider("Number of estimators", 10, 500, 100, 10, key="fi_n_estimators")
        max_depth = st.slider("Max depth", 1, 10, 3, 1, key="fi_max_depth")
        
        if st.button("Calculate Feature Importance"):
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            
            model.fit(X, y)
            
            feature_importance = model.feature_importances_
            feature_names = X.columns
            
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
            importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
            
            st.dataframe(importance_df)

    with col2:
        if 'importance_df' in locals():
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_importance(model, ax=ax, height=0.5)
            plt.title('Feature Importance')
            st.pyplot(fig)

with tab4:
    st.markdown("<p class='medium-font'>Test Your Knowledge!</p>", unsafe_allow_html=True)

    questions = [
        {
            "question": "What does XGBoost stand for?",
            "options": [
                "eXtra Gradient Boosting",
                "eXtreme Gradient Boosting",
                "eXtended Gradient Boosting",
                "eXceptional Gradient Boosting"
            ],
            "correct": 1,
            "explanation": "XGBoost stands for eXtreme Gradient Boosting. It's an optimized distributed gradient boosting library."
        },
        {
            "question": "Which of the following is NOT a key feature of XGBoost?",
            "options": [
                "Regularization to prevent overfitting",
                "Handling of missing values",
                "Parallel processing",
                "Automatic feature selection"
            ],
            "correct": 3,
            "explanation": "While XGBoost has many advanced features, it does not perform automatic feature selection. It uses all provided features, but assigns importance to them."
        },
        {
            "question": "What is the typical effect of increasing the 'max_depth' parameter in XGBoost?",
            "options": [
                "It always improves model performance",
                "It may lead to overfitting if set too high",
                "It reduces the model's complexity",
                "It has no effect on the model's performance"
            ],
            "correct": 1,
            "explanation": "Increasing 'max_depth' allows the model to capture more complex patterns, but setting it too high may lead to overfitting as the model becomes too specific to the training data."
        }
    ]

    score = 0
    for i, q in enumerate(questions):
        st.markdown(f"<p class='small-font'><strong>Question {i+1}:</strong> {q['question']}</p>", unsafe_allow_html=True)
        user_answer = st.radio("Select your answer:", q['options'], key=f"q{i}")
        
        if st.button("Check Answer", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! 🎉")
                score += 1
            else:
                st.error("Incorrect. Try again! 🤔")
            st.info(q['explanation'])
        st.markdown("---")

    if st.button("Show Final Score"):
        st.markdown(f"<p class='big-font'>Your score: {score}/{len(questions)}</p>", unsafe_allow_html=True)
        if score == len(questions):
            st.balloons()

# Conclusion
st.markdown("<p class='big-font'>Congratulations! 🎊</p>", unsafe_allow_html=True)
st.markdown("""
<p class='small-font'>
You've explored XGBoost classification through interactive examples and visualizations. 
XGBoost is a powerful tool for various machine learning tasks, especially when dealing with structured/tabular data. 
Remember these key points:

1. XGBoost is an optimized implementation of gradient boosting.
2. It has built-in regularization to prevent overfitting.
3. Hyperparameter tuning is crucial for optimal performance.
4. XGBoost can handle missing values and provides feature importance.
5. It's widely used in competitions and industry due to its high performance.

Keep exploring and applying these concepts to solve real-world problems!
</p>
""", unsafe_allow_html=True)