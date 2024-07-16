import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from catboost import CatBoostClassifier, Pool
from sklearn.datasets import load_breast_cancer, load_iris

# Set page config
st.set_page_config(layout="wide", page_title="CatBoost Classification Explorer", page_icon="🐱")

# Custom CSS (unchanged, omitted for brevity)
st.markdown("""
<style>
    # ... (keep the existing CSS)
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🐱 CatBoost Classification Explorer 🐱</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("<p class='big-font'>Welcome to the CatBoost Classification Explorer!</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>Let's explore the power of CatBoost for classification tasks.</p>", unsafe_allow_html=True)

# Explanation
st.markdown("<p class='medium-font'>What is CatBoost?</p>", unsafe_allow_html=True)
st.markdown("""
<p class='small-font'>
CatBoost is a high-performance open-source library for gradient boosting on decision trees. Key points:

- Developed by Yandex researchers and engineers
- Supports both numerical and categorical features out of the box
- Implements symmetric trees, which results in a faster inference compared to classic gradient boosting schemes
- Uses ordered boosting to reduce overfitting
- Implements a novel gradient boosting scheme with a faster prediction time

CatBoost is particularly effective for datasets with categorical features and achieves state-of-the-art results on many machine learning tasks.
</p>
""", unsafe_allow_html=True)

# Tabs with custom styling
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
    st.markdown("<p class='medium-font'>CatBoost Model Training</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's train a CatBoost model on a selected dataset and evaluate its performance.
        </p>
        """, unsafe_allow_html=True)

        dataset = st.selectbox("Select dataset", ["Breast Cancer", "Iris"])
        X, y = load_data(dataset)
        
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", 0, 100, 42)
        
        n_iterations = st.slider("Number of iterations", 10, 1000, 100, 10)
        learning_rate = st.number_input("Learning rate", 0.01, 1.0, 0.1, 0.01)
        
        if st.button("Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            model = CatBoostClassifier(
                iterations=n_iterations,
                learning_rate=learning_rate,
                random_state=random_state,
                verbose=False
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
        
        n_iterations = st.slider("Number of iterations", 10, 1000, 100, 10, key="hp_iterations")
        learning_rate = st.number_input("Learning rate", 0.01, 1.0, 0.1, 0.01, key="hp_lr")
        depth = st.slider("Tree depth", 1, 10, 6, 1)
        l2_leaf_reg = st.number_input("L2 regularization coefficient", 0.1, 10.0, 3.0, 0.1)
        
        if st.button("Train and Evaluate"):
            results = []
            for iter in range(50, n_iterations+1, 50):
                model = CatBoostClassifier(
                    iterations=iter,
                    learning_rate=learning_rate,
                    depth=depth,
                    l2_leaf_reg=l2_leaf_reg,
                    random_state=42,
                    verbose=False
                )
                
                model.fit(X_train, y_train)
                train_accuracy = accuracy_score(y_train, model.predict(X_train))
                test_accuracy = accuracy_score(y_test, model.predict(X_test))
                results.append((iter, train_accuracy, test_accuracy))
            
            results_df = pd.DataFrame(results, columns=['Iterations', 'Train Accuracy', 'Test Accuracy'])
            st.dataframe(results_df)

    with col2:
        if 'results_df' in locals():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results_df['Iterations'], y=results_df['Train Accuracy'], mode='lines+markers', name='Train Accuracy'))
            fig.add_trace(go.Scatter(x=results_df['Iterations'], y=results_df['Test Accuracy'], mode='lines+markers', name='Test Accuracy'))
            fig.update_layout(
                title='Model Performance vs Number of Iterations',
                xaxis_title='Number of Iterations',
                yaxis_title='Accuracy'
            )
            st.plotly_chart(fig)

with tab3:
    st.markdown("<p class='medium-font'>Feature Importance</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's examine which features are most important for our CatBoost model.
        </p>
        """, unsafe_allow_html=True)

        dataset = st.selectbox("Select dataset", ["Breast Cancer", "Iris"], key="fi_dataset")
        X, y = load_data(dataset)
        
        n_iterations = st.slider("Number of iterations", 10, 1000, 100, 10, key="fi_iterations")
        learning_rate = st.number_input("Learning rate", 0.01, 1.0, 0.1, 0.01, key="fi_lr")
        
        if st.button("Calculate Feature Importance"):
            model = CatBoostClassifier(
                iterations=n_iterations,
                learning_rate=learning_rate,
                random_state=42,
                verbose=False
            )
            
            model.fit(X, y)
            
            feature_importance = model.get_feature_importance()
            feature_names = X.columns
            
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
            importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
            
            st.dataframe(importance_df)

    with col2:
        if 'importance_df' in locals():
            fig = go.Figure(go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h'
            ))
            fig.update_layout(
                title='Feature Importance',
                xaxis_title='Importance',
                yaxis_title='Feature',
                height=800
            )
            st.plotly_chart(fig)

with tab4:
    st.markdown("<p class='medium-font'>Test Your Knowledge!</p>", unsafe_allow_html=True)

    questions = [
        {
            "question": "What is a key advantage of CatBoost over traditional gradient boosting methods?",
            "options": [
                "It only works with numerical features",
                "It has built-in support for categorical features",
                "It's slower but more accurate",
                "It requires more memory"
            ],
            "correct": 1,
            "explanation": "CatBoost has built-in support for categorical features, which eliminates the need for pre-processing steps like one-hot encoding."
        },
        {
            "question": "What is ordered boosting in CatBoost?",
            "options": [
                "A technique to sort features by importance",
                "A method to reduce overfitting",
                "An algorithm to optimize hyperparameters",
                "A way to handle missing values"
            ],
            "correct": 1,
            "explanation": "Ordered boosting is a technique used in CatBoost to reduce overfitting by considering the time-based nature of the data."
        },
        {
            "question": "How does increasing the number of iterations typically affect a CatBoost model?",
            "options": [
                "It always improves both training and test accuracy",
                "It always reduces both training and test accuracy",
                "It typically improves training accuracy but may lead to overfitting",
                "It has no effect on model performance"
            ],
            "correct": 2,
            "explanation": "Increasing the number of iterations typically improves training accuracy but may lead to overfitting if the number becomes too large."
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
st.markdown("<p class='small-font'>You've explored CatBoost classification through interactive examples and visualizations. CatBoost is a powerful tool for machine learning tasks, especially when dealing with categorical features. Keep exploring and applying these concepts to solve real-world problems!</p>", unsafe_allow_html=True)