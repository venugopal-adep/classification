import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(layout="wide", page_title="AdaBoost Classification Explorer", page_icon="🎭")

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
st.markdown("<h1 style='text-align: center;'>🎭 AdaBoost Classification Explorer 🎭</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("<p class='big-font'>Welcome to the AdaBoost Classification Explorer!</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>Let's explore the power of AdaBoost for classification tasks.</p>", unsafe_allow_html=True)

# Explanation
st.markdown("<p class='medium-font'>What is AdaBoost?</p>", unsafe_allow_html=True)
st.markdown("""
<p class='small-font'>
AdaBoost (Adaptive Boosting) is a boosting ensemble method. Key points:

- Combines multiple weak learners to create a strong classifier
- Iteratively adjusts the weight of misclassified samples
- Often uses decision trees (typically stumps) as base learners
- Effective for binary classification problems
- Less prone to overfitting compared to some other algorithms
- Can be sensitive to noisy data and outliers

AdaBoost was one of the first boosting algorithms to be adapted to practical applications.
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
    st.markdown("<p class='medium-font'>AdaBoost Model Training</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's train an AdaBoost model on a selected dataset and evaluate its performance.
        </p>
        """, unsafe_allow_html=True)

        dataset = st.selectbox("Select dataset", ["Breast Cancer", "Iris"])
        X, y = load_data(dataset)
        
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", 0, 100, 42)
        
        n_estimators = st.slider("Number of estimators", 10, 500, 50, 10)
        learning_rate = st.number_input("Learning rate", 0.01, 2.0, 1.0, 0.01)
        
        if st.button("Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            model = AdaBoostClassifier(
                n_estimators=n_estimators,
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
        
        param_to_tune = st.selectbox("Parameter to tune", ["n_estimators", "learning_rate"])
        
        if st.button("Tune Hyperparameter"):
            results = []
            if param_to_tune == "n_estimators":
                for n_est in range(10, 210, 20):
                    model = AdaBoostClassifier(n_estimators=n_est, random_state=42)
                    model.fit(X_train, y_train)
                    test_accuracy = accuracy_score(y_test, model.predict(X_test))
                    results.append((n_est, test_accuracy))
            else:  # learning_rate
                for lr in [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]:
                    model = AdaBoostClassifier(learning_rate=lr, random_state=42)
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
        Let's examine which features are most important for our AdaBoost model.
        </p>
        """, unsafe_allow_html=True)

        dataset = st.selectbox("Select dataset", ["Breast Cancer", "Iris"], key="fi_dataset")
        X, y = load_data(dataset)
        
        n_estimators = st.slider("Number of estimators", 10, 500, 50, 10, key="fi_n_estimators")
        learning_rate = st.number_input("Learning rate", 0.01, 2.0, 1.0, 0.01, key="fi_learning_rate")
        
        if st.button("Calculate Feature Importance"):
            model = AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
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
            importance_df.plot(x='feature', y='importance', kind='bar', ax=ax)
            plt.title('Feature Importance')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(rotation=90)
            st.pyplot(fig)

with tab4:
    st.markdown("<p class='medium-font'>Test Your Knowledge!</p>", unsafe_allow_html=True)

    questions = [
        {
            "question": "What does AdaBoost stand for?",
            "options": [
                "Advanced Boosting",
                "Adaptive Boosting",
                "Adjusted Boosting",
                "Automated Boosting"
            ],
            "correct": 1,
            "explanation": "AdaBoost stands for Adaptive Boosting. It adapts to the errors of weak learners to improve classification."
        },
        {
            "question": "Which of the following is a key characteristic of AdaBoost?",
            "options": [
                "It only works with neural networks as base learners",
                "It assigns higher weights to misclassified samples in subsequent iterations",
                "It requires a large number of features to work effectively",
                "It's primarily used for regression tasks"
            ],
            "correct": 1,
            "explanation": "AdaBoost assigns higher weights to misclassified samples in each iteration, focusing more on difficult examples."
        },
        {
            "question": "How does increasing the learning rate typically affect an AdaBoost model?",
            "options": [
                "It always improves model performance",
                "It may lead to overfitting if set too high",
                "It always reduces model performance",
                "It has no effect on the model's performance"
            ],
            "correct": 1,
            "explanation": "Increasing the learning rate in AdaBoost can lead to overfitting if set too high, as it makes the model more sensitive to errors in individual weak learners."
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
You've explored AdaBoost classification through interactive examples and visualizations. 
AdaBoost is a powerful ensemble method for various machine learning tasks. 
Remember these key points:

1. AdaBoost combines weak learners to create a strong classifier.
2. It adapts by focusing more on misclassified samples in subsequent iterations.
3. The number of estimators and learning rate are crucial hyperparameters.
4. AdaBoost can provide feature importance scores.
5. It's effective for binary classification but can be extended to multi-class problems.

Keep exploring and applying these concepts to solve real-world problems!
</p>
""", unsafe_allow_html=True)