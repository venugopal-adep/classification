import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import plotly.graph_objects as go

@st.cache_data
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y

def main():
    st.set_page_config(page_title="Bagging and Boosting Techniques Demo", layout="wide")
    
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
    
    st.title("🌳 Bagging and Boosting Techniques")
    
    tabs = st.tabs(["📚 Learn", "🧪 Experiment", "📊 Visualization", "🧠 Quiz"])
    
    with tabs[0]:
        learn_section()
    
    with tabs[1]:
        experiment_section()
    
    with tabs[2]:
        visualization_section()
    
    with tabs[3]:
        quiz_section()

def learn_section():
    st.header("Understanding Bagging and Boosting")
    
    st.write("""
    Bagging (Bootstrap Aggregating) and Boosting are ensemble learning techniques used to improve the performance and robustness of machine learning models.

    Key concepts:
    """)

    concepts = {
        "Ensemble Learning": "Combining multiple models to improve overall performance",
        "Bagging": "Training multiple models on random subsets of the data and averaging their predictions",
        "Boosting": "Training multiple models sequentially, with each model trying to correct the errors of the previous ones"
    }

    for concept, description in concepts.items():
        st.subheader(f"{concept}")
        st.write(description)

    st.subheader("Common Bagging and Boosting Techniques:")
    methods = {
        "Random Forest": "A bagging technique that uses decision trees as base learners",
        "AdaBoost": "A boosting algorithm that adjusts the weight of instances based on previous errors",
        "Gradient Boosting": "A boosting technique that builds trees to minimize the loss function gradient"
    }

    for method, description in methods.items():
        st.write(f"**{method}**: {description}")

    st.write("""
    Bagging and Boosting are important because:
    - They often improve model accuracy and reduce overfitting
    - They can handle complex relationships in data
    - They provide robustness against noisy data
    - They can give insights into feature importance
    """)

def experiment_section():
    st.header("🧪 Experiment with Bagging and Boosting")

    X, y = load_data()

    st.subheader("Dataset Overview")
    st.write(f"Dataset shape: {X.shape}")
    st.write(f"Features: {', '.join(X.columns)}")
    st.write(f"Target distribution:\n{y.value_counts(normalize=True)}")

    # Prepare the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model selection
    st.subheader("Select Model")
    model_choice = st.selectbox("Choose a model", 
                                ["Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting"])

    # Hyperparameter tuning
    st.subheader("Hyperparameter Tuning")
    if model_choice == "Decision Tree":
        max_depth = st.slider("Max Depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif model_choice == "Random Forest":
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
        max_depth = st.slider("Max Depth", 1, 20, 5)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_choice == "AdaBoost":
        n_estimators = st.slider("Number of Estimators", 10, 200, 50)
        learning_rate = st.slider("Learning Rate", 0.01, 2.0, 1.0)
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    else:  # Gradient Boosting
        n_estimators = st.slider("Number of Estimators", 10, 200, 100)
        learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
        max_depth = st.slider("Max Depth", 1, 10, 3)
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, 
                                           max_depth=max_depth, random_state=42)

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Display results
    st.subheader("Model Performance")
    st.text(classification_report(y_test, y_pred))

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot(plt)

    st.write("""
    Experiment with different models and hyperparameters to see how they affect performance.
    Pay attention to changes in accuracy, precision, recall, and F1-score.
    """)

def visualization_section():
    st.header("📊 Visualizing Model Comparisons")

    X, y = load_data()

    # Prepare the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models to compare
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

    # Plot results
    fig = go.Figure(data=[go.Bar(x=list(results.keys()), y=list(results.values()))])
    fig.update_layout(title='Model Accuracy Comparison',
                      xaxis_title='Model',
                      yaxis_title='Accuracy',
                      yaxis_range=[0.9, 1])  # Adjust this range as needed
    st.plotly_chart(fig)

    # Feature importance
    st.subheader("Feature Importance")
    if st.checkbox("Show Feature Importance"):
        model_for_importance = RandomForestClassifier(random_state=42)
        model_for_importance.fit(X_train_scaled, y_train)
        importances = model_for_importance.feature_importances_
        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
        feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

        fig = go.Figure(data=[go.Bar(x=feature_importance['importance'], y=feature_importance['feature'], orientation='h')])
        fig.update_layout(title='Top 10 Feature Importances',
                          xaxis_title='Importance',
                          yaxis_title='Feature')
        st.plotly_chart(fig)

    st.write("""
    This visualization compares the accuracy of different models on the test set.
    The feature importance plot shows which features have the most impact on the model's decisions.
    """)

def quiz_section():
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main difference between bagging and boosting?",
            "options": [
                "Bagging uses only decision trees, while boosting can use any model",
                "Bagging trains models in parallel, while boosting trains them sequentially",
                "Bagging is only for classification, while boosting is only for regression",
                "Bagging reduces variance, while boosting reduces bias"
            ],
            "correct": "Bagging trains models in parallel, while boosting trains them sequentially",
            "explanation": "Bagging trains multiple models independently and in parallel, often on different subsets of the data. Boosting, on the other hand, trains models sequentially, with each model trying to correct the errors of the previous ones."
        },
        {
            "question": "Which of the following is an example of a bagging technique?",
            "options": [
                "AdaBoost",
                "Gradient Boosting",
                "Random Forest",
                "XGBoost"
            ],
            "correct": "Random Forest",
            "explanation": "Random Forest is a bagging technique that creates multiple decision trees using random subsets of the data and features. AdaBoost, Gradient Boosting, and XGBoost are all boosting techniques."
        },
        {
            "question": "What is a potential advantage of boosting over bagging?",
            "options": [
                "Boosting is always faster to train",
                "Boosting can achieve higher accuracy by focusing on difficult examples",
                "Boosting is less prone to overfitting",
                "Boosting requires fewer hyperparameters to tune"
            ],
            "correct": "Boosting can achieve higher accuracy by focusing on difficult examples",
            "explanation": "Boosting techniques often achieve higher accuracy than bagging because they focus on the examples that are difficult to classify, adjusting the model iteratively to correct previous errors. However, this can also make boosting more prone to overfitting if not properly regularized."
        },
        {
            "question": "In the context of Random Forests, what does 'feature importance' typically measure?",
            "options": [
                "The number of times a feature appears in the dataset",
                "The correlation between a feature and the target variable",
                "The average depth of the feature in the decision trees",
                "The average decrease in impurity across all trees when the feature is used for splitting"
            ],
            "correct": "The average decrease in impurity across all trees when the feature is used for splitting",
            "explanation": "In Random Forests, feature importance typically measures how much each feature contributes to decreasing impurity (like Gini impurity or entropy) when it's used for splitting across all trees in the forest. Features that lead to larger decreases in impurity are considered more important."
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