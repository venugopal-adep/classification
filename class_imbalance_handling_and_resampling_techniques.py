import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import plotly.graph_objects as go
import requests
from io import StringIO

@st.cache_data
def load_data():
    # URL of the dataset
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    
    # Download the data
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text))
    
    return data

def main():
    st.set_page_config(page_title="Class Imbalance Handling Demo", layout="wide")
    
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
    
    st.title("⚖️ Class Imbalance Handling with Resampling Techniques")
    
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
    st.header("Understanding Class Imbalance and Resampling Techniques")
    
    st.write("""
    Class imbalance is a common problem in machine learning where one class (the majority class) significantly outweighs the other class (the minority class) in the dataset.
    This can lead to biased models that perform poorly on the minority class.

    Resampling techniques are used to address class imbalance by adjusting the class distribution of a dataset. The main approaches are:
    """)

    techniques = {
        "Oversampling": "Increasing the number of instances in the minority class",
        "Undersampling": "Reducing the number of instances in the majority class",
        "Combination Methods": "Using both oversampling and undersampling"
    }

    for technique, description in techniques.items():
        st.subheader(f"{technique}")
        st.write(description)

    st.subheader("Common Resampling Techniques:")
    methods = {
        "Random Oversampling": "Randomly duplicate examples in the minority class",
        "SMOTE (Synthetic Minority Over-sampling Technique)": "Create synthetic examples in the minority class",
        "Random Undersampling": "Randomly remove examples in the majority class",
        "SMOTEENN": "Combine SMOTE with Edited Nearest Neighbors",
        "SMOTETomek": "Combine SMOTE with Tomek links"
    }

    for method, description in methods.items():
        st.write(f"**{method}**: {description}")

    st.write("""
    Handling class imbalance is crucial because:
    - It helps prevent bias towards the majority class
    - It improves the model's ability to predict minority class instances
    - It can lead to better overall model performance, especially for metrics like F1-score
    """)

def experiment_section():
    st.header("🧪 Experiment with Resampling Techniques")

    data = load_data()

    st.subheader("Original Data Overview")
    st.write(data.head())
    st.write(f"Dataset shape: {data.shape}")
    st.write(f"Class distribution:\n{data['Class'].value_counts(normalize=True)}")

    # Prepare the data
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Resampling Options
    st.subheader("Select Resampling Technique")
    resampling_method = st.selectbox("Choose a resampling method", 
                                     ["None", "Random Oversampling", "SMOTE", 
                                      "Random Undersampling", "SMOTEENN", "SMOTETomek"])

    if resampling_method != "None":
        if resampling_method == "Random Oversampling":
            resampler = RandomOverSampler(random_state=42)
        elif resampling_method == "SMOTE":
            resampler = SMOTE(random_state=42)
        elif resampling_method == "Random Undersampling":
            resampler = RandomUnderSampler(random_state=42)
        elif resampling_method == "SMOTEENN":
            resampler = SMOTEENN(random_state=42)
        elif resampling_method == "SMOTETomek":
            resampler = SMOTETomek(random_state=42)

        X_resampled, y_resampled = resampler.fit_resample(X_train_scaled, y_train)

        st.write(f"Shape after resampling: {X_resampled.shape}")
        st.write(f"Class distribution after resampling:\n{pd.Series(y_resampled).value_counts(normalize=True)}")
    else:
        X_resampled, y_resampled = X_train_scaled, y_train

    # Train the model
    model = LogisticRegression(random_state=42)
    model.fit(X_resampled, y_resampled)

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
    Experiment with different resampling techniques to see how they affect the model's performance.
    Pay attention to the changes in precision, recall, and F1-score, especially for the minority class (1).
    """)

def visualization_section():
    st.header("📊 Visualizing Resampling Effects")

    data = load_data()

    # Prepare the data
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Resampling methods
    methods = {
        "Original": None,
        "Random Oversampling": RandomOverSampler(random_state=42),
        "SMOTE": SMOTE(random_state=42),
        "Random Undersampling": RandomUnderSampler(random_state=42),
        "SMOTEENN": SMOTEENN(random_state=42),
        "SMOTETomek": SMOTETomek(random_state=42)
    }

    # Calculate ROC curves for each method
    fpr = {}
    tpr = {}
    roc_auc = {}

    for name, resampler in methods.items():
        if resampler is not None:
            X_resampled, y_resampled = resampler.fit_resample(X_train_scaled, y_train)
        else:
            X_resampled, y_resampled = X_train_scaled, y_train

        model = LogisticRegression(random_state=42)
        model.fit(X_resampled, y_resampled)

        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr[name], tpr[name], _ = roc_curve(y_test, y_pred_proba)
        roc_auc[name] = auc(fpr[name], tpr[name])

    # Plot ROC curves
    fig = go.Figure()
    for name in methods:
        fig.add_trace(go.Scatter(x=fpr[name], y=tpr[name],
                                 mode='lines',
                                 name=f'{name} (AUC = {roc_auc[name]:.2f})'))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines',
                             name='Random Classifier',
                             line=dict(dash='dash')))

    fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      legend_title='Resampling Method',
                      width=800,
                      height=600)

    st.plotly_chart(fig)

    st.write("""
    This plot shows the ROC curves for different resampling methods.
    The closer the curve follows the top-left corner, the better the model's performance.
    The AUC (Area Under the Curve) provides a single score to compare methods, with higher values indicating better performance.
    """)

def quiz_section():
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main goal of resampling techniques in handling class imbalance?",
            "options": [
                "To increase the overall size of the dataset",
                "To balance the class distribution in the dataset",
                "To remove outliers from the dataset",
                "To normalize the features"
            ],
            "correct": "To balance the class distribution in the dataset",
            "explanation": "Resampling techniques aim to balance the class distribution in imbalanced datasets, which helps in preventing bias towards the majority class and improves the model's ability to predict minority class instances."
        },
        {
            "question": "Which of the following is NOT a common resampling technique?",
            "options": [
                "Random Oversampling",
                "SMOTE",
                "Random Undersampling",
                "Feature Scaling"
            ],
            "correct": "Feature Scaling",
            "explanation": "Feature Scaling is not a resampling technique. It's a preprocessing step used to normalize the range of features in a dataset. Random Oversampling, SMOTE, and Random Undersampling are all common resampling techniques used to handle class imbalance."
        },
        {
            "question": "What is a potential drawback of random undersampling?",
            "options": [
                "It can lead to overfitting",
                "It increases the size of the dataset",
                "It may discard potentially useful information",
                "It always decreases model performance"
            ],
            "correct": "It may discard potentially useful information",
            "explanation": "Random undersampling reduces the number of instances in the majority class, which can potentially discard useful information. This is one of its main drawbacks, although it can still be effective in many situations."
        },
        {
            "question": "What does SMOTE stand for?",
            "options": [
                "Simple Method of Oversampling Technique",
                "Synthetic Minority Over-sampling Technique",
                "Standardized Method for Optimizing Training Examples",
                "Statistical Model for Outlier Treatment and Extraction"
            ],
            "correct": "Synthetic Minority Over-sampling Technique",
            "explanation": "SMOTE stands for Synthetic Minority Over-sampling Technique. It works by creating synthetic examples in the minority class, rather than simply duplicating existing examples."
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