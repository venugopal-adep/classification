import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from io import StringIO

def main():
    st.set_page_config(page_title="Handling Imbalanced Data using SMOTE", layout="wide")
    
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
    
    st.title("🔄 Handling Imbalanced Data using SMOTE")
    
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
    st.header("Understanding SMOTE and Imbalanced Data")
    
    st.write("""
    Imbalanced data is a common problem in machine learning, especially in classification tasks. It occurs when one class 
    (the majority class) significantly outweighs the other class (the minority class) in the dataset.

    SMOTE (Synthetic Minority Over-sampling Technique) is a popular method for handling imbalanced datasets. Here's how it works:
    """)

    st.subheader("SMOTE Algorithm:")
    st.write("""
    1. For each minority class sample:
        a. Find its k-nearest neighbors (usually k=5)
        b. Randomly select one of these neighbors
        c. Create a synthetic sample along the line connecting the original sample and the selected neighbor
    2. Repeat this process until the desired balance is achieved
    """)

    st.subheader("Advantages of SMOTE:")
    advantages = [
        "Increases the representation of the minority class",
        "Helps prevent overfitting to the majority class",
        "Improves model performance on minority class predictions",
        "Creates synthetic samples instead of duplicating existing ones"
    ]
    for adv in advantages:
        st.write(f"- {adv}")

    st.subheader("Considerations:")
    considerations = [
        "May not be suitable for all types of data (e.g., time series)",
        "Can potentially create noisy samples if not carefully tuned",
        "Should be applied only to the training set to prevent data leakage"
    ]
    for con in considerations:
        st.write(f"- {con}")

    st.write("""
    SMOTE is particularly useful in scenarios like fraud detection, rare disease diagnosis, and anomaly detection, 
    where the events of interest are relatively rare but critically important to identify.
    """)

@st.cache_data
def load_data():
    # URL of the dataset
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    
    # Download the data
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text))
    
    return data

def experiment_section():
    st.header("🧪 Experiment with SMOTE")

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

    # Train a model without SMOTE
    model_without_smote = LogisticRegression(random_state=42)
    model_without_smote.fit(X_train_scaled, y_train)
    y_pred_without_smote = model_without_smote.predict(X_test_scaled)

    st.subheader("Model Performance without SMOTE")
    st.text(classification_report(y_test, y_pred_without_smote))

    # Apply SMOTE
    st.subheader("Apply SMOTE")
    sampling_strategy = st.slider("Select sampling strategy (ratio of minority to majority class)", 0.1, 1.0, 0.5, 0.1)
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    st.write(f"Class distribution after SMOTE:\n{pd.Series(y_train_smote).value_counts(normalize=True)}")

    # Train a model with SMOTE
    model_with_smote = LogisticRegression(random_state=42)
    model_with_smote.fit(X_train_smote, y_train_smote)
    y_pred_with_smote = model_with_smote.predict(X_test_scaled)

    st.subheader("Model Performance with SMOTE")
    st.text(classification_report(y_test, y_pred_with_smote))

    st.write("""
    Experiment with different sampling strategies to see how it affects the model's performance.
    Notice how SMOTE improves the recall for the minority class (fraudulent transactions).
    """)

def visualization_section():
    st.header("📊 Visualizing SMOTE Effects")

    data = load_data()

    # Prepare the data
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Split the data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Select two features for visualization
    feature1 = st.selectbox("Select first feature for visualization", X.columns, index=0)
    feature2 = st.selectbox("Select second feature for visualization", X.columns, index=1)

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        feature1: X_train_scaled[:, X.columns.get_loc(feature1)],
        feature2: X_train_scaled[:, X.columns.get_loc(feature2)],
        'Class': y_train
    })

    # Plot original data
    st.subheader("Original Data Distribution")
    fig = px.scatter(plot_data, x=feature1, y=feature2, color='Class')
    st.plotly_chart(fig)

    # Apply SMOTE
    sampling_strategy = st.slider("Select sampling strategy for visualization", 0.1, 1.0, 0.5, 0.1)
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    # Create a DataFrame for plotting SMOTE data
    plot_data_smote = pd.DataFrame({
        feature1: X_train_smote[:, X.columns.get_loc(feature1)],
        feature2: X_train_smote[:, X.columns.get_loc(feature2)],
        'Class': y_train_smote
    })

    # Plot SMOTE data
    st.subheader("Data Distribution after SMOTE")
    fig_smote = px.scatter(plot_data_smote, x=feature1, y=feature2, color='Class')
    st.plotly_chart(fig_smote)

    st.write("""
    Observe how SMOTE creates synthetic samples for the minority class, balancing the dataset.
    The new points are created along the lines connecting existing minority class samples.
    """)

def quiz_section():
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main purpose of SMOTE?",
            "options": [
                "To remove outliers from the dataset",
                "To balance imbalanced datasets by oversampling the minority class",
                "To reduce the dimensionality of the dataset",
                "To normalize the features of the dataset"
            ],
            "correct": "To balance imbalanced datasets by oversampling the minority class",
            "explanation": "SMOTE (Synthetic Minority Over-sampling Technique) is primarily used to address class imbalance by creating synthetic examples of the minority class, thus balancing the dataset."
        },
        {
            "question": "How does SMOTE create new samples?",
            "options": [
                "By simply duplicating existing minority samples",
                "By randomly generating new data points",
                "By interpolating between existing minority samples",
                "By removing majority samples"
            ],
            "correct": "By interpolating between existing minority samples",
            "explanation": "SMOTE creates new synthetic samples by interpolating between existing minority class samples and their nearest neighbors."
        },
        {
            "question": "When applying SMOTE, on which part of the dataset should it be used?",
            "options": [
                "The entire dataset",
                "Only the training set",
                "Only the test set",
                "Both training and test sets"
            ],
            "correct": "Only the training set",
            "explanation": "SMOTE should be applied only to the training set to prevent data leakage. The test set should remain untouched to provide an unbiased evaluation of the model's performance."
        },
        {
            "question": "What is a potential drawback of using SMOTE?",
            "options": [
                "It always leads to overfitting",
                "It can potentially create noisy or unrealistic samples",
                "It reduces the overall size of the dataset",
                "It only works with numerical features"
            ],
            "correct": "It can potentially create noisy or unrealistic samples",
            "explanation": "While SMOTE is generally effective, it can sometimes create synthetic samples that are noisy or unrealistic, especially if the feature space is complex or if there are outliers in the minority class."
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