import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px

def main():
    st.set_page_config(page_title="Confusion Matrix Visualization", layout="wide")
    
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
    
    st.title("🧩 Confusion Matrix Visualization")
    
    tabs = st.tabs(["📚 Learn", "🔍 Explore", "📊 Metrics", "🧠 Quiz"])
    
    with tabs[0]:
        learn_section()
    
    with tabs[1]:
        explore_section()
    
    with tabs[2]:
        metrics_section()
    
    with tabs[3]:
        quiz_section()

def learn_section():
    st.header("What is a Confusion Matrix?")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known.

        It allows visualization of the performance of an algorithm, typically a supervised learning one.

        The matrix compares the actual target values with those predicted by the machine learning model.
        """)
    
    with col2:
        st.image("https://miro.medium.com/max/712/1*Z54JgbS4DUwWSknhDCvNTQ.png", caption="Confusion Matrix Structure")
    
    st.subheader("Components of a Confusion Matrix")
    st.write("""
    For a binary classification problem, a confusion matrix has four components:

    1. **True Positive (TP)**: Correctly predicted positive values
    2. **True Negative (TN)**: Correctly predicted negative values
    3. **False Positive (FP)**: Incorrectly predicted positive values (Type I error)
    4. **False Negative (FN)**: Incorrectly predicted negative values (Type II error)
    """)

def explore_section():
    st.header("🔍 Interactive Confusion Matrix")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Adjust Values")
        tp = st.number_input("True Positives (TP)", min_value=0, max_value=1000, value=80)
        fn = st.number_input("False Negatives (FN)", min_value=0, max_value=1000, value=20)
        fp = st.number_input("False Positives (FP)", min_value=0, max_value=1000, value=30)
        tn = st.number_input("True Negatives (TN)", min_value=0, max_value=1000, value=870)
    
    with col2:
        confusion_matrix = pd.DataFrame({
            'Actual Positive': [tp, fn],
            'Actual Negative': [fp, tn]
        }, index=['Predicted Positive', 'Predicted Negative'])
        
        fig = px.imshow(confusion_matrix, text_auto=True, color_continuous_scale='Blues',
                        labels=dict(x="Actual", y="Predicted", color="Count"))
        fig.update_layout(width=500, height=500)
        st.plotly_chart(fig)
    
    st.write("""
    💡 **Interpretation:**
    - **True Positives (TP)**: Correctly identified positive cases
    - **False Negatives (FN)**: Positive cases incorrectly identified as negative
    - **False Positives (FP)**: Negative cases incorrectly identified as positive
    - **True Negatives (TN)**: Correctly identified negative cases
    
    Adjust the values to see how the confusion matrix changes!
    """)

def metrics_section():
    st.header("📊 Confusion Matrix Metrics")
    
    tp = st.session_state.get('tp', 80)
    fn = st.session_state.get('fn', 20)
    fp = st.session_state.get('fp', 30)
    tn = st.session_state.get('tn', 870)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    specificity = tn / (tn + fp)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.write("(TP + TN) / (TP + TN + FP + FN)")
        
        st.metric("Precision", f"{precision:.4f}")
        st.write("TP / (TP + FP)")
        
        st.metric("Recall (Sensitivity)", f"{recall:.4f}")
        st.write("TP / (TP + FN)")
    
    with col2:
        st.metric("F1 Score", f"{f1_score:.4f}")
        st.write("2 * (Precision * Recall) / (Precision + Recall)")
        
        st.metric("Specificity", f"{specificity:.4f}")
        st.write("TN / (TN + FP)")
    
    st.write("""
    💡 **Metrics Explained:**
    - **Accuracy**: Overall correctness of the model
    - **Precision**: Accuracy of positive predictions
    - **Recall (Sensitivity)**: Fraction of actual positives correctly identified
    - **F1 Score**: Harmonic mean of precision and recall
    - **Specificity**: Fraction of actual negatives correctly identified
    """)

def quiz_section():
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "What does the True Positive (TP) represent in a confusion matrix?",
            "options": [
                "Correctly predicted positive values",
                "Incorrectly predicted positive values",
                "Correctly predicted negative values",
                "Incorrectly predicted negative values"
            ],
            "correct": "Correctly predicted positive values",
            "explanation": "True Positive (TP) represents the number of positive instances that were correctly identified by the model. For example, in a disease detection model, TP would be the number of sick patients correctly identified as sick."
        },
        {
            "question": "How is accuracy calculated using the confusion matrix components?",
            "options": [
                "(TP + TN) / (TP + TN + FP + FN)",
                "TP / (TP + FP)",
                "TP / (TP + FN)",
                "TN / (TN + FP)"
            ],
            "correct": "(TP + TN) / (TP + TN + FP + FN)",
            "explanation": "Accuracy is calculated as (TP + TN) / (TP + TN + FP + FN). It represents the overall correctness of the model by considering all correct predictions (both positive and negative) over the total number of cases."
        },
        {
            "question": "What metric is best to use when the classes in your dataset are imbalanced?",
            "options": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ],
            "correct": "F1 Score",
            "explanation": "The F1 Score is often the best metric for imbalanced datasets. It's the harmonic mean of precision and recall, providing a balanced measure that takes both false positives and false negatives into account. Accuracy can be misleading for imbalanced datasets, as it might show high values even when the model performs poorly on the minority class."
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