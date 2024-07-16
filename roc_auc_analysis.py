import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd
import plotly.express as px

def main():
    st.set_page_config(page_title="AUC and ROC Analysis Demo", layout="wide")
    
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
    
    st.title("🎯 Understanding AUC and ROC Analysis")
    
    tabs = st.tabs(["📚 Learn", "🧪 Experiment", "🏥 Real-world Example", "🧠 Quiz"])
    
    with tabs[0]:
        learn_section()
    
    with tabs[1]:
        experiment_section()
    
    with tabs[2]:
        example_section()
    
    with tabs[3]:
        quiz_section()

def learn_section():
    st.header("What are ROC and AUC?")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        📊 **ROC (Receiver Operating Characteristic) Curve**:
        - A graph showing the performance of a classification model at all classification thresholds.
        - Plots True Positive Rate vs False Positive Rate.
        
        📈 **AUC (Area Under the Curve)**:
        - A single number summarizing the ROC curve.
        - Represents the model's ability to distinguish between classes.
        - Ranges from 0 to 1 (higher is better).
        """)
    
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/3/36/ROC_space-2.png", caption="ROC Curve Example")
    
    st.subheader("📐 Important Formulas:")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r"True\,Positive\,Rate\,(TPR) = \frac{True\,Positives}{True\,Positives + False\,Negatives}")
        st.latex(r"False\,Positive\,Rate\,(FPR) = \frac{False\,Positives}{False\,Positives + True\,Negatives}")
    with col2:
        st.latex(r"AUC = \int_{0}^{1} TPR\,d(FPR)")
        st.write("AUC is the area under the ROC curve, representing the model's overall performance.")

def experiment_section():
    st.header("🔬 Interactive ROC Curve Experiment")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Adjust Model Performance")
        tpr = st.slider("True Positive Rate", 0.0, 1.0, 0.8, 0.01)
        fpr = st.slider("False Positive Rate", 0.0, 1.0, 0.2, 0.01)
        
        threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01)
        st.write(f"Threshold: {threshold:.2f}")
        st.write("Lower threshold → Higher TPR and FPR")
        st.write("Higher threshold → Lower TPR and FPR")
    
    with col2:
        # Generate ROC curve points
        fpr_points = np.linspace(0, 1, 100)
        tpr_points = np.minimum(1, tpr / fpr * fpr_points)
        
        # Calculate AUC
        auc_value = auc([0, fpr, 1], [0, tpr, 1])
        
        # Create ROC curve plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr_points, y=tpr_points, mode='lines', name='ROC Curve', line=dict(color='#ff6e40', width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash', color='#1e3d59')))
        fig.add_trace(go.Scatter(x=[fpr], y=[tpr], mode='markers', name='Current Point', marker=dict(size=12, color='#ffa41b', symbol='star')))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500,
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig)
        
        st.metric("Current AUC", f"{auc_value:.3f}")
        
        st.write("""
        🔍 **Experiment with the sliders to see how changing the True Positive Rate and False Positive Rate affects the ROC curve and AUC.**
        
        - A perfect classifier would have a point at (0, 1), indicating no false positives and all true positives.
        - The diagonal line represents a random classifier (AUC = 0.5).
        - The closer the curve follows the top-left corner, the better the model's performance (higher AUC).
        """)

def example_section():
    st.header("🏥 Real-world Example: Medical Diagnosis")
    st.write("""
    Let's consider a medical test for diagnosing a disease:
    
    - True Positive (TP): Correctly identified sick patients
    - False Positive (FP): Healthy patients incorrectly identified as sick
    - True Negative (TN): Correctly identified healthy patients
    - False Negative (FN): Sick patients incorrectly identified as healthy
    """)
    
    # Interactive confusion matrix
    st.subheader("📊 Interactive Confusion Matrix")
    col1, col2 = st.columns(2)
    with col1:
        tp = st.number_input("True Positives", min_value=0, max_value=100, value=40)
        fn = st.number_input("False Negatives", min_value=0, max_value=100, value=10)
    with col2:
        fp = st.number_input("False Positives", min_value=0, max_value=100, value=5)
        tn = st.number_input("True Negatives", min_value=0, max_value=100, value=45)
    
    confusion_matrix = pd.DataFrame({
        'Actual Positive': [tp, fn],
        'Actual Negative': [fp, tn]
    }, index=['Predicted Positive', 'Predicted Negative'])
    
    fig = px.imshow(confusion_matrix, text_auto=True, color_continuous_scale='YlOrRd')
    fig.update_layout(width=500, height=500)
    st.plotly_chart(fig)
    
    # Calculate metrics
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    precision = tp / (tp + fp)
    f1_score = 2 * (precision * tpr) / (precision + tpr)
    
    st.subheader("📈 Calculated Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("True Positive Rate (Sensitivity)", f"{tpr:.2f}")
        st.metric("False Positive Rate", f"{fpr:.2f}")
        st.metric("Accuracy", f"{accuracy:.2f}")
    with col2:
        st.metric("Precision", f"{precision:.2f}")
        st.metric("F1 Score", f"{f1_score:.2f}")
    
    st.write("""
    💡 **Interpretation:**
    - A high True Positive Rate (Sensitivity) indicates the test is good at identifying sick patients.
    - A low False Positive Rate means the test rarely misidentifies healthy patients as sick.
    - High Accuracy suggests overall good performance, but it's important to consider other metrics too.
    - Precision shows how many of the positive predictions are actually correct.
    - F1 Score is the harmonic mean of precision and sensitivity, providing a balanced measure of the test's performance.
    """)

def quiz_section():
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "What does a diagonal line in an ROC curve (from bottom-left to top-right) represent?",
            "options": ["A perfect classifier", "A random classifier", "The worst possible classifier", "It has no specific meaning"],
            "correct": "A random classifier",
            "explanation": "A diagonal line in an ROC curve represents a classifier that performs no better than random guessing. It has an AUC of 0.5, meaning it correctly identifies positive cases 50% of the time, which is equivalent to flipping a coin."
        },
        {
            "question": "If a model has an AUC of 0.9, what does this indicate about its performance?",
            "options": ["Poor performance", "Average performance", "Good performance", "Perfect performance"],
            "correct": "Good performance",
            "explanation": "AUC ranges from 0 to 1, where 1 represents a perfect classifier. An AUC of 0.9 is quite high, indicating that the model has a 90% chance of correctly distinguishing between positive and negative classes. In practice, this would be considered good performance for most applications."
        },
        {
            "question": "What happens to the ROC curve as you increase the False Positive Rate?",
            "options": ["It moves towards the top-left corner", "It moves towards the bottom-right corner", "It moves upward", "It doesn't change"],
            "correct": "It moves upward",
            "explanation": "As you increase the False Positive Rate, you're essentially lowering the threshold for classifying positive instances. This means you'll catch more true positives (increasing True Positive Rate) but at the cost of more false positives. On the ROC plot, this translates to moving upward and to the right. You can verify this by adjusting the False Positive Rate slider in the interactive demo above."
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