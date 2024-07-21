import streamlit as st
import plotly.graph_objects as go
import random
import pandas as pd

# Set page config
st.set_page_config(layout="wide", page_title="Confusion Matrix Explorer", page_icon="🧮")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px !important;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .tab-subheader {
        font-size: 28px !important;
        font-weight: bold;
        color: #3CB371;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .content-text {
        font-size: 18px !important;
        line-height: 1.6;
    }
    .stButton>button {
        background-color: #66CDAA;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #3CB371;
        transform: scale(1.05);
    }
    .highlight {
        background-color: #E0FFF0;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .interpretation {
        background-color: #F0FFF0;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #3CB371;
    }
    .quiz-question {
        background-color: #F0FFF0;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #3CB371;
    }
    .explanation {
        background-color: #E0FFF0;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🧮 Understanding Confusion Matrix 🧮</h1>", unsafe_allow_html=True)
st.markdown("<p class='content-text'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)

# Initialize session state
if 'tp' not in st.session_state:
    st.session_state.tp = st.session_state.tn = st.session_state.fp = st.session_state.fn = 0

# Sidebar
st.sidebar.title("Controls")
if st.sidebar.button('Generate Random Confusion Matrix'):
    st.session_state.tp = random.randint(0, 100)
    st.session_state.tn = random.randint(0, 100)
    st.session_state.fp = random.randint(0, 100)
    st.session_state.fn = random.randint(0, 100)

metric = st.sidebar.selectbox('Select Metric', ('Accuracy', 'Precision', 'Recall', 'F1 Score'))

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Confusion Matrix", "📊 Metrics", "🎓 Learn More", "🧠 Quiz"])

with tab1:
    st.markdown("<h2 class='tab-subheader'>Confusion Matrix Visualization</h2>", unsafe_allow_html=True)

    CM = [[f'FN<br>{st.session_state.fn}', f'TN<br>{st.session_state.tn}'],
          [f'TP<br>{st.session_state.tp}', f'FP<br>{st.session_state.fp}']]
    
    fig = go.Figure(data=go.Heatmap(
        z=[[st.session_state.fn, st.session_state.tn], [st.session_state.tp, st.session_state.fp]], 
        text=CM,
        texttemplate="%{text}",
        textfont={"size":20},
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Positive', 'Actual Negative'],
        hoverongaps = False,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        width=800, 
        height=600, 
        title_text='Confusion Matrix',
        title_font_size=24
    )
    
    st.plotly_chart(fig)

    st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
    st.markdown("""
    <p class='content-text'>
    <strong>Interpretation:</strong>
    - TP (True Positive): Correctly identified positive cases
    - TN (True Negative): Correctly identified negative cases
    - FP (False Positive): Incorrectly identified positive cases (Type I error)
    - FN (False Negative): Incorrectly identified negative cases (Type II error)
    </p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<h2 class='tab-subheader'>Performance Metrics</h2>", unsafe_allow_html=True)
    
    total = st.session_state.tp + st.session_state.tn + st.session_state.fp + st.session_state.fn
    accuracy = (st.session_state.tp + st.session_state.tn) / total if total > 0 else 0
    precision = st.session_state.tp / (st.session_state.tp + st.session_state.fp) if st.session_state.tp + st.session_state.fp > 0 else 0
    recall = st.session_state.tp / (st.session_state.tp + st.session_state.fn) if st.session_state.tp + st.session_state.fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    metrics_data = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    })

    fig = go.Figure(data=[go.Bar(x=metrics_data['Metric'], y=metrics_data['Value'])])
    fig.update_layout(title_text='Performance Metrics', xaxis_title='Metric', yaxis_title='Value')
    st.plotly_chart(fig)

    if metric == 'Accuracy':
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("<h3 class='content-text'>Accuracy</h3>", unsafe_allow_html=True)
        st.markdown("<p class='content-text'>Accuracy is the proportion of correct predictions (both true positives and true negatives) among the total number of cases examined.</p>", unsafe_allow_html=True)
        st.latex(r'''Accuracy = \frac{TP+TN}{TP+FP+FN+TN}''')
        st.markdown(f"<p class='content-text'><strong>Accuracy: {accuracy:.2f}</strong></p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    elif metric == 'Precision':
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("<h3 class='content-text'>Precision</h3>", unsafe_allow_html=True)
        st.markdown("<p class='content-text'>Precision is the proportion of true positives among all positive predictions. It answers the question: 'Of all the patients we predicted to have the disease, how many actually have it?'</p>", unsafe_allow_html=True)
        st.latex(r'''Precision = \frac{TP}{TP+FP}''')
        st.markdown(f"<p class='content-text'><strong>Precision: {precision:.2f}</strong></p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    elif metric == 'Recall':
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("<h3 class='content-text'>Recall</h3>", unsafe_allow_html=True)
        st.markdown("<p class='content-text'>Recall is the proportion of true positives among all actual positives. It answers the question: 'Of all the patients who actually have the disease, how many did we correctly identify?'</p>", unsafe_allow_html=True)
        st.latex(r'''Recall = \frac{TP}{TP+FN}''')
        st.markdown(f"<p class='content-text'><strong>Recall: {recall:.2f}</strong></p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("<h3 class='content-text'>F1 Score</h3>", unsafe_allow_html=True)
        st.markdown("<p class='content-text'>F1 Score is the harmonic mean of precision and recall. It provides a balanced evaluation of the model's performance.</p>", unsafe_allow_html=True)
        st.latex(r'''F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}''')
        st.markdown(f"<p class='content-text'><strong>F1 Score: {f1:.2f}</strong></p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<h2 class='tab-subheader'>Learn More About Confusion Matrix</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    A confusion matrix is a table used to describe the performance of a classification model on a set of test data for which the true values are known. It allows visualization of the performance of an algorithm.

    <b>Key Components:</b>
    1. True Positive (TP): Correctly predicted positive class
    2. True Negative (TN): Correctly predicted negative class
    3. False Positive (FP): Incorrectly predicted positive class
    4. False Negative (FN): Incorrectly predicted negative class

    <b>Use Cases:</b>
    - Medical Diagnosis: Evaluating the accuracy of disease detection tests
    - Spam Detection: Assessing email filtering algorithms
    - Fraud Detection: Measuring the effectiveness of fraud identification systems

    Understanding the confusion matrix and its derived metrics is crucial for:
    1. Evaluating model performance
    2. Choosing appropriate metrics for specific problems
    3. Balancing trade-offs between different types of errors

    Experiment with different scenarios to see how changes in the confusion matrix affect various performance metrics!
    </p>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown("<h2 class='tab-subheader'>Test Your Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What does a True Positive (TP) represent in a confusion matrix?",
            "options": [
                "Correctly predicted negative class",
                "Correctly predicted positive class",
                "Incorrectly predicted positive class",
                "Incorrectly predicted negative class"
            ],
            "correct": 1,
            "explanation": "A True Positive (TP) represents a case where the model correctly predicted the positive class."
        },
        {
            "question": "Which metric is the harmonic mean of precision and recall?",
            "options": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ],
            "correct": 3,
            "explanation": "The F1 Score is the harmonic mean of precision and recall, providing a balanced measure of the model's performance."
        },
        {
            "question": "What does high precision and low recall indicate?",
            "options": [
                "The model is good at identifying positive cases but misses many",
                "The model identifies most positive cases but has many false positives",
                "The model is equally good at identifying positive and negative cases",
                "The model is poor at identifying both positive and negative cases"
            ],
            "correct": 0,
            "explanation": "High precision and low recall indicate that the model is accurate when it predicts positive cases, but it misses many actual positive cases."
        },
        {
            "question": "In a medical test context, what is a Type II error?",
            "options": [
                "False Positive",
                "False Negative",
                "True Positive",
                "True Negative"
            ],
            "correct": 1,
            "explanation": "A Type II error is a False Negative, where the test fails to detect the presence of a condition when it is actually present."
        }
    ]

    score = 0
    for i, q in enumerate(questions):
        st.markdown(f"<div class='quiz-question'>", unsafe_allow_html=True)
        st.markdown(f"<p class='content-text'><strong>Question {i+1}:</strong> {q['question']}</p>", unsafe_allow_html=True)
        user_answer = st.radio("Select your answer:", q['options'], key=f"q{i}")
        
        if st.button("Check Answer", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! 🎉")
                score += 1
            else:
                st.error("Incorrect. Try again! 🤔")
            st.markdown(f"<div class='explanation'><p>{q['explanation']}</p></div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

    if st.button("Show Final Score"):
        st.markdown(f"<p class='tab-subheader'>Your score: {score}/{len(questions)}</p>", unsafe_allow_html=True)
        if score == len(questions):
            st.balloons()
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're a Confusion Matrix expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering Confusion Matrices. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)
