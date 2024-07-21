import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
import plotly.express as px

# Set page config
st.set_page_config(layout="wide", page_title="Linear vs Logistic Regression Explorer", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px !important;
        font-weight: bold;
        color: #4B0082;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .tab-subheader {
        font-size: 28px !important;
        font-weight: bold;
        color: #8A2BE2;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .content-text {
        font-size: 18px !important;
        line-height: 1.6;
    }
    .stButton>button {
        background-color: #9370DB;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #8A2BE2;
        transform: scale(1.05);
    }
    .highlight {
        background-color: #E6E6FA;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .quiz-question {
        background-color: #F0E6FA;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #8A2BE2;
    }
    .explanation {
        background-color: #E6F3FF;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>📊 Linear vs Logistic Regression Explorer 📊</h1>", unsafe_allow_html=True)

# Functions (same as before)
def generate_data(num_points, noise_level, logistic_x_shift):
    x = np.random.uniform(-5, 5, num_points)
    y_linear = 2*x + np.random.normal(0, noise_level, num_points)
    y_logistic = 1 / (1 + np.exp(-(x-logistic_x_shift)))
    y_logistic = np.where(np.random.rand(num_points) < y_logistic, 1, 0)
    return x, y_linear, y_logistic

def plot_data(x, y, mode, title, color, fit_line=None, fit_color=None, threshold=None):
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode=mode, marker=dict(color=color)))
    if fit_line is not None:
        fig.add_trace(go.Scatter(x=x, y=fit_line, mode='lines', line=dict(color=fit_color)))
    if threshold is not None:
        fig.add_shape(type='line', x0=x.min(), x1=x.max(), y0=threshold, y1=threshold,
                      line=dict(color='black', dash='dash'))
        fig.add_annotation(x=x.mean(), y=threshold, text=f'Decision Threshold: {threshold:.2f}',
                           showarrow=False, yshift=10)
    fig.update_layout(title=title, xaxis_title='Independent Variable', yaxis_title='Dependent Variable')
    return fig

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualization", "🧮 Interactive Calculator", "🎓 Learn More", "🧠 Quiz"])

# Tab 1, 2, 3 content remains the same as in the previous version

with tab4:
    st.markdown("<h2 class='tab-subheader'>Test Your Regression Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "Which type of regression would be most appropriate for predicting house prices based on square footage?",
            "options": ["Linear Regression", "Logistic Regression", "Both", "Neither"],
            "correct": 0,
            "explanation": "Linear regression is suitable for predicting continuous values like house prices based on features like square footage."
        },
        {
            "question": "In logistic regression, what does the output represent?",
            "options": ["Exact values", "Probabilities", "Categories", "Errors"],
            "correct": 1,
            "explanation": "Logistic regression outputs probabilities, typically representing the likelihood of an instance belonging to a particular class."
        },
        {
            "question": "Which regression type is more suitable for classifying emails as spam or not spam?",
            "options": ["Linear Regression", "Logistic Regression", "Both", "Neither"],
            "correct": 1,
            "explanation": "Logistic regression is ideal for binary classification tasks like spam detection, where the outcome is either spam (1) or not spam (0)."
        },
        {
            "question": "What shape does the logistic regression function typically have?",
            "options": ["Straight line", "Parabola", "S-curve (sigmoid)", "Circle"],
            "correct": 2,
            "explanation": "The logistic function has an S-shaped curve (sigmoid), which maps any input to a value between 0 and 1."
        }
    ]

    score = 0
    for i, q in enumerate(questions):
        st.markdown(f"<div class='quiz-question'>", unsafe_allow_html=True)
        st.markdown(f"<p class='content-text'><strong>Question {i+1}:</strong> {q['question']}</p>", unsafe_allow_html=True)
        user_answer = st.radio("Select your answer:", q['options'], key=f"q{i}")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Check Answer", key=f"check{i}"):
                if q['options'].index(user_answer) == q['correct']:
                    st.success("Correct! 🎉")
                    score += 1
                else:
                    st.error("Incorrect. Try again! 🤔")
                st.markdown(f"<div class='explanation'><p>{q['explanation']}</p></div>", unsafe_allow_html=True)
        
        with col2:
            if i == 0:  # Example visualization for house prices
                x = np.array([1000, 1500, 2000, 2500, 3000])
                y = 100000 + 200 * x + np.random.normal(0, 10000, 5)
                fig = px.scatter(x=x, y=y, labels={'x': 'Square Footage', 'y': 'Price'})
                fig.add_trace(go.Scatter(x=x, y=100000 + 200 * x, mode='lines', name='Linear Regression'))
                fig.update_layout(title="House Price vs Square Footage")
                st.plotly_chart(fig, use_container_width=True)
            elif i == 1:  # Logistic regression output visualization
                x = np.linspace(-5, 5, 100)
                y = 1 / (1 + np.exp(-x))
                fig = px.line(x=x, y=y, labels={'x': 'Input', 'y': 'Probability'})
                fig.update_layout(title="Logistic Regression Output")
                st.plotly_chart(fig, use_container_width=True)
            elif i == 2:  # Spam classification visualization
                x = np.random.rand(100)
                y = (x > 0.5).astype(int)
                fig = px.scatter(x=x, y=y, labels={'x': 'Feature', 'y': 'Spam (1) or Not Spam (0)'})
                fig.update_layout(title="Email Spam Classification")
                st.plotly_chart(fig, use_container_width=True)
            elif i == 3:  # Sigmoid function visualization
                x = np.linspace(-10, 10, 100)
                y = 1 / (1 + np.exp(-x))
                fig = px.line(x=x, y=y, labels={'x': 'Input', 'y': 'Output'})
                fig.update_layout(title="Logistic (Sigmoid) Function")
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

    if st.button("Show Final Score"):
        st.markdown(f"<p class='tab-subheader'>Your score: {score}/{len(questions)}</p>", unsafe_allow_html=True)
        if score == len(questions):
            st.balloons()
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're a regression expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering regression concepts. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)

        # Visualization of score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Quiz Score"},
            gauge = {
                'axis': {'range': [None, len(questions)]},
                'steps': [
                    {'range': [0, len(questions)//2], 'color': "lightgray"},
                    {'range': [len(questions)//2, len(questions)], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': score}}))
        st.plotly_chart(fig, use_container_width=True)

# Conclusion
st.markdown("<h2 class='tab-subheader'>Happy Exploring! 🎊</h2>", unsafe_allow_html=True)
st.markdown("""
<p class='content-text'>
You've explored the world of linear and logistic regression! Remember:

1. Linear regression helps predict continuous outcomes.
2. Logistic regression is great for binary classification problems.
3. The choice between them depends on your specific problem and data.
4. Visualizing the data and regression lines helps in understanding the relationships.
5. Always consider the assumptions and limitations of each method.

Keep exploring and applying these regression techniques in your data analysis journey!
</p>
""", unsafe_allow_html=True)
