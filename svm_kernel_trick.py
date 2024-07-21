import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import plotly.express as px

# Set page config
st.set_page_config(layout="wide", page_title="SVM Kernel Trick Explorer", page_icon="🧠")

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
st.markdown("<h1 class='main-header'>🧠 Support Vector Machines: The Kernel Trick 🧠</h1>", unsafe_allow_html=True)
st.markdown("<p class='content-text'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)

# Functions
def kernel_transform(X):
    return np.array([[x[0], x[1], x[0]**2 + x[1]**2] for x in X])

def plot_data(X, y, title):
    fig = go.Figure()
    
    colors = ['green' if label == 1 else 'red' for label in y]
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=colors, size=10)))
    fig.update_layout(width=600, height=600, title=title)
    return fig

def plot_3d_data(X, y, title):
    X_transformed = kernel_transform(X)
    
    fig = go.Figure()
    
    colors = ['green' if label == 1 else 'red' for label in y]
    fig.add_trace(go.Scatter3d(x=X_transformed[:, 0], y=X_transformed[:, 1], z=X_transformed[:, 2], 
                               mode='markers', marker=dict(color=colors, size=5)))
    fig.update_layout(width=800, height=800, title=title, 
                      scene=dict(xaxis_title='X1', yaxis_title='X2', zaxis_title='X1^2 + X2^2'))
    return fig

def generate_data(num_points):
    X = np.random.randn(num_points, 2)
    y = np.array([1 if np.linalg.norm(x) < 1 else -1 for x in X])
    return X, y

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🧮 SVM Models", "🎓 Learn More", "🧠 Quiz"])

with tab1:
    st.markdown("<h2 class='tab-subheader'>Data Visualization</h2>", unsafe_allow_html=True)
    
    np.random.seed(0)
    num_points = st.slider('Number of data points:', 10, 100, 50, 10)
    test_size = st.slider('Test data size (%):', 10, 50, 20, 5)
    
    X, y = generate_data(num_points)
    
    test_size = int(num_points * test_size / 100)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<p class='content-text'>Original 2D Data:</p>", unsafe_allow_html=True)
        fig_2d = plot_data(X, y, '2D Data Points (Not Linearly Separable)')
        st.plotly_chart(fig_2d, use_container_width=True)
    
    with col2:
        st.markdown("<p class='content-text'>Transformed 3D Data:</p>", unsafe_allow_html=True)
        fig_3d = plot_3d_data(X, y, '3D Transformed Data Points (Linearly Separable)')
        st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    st.markdown("<h2 class='tab-subheader'>SVM Models Comparison</h2>", unsafe_allow_html=True)
    
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(X_train, y_train)
    y_pred_linear = svm_linear.predict(X_test)
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    
    svm_rbf = SVC(kernel='rbf')
    svm_rbf.fit(X_train, y_train)
    y_pred_rbf = svm_rbf.predict(X_test)
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
    
    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.markdown("<p class='content-text'><strong>Prediction Accuracy:</strong></p>", unsafe_allow_html=True)
    st.write(f'Linear SVM Accuracy: {accuracy_linear:.2f}')
    st.write(f'RBF Kernel SVM Accuracy: {accuracy_rbf:.2f}')
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualization of accuracies
    fig = go.Figure(data=[
        go.Bar(name='Linear SVM', x=['Accuracy'], y=[accuracy_linear]),
        go.Bar(name='RBF Kernel SVM', x=['Accuracy'], y=[accuracy_rbf])
    ])
    fig.update_layout(title='SVM Models Accuracy Comparison', yaxis_title='Accuracy')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("<h2 class='tab-subheader'>Learn More About SVM and Kernel Trick</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    <b>Support Vector Machines (SVM):</b> A powerful machine learning algorithm used for classification and regression tasks.
    
    <b>The Kernel Trick:</b> A technique used in SVM to transform data from low dimensions to high dimensions, making it easier to find a separating hyperplane.
    
    Key points:
    1. Not all data is linearly separable in its original space.
    2. The kernel trick offers an efficient way to transform data to higher dimensions.
    3. It utilizes pairwise comparisons in the original data points.
    4. Common kernels include Linear, Polynomial, and Radial Basis Function (RBF).
    
    <b>Advantages of the Kernel Trick:</b>
    - Allows SVM to handle non-linear decision boundaries.
    - Computationally efficient, as it doesn't actually compute the high-dimensional space.
    - Versatile, with different kernels suitable for various types of data.
    
    <b>Applications:</b>
    - Image classification
    - Text categorization
    - Bioinformatics
    - Financial analysis
    </p>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown("<h2 class='tab-subheader'>Test Your SVM Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What is the main advantage of using the Kernel Trick in SVMs?",
            "options": [
                "It reduces the number of support vectors",
                "It allows handling non-linear decision boundaries without explicit transformation",
                "It speeds up the training process",
                "It reduces overfitting"
            ],
            "correct": 1,
            "explanation": "The Kernel Trick allows SVMs to handle non-linear decision boundaries by implicitly mapping the data to a higher-dimensional space without actually performing the transformation."
        },
        {
            "question": "Which of the following is NOT a commonly used kernel in SVMs?",
            "options": ["Linear", "Polynomial", "Radial Basis Function (RBF)", "Logarithmic"],
            "correct": 3,
            "explanation": "Linear, Polynomial, and RBF are common kernels used in SVMs. Logarithmic is not a standard kernel."
        },
        {
            "question": "What does the C parameter in SVM control?",
            "options": [
                "The kernel function",
                "The number of support vectors",
                "The trade-off between margin maximization and misclassification",
                "The dimensionality of the feature space"
            ],
            "correct": 2,
            "explanation": "The C parameter in SVM controls the trade-off between having a smooth decision boundary and classifying training points correctly."
        },
        {
            "question": "In the context of SVMs, what is a support vector?",
            "options": [
                "The vector that defines the hyperplane",
                "Any data point in the dataset",
                "The data points closest to the decision boundary",
                "The mean of all data points"
            ],
            "correct": 2,
            "explanation": "Support vectors are the data points that lie closest to the decision surface. They are the most difficult to classify and directly influence the optimal location of the decision boundary."
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
            if i == 0:  # Visualization for non-linear decision boundary
                X, y = generate_data(200)
                fig = plot_data(X, y, "Non-linear Decision Boundary")
                st.plotly_chart(fig, use_container_width=True)
            elif i == 1:  # Visualization of different kernels
                x = np.linspace(-5, 5, 100)
                linear = x
                poly = x**2
                rbf = np.exp(-x**2)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=linear, mode='lines', name='Linear'))
                fig.add_trace(go.Scatter(x=x, y=poly, mode='lines', name='Polynomial'))
                fig.add_trace(go.Scatter(x=x, y=rbf, mode='lines', name='RBF'))
                fig.update_layout(title="Different Kernel Functions")
                st.plotly_chart(fig, use_container_width=True)
            elif i == 2:  # Visualization of C parameter effect
                st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_rbf_parameters_001.png", caption="Effect of C parameter")
            elif i == 3:  # Visualization of support vectors
                st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_svm_margin_001.png", caption="Support Vectors Illustration")
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

    if st.button("Show Final Score"):
        st.markdown(f"<p class='tab-subheader'>Your score: {score}/{len(questions)}</p>", unsafe_allow_html=True)
        if score == len(questions):
            st.balloons()
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're an SVM expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering SVMs. Keep learning! 📚</p>", unsafe_allow_html=True)
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
st.markdown("""
<p class='content-text'>
You've explored the power of Support Vector Machines and the Kernel Trick! Remember:

1. SVMs are versatile classifiers that can handle both linear and non-linear data.
2. The Kernel Trick allows SVMs to find complex decision boundaries without expensive computations.
3. Different kernels (Linear, RBF) can lead to different model performances.
4. Visualizing data transformation helps understand how the Kernel Trick works.
5. The C parameter in SVM controls the trade-off between margin maximization and misclassification.
6. Support vectors are the key data points that define the decision boundary.

Keep exploring and applying these concepts in your machine learning journey!
</p>
""", unsafe_allow_html=True)



# Add references
st.markdown("<h2 class='tab-subheader'>References</h2>", unsafe_allow_html=True)
st.markdown("""
<p class='content-text'>
1. Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine learning, 20(3), 273-297.
2. Schölkopf, B., & Smola, A. J. (2002). Learning with kernels: support vector machines, regularization, optimization, and beyond. MIT press.
3. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
</p>
""", unsafe_allow_html=True)

# Add a footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0E1117;
    color: #FAFAFA;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>
<div class='footer'>
    Created with ❤️ by Venugopal Adep | © 2023 All Rights Reserved
</div>
""", unsafe_allow_html=True)
