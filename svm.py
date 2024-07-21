import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set page config
st.set_page_config(layout="wide", page_title="SVM Interactive Tool", page_icon="🤖")

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
st.markdown("<h1 class='main-header'>🤖 Support Vector Machines (SVM) Interactive Tool 🤖</h1>", unsafe_allow_html=True)
st.markdown("<p class='content-text'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)

def plot_svm(X, y, svc):
    # Create a mesh grid for plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict the class for each point in the mesh grid
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create the plot
    fig = go.Figure(data=[
        go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='RdBu', showscale=False, opacity=0.8),
        go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y, colorscale='RdBu', size=10))
    ])

    fig.update_layout(title='Support Vector Machine Decision Boundary',
                      xaxis_title='Feature 1',
                      yaxis_title='Feature 2',
                      plot_bgcolor='white',
                      height=600,
                      width=800,
                      font=dict(size=14))

    return fig

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 SVM Visualization", "🧮 Model Training", "🎓 Learn More", "🧠 Quiz"])

with tab1:
    st.markdown("<h2 class='tab-subheader'>SVM Visualization</h2>", unsafe_allow_html=True)

    # Sidebar layout
    st.sidebar.title("Options")

    # Dataset selection
    dataset_options = ['Moons', 'Circles', 'Blobs']
    selected_dataset = st.sidebar.selectbox("Select a dataset", dataset_options)

    # Kernel selection
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    selected_kernel = st.sidebar.selectbox("Select a kernel function", kernel_options)

    # Regularization parameter (C)
    c_value = st.sidebar.slider("Regularization parameter (C)", min_value=0.1, max_value=10.0, step=0.1, value=1.0)

    # Generate the selected dataset
    if selected_dataset == 'Moons':
        X, y = datasets.make_moons(noise=0.3, random_state=0)
    elif selected_dataset == 'Circles':
        X, y = datasets.make_circles(noise=0.2, factor=0.5, random_state=0)
    else:
        X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the SVM model
    svc = SVC(kernel=selected_kernel, C=c_value)
    svc.fit(X_train_scaled, y_train)

    # Plot the decision boundary
    fig = plot_svm(X_train_scaled, y_train, svc)
    st.plotly_chart(fig)

with tab2:
    st.markdown("<h2 class='tab-subheader'>Model Training and Evaluation</h2>", unsafe_allow_html=True)
    
    # Evaluate the model
    y_pred = svc.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.write(f"Test Accuracy: {accuracy:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <p class='content-text'>
    The model's performance can be affected by:
    - The chosen dataset
    - The selected kernel function
    - The regularization parameter (C)
    
    Try adjusting these parameters in the sidebar to see how they impact the decision boundary and model accuracy.
    </p>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("<h2 class='tab-subheader'>Learn More About SVM</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    <b>Support Vector Machines (SVM)</b> is a supervised machine learning algorithm used for classification and regression tasks. In the case of binary classification, SVM aims to find the optimal hyperplane that separates the two classes with the maximum margin.

    Key concepts in SVM:
    1. <b>Support Vectors:</b> The data points closest to the decision boundary. These points have the most influence on the position and orientation of the hyperplane.
    2. <b>Hyperplane:</b> The decision boundary that separates the two classes. It is a line in 2D space or a plane in higher-dimensional space.
    3. <b>Margin:</b> The distance between the hyperplane and the closest data points from each class. SVM tries to maximize this margin for better generalization.
    4. <b>Kernel Trick:</b> A technique that allows SVM to operate in high-dimensional feature spaces without explicitly computing the coordinates of the data in that space.

    <b>Interpreting the Plot:</b>
    - The data points are represented as scatter points, with their color indicating the true class label.
    - The decision boundary is shown as a contour line separating the classes.
    - The margin is the distance between the decision boundary and the closest data points from each class.

    Experiment with different datasets, kernel functions, and regularization parameters to see how they affect the decision boundary and the model's performance.
    </p>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown("<h2 class='tab-subheader'>Test Your SVM Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "What does SVM stand for?",
            "options": ["Support Vector Machine", "Standard Variable Model", "System Validation Method", "Statistical Variance Measure"],
            "correct": 0,
            "explanation": "SVM stands for Support Vector Machine, which is a powerful supervised learning algorithm used for classification and regression tasks."
        },
        {
            "question": "What is the main goal of SVM in binary classification?",
            "options": ["Minimize the margin", "Maximize the margin", "Minimize the number of support vectors", "Maximize the number of support vectors"],
            "correct": 1,
            "explanation": "In binary classification, SVM aims to find the optimal hyperplane that separates the two classes with the maximum margin for better generalization."
        },
        {
            "question": "What is the role of the kernel function in SVM?",
            "options": ["To reduce the number of features", "To increase the number of support vectors", "To transform the data into a higher-dimensional space", "To decrease the model's complexity"],
            "correct": 2,
            "explanation": "The kernel function in SVM allows the algorithm to operate in high-dimensional feature spaces without explicitly computing the coordinates of the data in that space, enabling it to handle non-linear decision boundaries."
        },
        {
            "question": "What does the regularization parameter (C) control in SVM?",
            "options": ["The number of features", "The trade-off between margin maximization and misclassification", "The kernel function", "The number of support vectors"],
            "correct": 1,
            "explanation": "The regularization parameter (C) in SVM controls the trade-off between having a smooth decision boundary (larger margin) and classifying the training points correctly (smaller margin)."
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
            st.markdown("<p class='content-text' style='color: green; font-weight: bold;'>Congratulations! You're an SVM expert! 🏆</p>", unsafe_allow_html=True)
        elif score >= len(questions) // 2:
            st.markdown("<p class='content-text' style='color: blue;'>Good job! You're on your way to mastering SVMs. Keep learning! 📚</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='content-text' style='color: orange;'>You're making progress! Review the explanations and try again to improve your score. 💪</p>", unsafe_allow_html=True)

# Conclusion
st.markdown("<h2 class='tab-subheader'>Explore and Learn! 🚀</h2>", unsafe_allow_html=True)
st.markdown("""
<p class='content-text'>
You've explored the power of Support Vector Machines! Remember:

1. SVMs find the optimal hyperplane to separate classes with maximum margin.
2. The kernel trick allows SVMs to handle non-linear decision boundaries.
3. The regularization parameter (C) controls the trade-off between margin size and misclassification.
4. Different datasets and kernels can lead to different decision boundaries and model performance.
5. Visualizing the decision boundary helps in understanding how SVM classifies data points.

Keep exploring and applying these concepts in your machine learning journey!
</p>
""", unsafe_allow_html=True)

# Footer
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
