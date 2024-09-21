import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(layout="wide", page_title="SVM Interactive Tool", page_icon="🤖")

# Custom CSS (unchanged)
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

def plot_svm(X, y, y_pred, svc):
    # Create a mesh grid for plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict the class for each point in the mesh grid
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create the plot with subplots
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], 
                        specs=[[{"type": "xy"}, {"type": "table"}]])

    # Add decision boundary contour
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z,
        colorscale=[[0, 'lightblue'], [1, 'mistyrose']],
        opacity=0.5,
        showscale=False
    ), row=1, col=1)

    # Add scatter plot for actual data points
    colors = ['blue' if label == 0 else 'red' for label in y]
    markers = ['circle' if pred == actual else 'x' for pred, actual in zip(y_pred, y)]
    
    hover_texts = []
    for actual, pred in zip(y, y_pred):
        if actual == 0 and pred == 0:
            hover_texts.append("TN (BpB)")
        elif actual == 1 and pred == 1:
            hover_texts.append("TP (RpR)")
        elif actual == 0 and pred == 1:
            hover_texts.append("FP (BpR)")
        else:
            hover_texts.append("FN (RpB)")

    for color, marker in [('blue', 'circle'), ('blue', 'x'), ('red', 'circle'), ('red', 'x')]:
        mask = np.logical_and(np.array(colors) == color, np.array(markers) == marker)
        fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1],
            mode='markers',
            marker=dict(color=color, symbol=marker, size=10, line=dict(width=2, color='black')),
            name=f"{'Negative' if color == 'blue' else 'Positive'} {'(Correct)' if marker == 'circle' else '(Incorrect)'}",
            hovertext=[text for text, m in zip(hover_texts, mask) if m],
            hoverinfo='text'
        ), row=1, col=1)

    # Calculate TP, TN, FP, FN
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # Add annotations
    annotations = [
        dict(x=X[:, 0].min(), y=X[:, 1].max(), xref="x", yref="y", text=f"TN: {tn}", showarrow=False),
        dict(x=X[:, 0].max(), y=X[:, 1].max(), xref="x", yref="y", text=f"FP: {fp}", showarrow=False),
        dict(x=X[:, 0].min(), y=X[:, 1].min(), xref="x", yref="y", text=f"FN: {fn}", showarrow=False),
        dict(x=X[:, 0].max(), y=X[:, 1].min(), xref="x", yref="y", text=f"TP: {tp}", showarrow=False)
    ]

    fig.update_layout(
        title='Support Vector Machine Decision Boundary',
        annotations=annotations,
        height=600,
        width=1000
    )

    fig.update_xaxes(title_text="Feature 1", row=1, col=1)
    fig.update_yaxes(title_text="Feature 2", row=1, col=1)

    return fig, tn, fp, fn, tp

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎓 Learn", "📊 SVM Visualization & Training", "🧠 Quiz"])

with tab1:
    st.markdown("<h2 class='tab-subheader'>Learn About SVM</h2>", unsafe_allow_html=True)
    
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

with tab2:
    st.markdown("<h2 class='tab-subheader'>SVM Visualization & Training</h2>", unsafe_allow_html=True)
    
    # Sidebar layout
    st.sidebar.title("Options")
    
    # Dataset selection
    dataset_options = ['Moons', 'Circles', 'Blobs']
    selected_dataset = st.sidebar.selectbox("Select a dataset", dataset_options)
    
    # Number of points slider
    n_samples = st.sidebar.slider("Number of points", min_value=5, max_value=100, value=10, step=5)
    
    # Kernel selection
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    selected_kernel = st.sidebar.selectbox("Select a kernel function", kernel_options)
    
    # Regularization parameter (C)
    c_value = st.sidebar.slider("Regularization parameter (C)", min_value=0.1, max_value=10.0, step=0.1, value=1.0)
    
    # Generate the selected dataset
    if selected_dataset == 'Moons':
        X, y = datasets.make_moons(n_samples=n_samples, noise=0.3, random_state=0)
    elif selected_dataset == 'Circles':
        X, y = datasets.make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=0)
    else:
        X, y = datasets.make_blobs(n_samples=n_samples, centers=2, random_state=0)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train the SVM model
    svc = SVC(kernel=selected_kernel, C=c_value)
    svc.fit(X_scaled, y)
    
    # Make predictions
    y_pred = svc.predict(X_scaled)
    
    # Plot the decision boundary and get confusion matrix values
    fig, tn, fp, fn, tp = plot_svm(X_scaled, y, y_pred, svc)
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Add table with calculations to the right of the plot
    calc_table = go.Table(
        header=dict(values=["Metric", "Formula", "Calculation", "Result"]),
        cells=dict(values=[
            ["Accuracy", "Precision", "Recall", "F1 Score"],
            ["(TP + TN) / (TP + TN + FP + FN)", "TP / (TP + FP)", "TP / (TP + FN)", "2 * (Precision * Recall) / (Precision + Recall)"],
            [f"({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn})", f"{tp} / ({tp} + {fp})", f"{tp} / ({tp} + {fn})", f"2 * ({precision:.2f} * {recall:.2f}) / ({precision:.2f} + {recall:.2f})"],
            [f"{accuracy:.2f}", f"{precision:.2f}", f"{recall:.2f}", f"{f1:.2f}"]
        ])
    )
    fig.add_trace(calc_table, row=1, col=2)
    
    st.plotly_chart(fig)
    
    st.markdown("""
    <p class='content-text'>
    The model's performance can be affected by:
    - The chosen dataset
    - The number of points
    - The selected kernel function
    - The regularization parameter (C)
    
    Try adjusting these parameters in the sidebar to see how they impact the decision boundary and model metrics.
    </p>
    """, unsafe_allow_html=True)


with tab3:
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

