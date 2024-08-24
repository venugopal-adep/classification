import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px
import math

# Set page configuration
st.set_page_config(page_title="Perceptron Explorer", layout="wide")

# Custom CSS for visual appeal
st.markdown("""
<style>
    .main {
        background-color: #f0f8ff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stTextInput>div>div>input {
        background-color: #e0e0e0;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .stTab {
        background-color: #f1f8ff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🧠 Perceptron Explorer")

st.markdown("""
Welcome to the Perceptron Explorer! Dive into the world of the simplest neural network
and discover how it performs binary classification tasks.
""")

# Functions
@st.cache_data
def generate_data(n_samples=100, noise=0.1):
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X += np.random.randn(n_samples, 2) * noise
    return X, y

def update_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Perceptron().fit(X_train, y_train)

    df = pd.DataFrame(X_train, columns=['x1', 'x2'])
    df['y'] = y_train

    fig = px.scatter(df, x='x1', y='x2', color='y', 
                     labels={"x1": "Feature 1", "x2": "Feature 2", "y": "Class"},
                     title="Perceptron Classification")

    x_range = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
    y_range = -(model.coef_[0][0] * x_range + model.intercept_[0]) / model.coef_[0][1]
    
    fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='Decision Boundary'))
    fig.update_layout(template="plotly_white")

    return fig, model, X_test, y_test

# Generate initial data
X, y = generate_data()
fig, model, X_test, y_test = update_model(X, y)

# Sidebar
st.sidebar.header("🎛️ Control Panel")
if st.sidebar.button("🔄 Generate New Data", key="generate_new_data"):
    X, y = generate_data()
    fig, model, X_test, y_test = update_model(X, y)

# Main content using tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧠 Learn", "🔬 Explore", "🏋️ Train", "🧪 Quiz", "🔢 Perceptron Calculator"])

with tab1:
    st.header("What is a Perceptron?")
    st.markdown("""
    A perceptron is the simplest type of artificial neural network. It's a linear classification algorithm that separates two classes using a line. Key points:

    - It's the building block of more complex neural networks
    - Suitable for linearly separable data
    - Uses the gradient descent algorithm for training
    - Can handle multiple input features
    """)

    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Perceptron_example.svg/1200px-Perceptron_example.svg.png", caption="Perceptron Structure")

with tab2:
    st.header("🔍 Perceptron Explorer")
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Parameters")
        st.write(f"Weights: {model.coef_[0]}")
        st.write(f"Bias: {model.intercept_[0]}")
    
    with col2:
        st.subheader("Model Performance")
        accuracy = model.score(X_test, y_test)
        st.write(f"Accuracy on test set: {accuracy:.2f}")

with tab3:
    st.header("🏋️ Perceptron Trainer")
    st.markdown("Adjust the parameters to see how they affect the perceptron's performance!")
    
    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("Number of Samples", 50, 500, 100, 10)
        noise = st.slider("Noise Level", 0.0, 1.0, 0.1, 0.05)
    
    with col2:
        max_iter = st.slider("Maximum Iterations", 100, 1000, 200, 50)
        tol = st.slider("Tolerance", 0.0001, 0.1, 0.001, 0.0001, format="%.4f")
    
    if st.button("Train Perceptron", key="train_perceptron"):
        X, y = generate_data(n_samples, noise)
        model = Perceptron(max_iter=max_iter, tol=tol)
        fig, model, X_test, y_test = update_model(X, y)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Parameters")
            st.write(f"Weights: {model.coef_[0]}")
            st.write(f"Bias: {model.intercept_[0]}")
        
        with col2:
            st.subheader("Model Performance")
            accuracy = model.score(X_test, y_test)
            st.write(f"Accuracy on test set: {accuracy:.2f}")

with tab4:
    st.header("🧪 Perceptron Quiz")
    
    questions = [
        {
            "question": "What type of classification does a perceptron perform?",
            "options": ["Linear", "Non-linear", "Multi-class", "Unsupervised"],
            "answer": 0,
            "explanation": "A perceptron performs linear classification, meaning it separates classes using a straight line (or hyperplane in higher dimensions)."
        },
        {
            "question": "Which algorithm is used to train the perceptron?",
            "options": ["K-means", "Gradient Descent", "Random Forest", "Support Vector Machines"],
            "answer": 1,
            "explanation": "The perceptron is trained using the gradient descent algorithm, which iteratively adjusts the weights to minimize the error."
        },
        {
            "question": "What is a limitation of the perceptron?",
            "options": ["It can only handle binary classification", "It requires a large amount of data", "It can't handle linearly separable data", "It can solve the XOR problem"],
            "answer": 0,
            "explanation": "A basic perceptron can only handle binary classification tasks. For multi-class problems, more complex neural networks are needed."
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}")
        st.write(q["question"])
        user_answer = st.radio(f"Select your answer for question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button("Check Answer", key=f"check_answer_{i}"):
            if user_answer == q["options"][q["answer"]]:
                st.success("Correct!")
            else:
                st.error(f"Incorrect. {q['explanation']}")

# Add this new section within the tabs
with tab5:
    st.header("🔢 Single Perceptron Calculator")
    st.markdown("""
    This interactive calculator demonstrates how a single perceptron processes inputs to produce an output.
    Adjust the weights, inputs, and bias to see how the perceptron behaves.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Inputs and Weights")
        num_inputs = st.slider("Number of inputs", 1, 5, 3, key="num_inputs")
        
        inputs = []
        weights = []
        for i in range(num_inputs):
            input_val = st.number_input(f"Input x{i+1}", value=1.0, step=0.1, key=f"input_{i}_tab5")
            weight = st.number_input(f"Weight w{i+1}", value=0.5, step=0.1, key=f"weight_{i}_tab5")
            inputs.append(input_val)
            weights.append(weight)
        
        bias = st.number_input("Bias", value=0.0, step=0.1, key="bias_tab5")

    with col2:
        st.subheader("Activation Function")
        activation_function = st.selectbox("Select Activation Function", 
                                           ["Step", "Sigmoid", "ReLU"], key="activation_function_tab5")

        def step(x):
            return 1 if x >= 0 else 0

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        def relu(x):
            return max(0, x)

        activation_functions = {
            "Step": step,
            "Sigmoid": sigmoid,
            "ReLU": relu
        }

        if st.button("Calculate Output", key="calculate_perceptron_output_tab5"):

            # Compute weighted sum
            weighted_sum = sum([i * w for i, w in zip(inputs, weights)]) + bias
            
            # Apply activation function
            output = activation_functions[activation_function](weighted_sum)

            st.subheader("Calculation Steps")
            st.markdown("1. Weighted Sum: $\sum{(input_i \times weight_i)} + bias$")
            
            # Create the LaTeX expression for the weighted sum
            latex_parts = [f"({i:.2f} \\times {w:.2f})" for i, w in zip(inputs, weights)]
            latex_expr = " + ".join(latex_parts)
            latex_full = f"{latex_expr} + {bias:.2f} = {weighted_sum:.4f}"
            
            # Display the LaTeX expression
            st.latex(latex_full)
            
            st.markdown(f"2. Apply {activation_function} Activation Function")
            if activation_function == "Step":
                st.latex(r"output = \begin{cases} 1 & \text{if weighted sum} \geq 0 \\ 0 & \text{otherwise} \end{cases}")
            elif activation_function == "Sigmoid":
                st.latex(r"output = \frac{1}{1 + e^{-" + f"{weighted_sum:.4f}" + r"}} = " + f"{output:.4f}")
            elif activation_function == "ReLU":
                st.latex(r"output = max(0, " + f"{weighted_sum:.4f}" + r") = " + f"{output:.4f}")

            st.subheader("Final Output")
            st.success(f"The perceptron output is: {output:.4f}")

    st.markdown("""
    ### How it works:
    1. The perceptron takes multiple inputs (x1, x2, ..., xn).
    2. Each input is multiplied by its corresponding weight (w1, w2, ..., wn).
    3. The weighted inputs are summed together, and a bias term is added.
    4. The result is passed through an activation function to produce the final output.
    5. This output can be interpreted as the perceptron's decision or prediction.
    """)

st.markdown("""
## 🎓 Conclusion

Congratulations on exploring the world of Perceptrons! Remember:

- 🧠 Perceptrons are the simplest form of neural networks.
- 📊 They perform linear classification, separating two classes with a line.
- 🔢 The model learns by adjusting weights and bias using gradient descent.
- 📈 Perceptrons work best with linearly separable data.

Keep exploring and learning about more advanced neural network architectures! 🚀
""")
