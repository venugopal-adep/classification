import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Circle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

def step(x):
    return 1 if x >= 0 else 0

def step_derivative(x):
    return 0  # The step function is not differentiable, so we use 0 as an approximation

activation_functions = {
    "Sigmoid": (sigmoid, sigmoid_derivative, r"$f(x) = \frac{1}{1 + e^{-x}}$"),
    "ReLU": (relu, relu_derivative, r"$f(x) = \max(0, x)$"),
    "Step": (step, step_derivative, r"$f(x) = \begin{cases} 1 & \text{if } x \geq 0 \\ 0 & \text{otherwise} \end{cases}$")
}

def create_perceptron_diagram(inputs, weights, bias, activation, output, actual, error):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Draw inputs
    for i, (x, w) in enumerate(zip(inputs, weights)):
        ax.add_patch(Circle((1, 9-i*2), 0.5, fc='lightpink'))
        ax.text(0.5, 9-i*2, f'x{i+1}={x:.2f}', ha='right', va='center')
        ax.text(1.5, 9-i*2, f'w{i+1}={w:.2f}', ha='left', va='center')
        ax.arrow(1.5, 9-i*2, 3, -(9-i*2-5), head_width=0.1, head_length=0.1, fc='k', ec='k')

    # Draw summation
    ax.add_patch(Circle((5, 5), 0.7, fc='lightgreen'))
    ax.text(5, 5, '∑', ha='center', va='center', fontsize=20)

    # Draw activation function
    ax.add_patch(Rectangle((6.5, 4.5), 1, 1, fc='lightblue'))
    ax.text(7, 5, 'γ', ha='center', va='center', fontsize=15)

    # Draw output
    ax.add_patch(Ellipse((9, 5), 1.2, 0.7, fc='lightsalmon'))
    ax.text(9, 5, f'Output\n{output:.2f}', ha='center', va='center')

    # Draw arrows
    ax.arrow(5.7, 5, 0.7, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(7.5, 5, 0.9, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')

    # Draw bias
    ax.text(5, 3.5, f'Bias = {bias:.2f}', ha='center', va='center')
    ax.arrow(5, 3.7, 0, 0.7, head_width=0.1, head_length=0.1, fc='k', ec='k')

    # Draw error
    ax.add_patch(Rectangle((7.5, 3.5), 1, 0.7, fc='white', ec='black'))
    ax.text(8, 3.85, f'Error\n{error:.4f}', ha='center', va='center')
    ax.arrow(8, 3.5, 0, -1, head_width=0.1, head_length=0.1, fc='k', ec='k')

    # Add actual value
    ax.text(9, 6.5, f'Actual: {actual:.2f}', ha='center', va='center')

    st.pyplot(fig)

st.set_page_config(page_title="Perceptron Neuron Simulator", layout="wide")

st.title("Perceptron Neuron Simulator")
st.write('**Developed by : Venugopal Adep**')

# Sidebar for input configuration
st.sidebar.header("Neuron Configuration")
num_inputs = st.sidebar.slider("Number of inputs", 1, 5, 3)

inputs = []
weights = []
for i in range(num_inputs):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x = st.number_input(f"Input x{i+1}", value=1.0, step=0.1)
        inputs.append(x)
    with col2:
        w = st.number_input(f"Weight w{i+1}", value=0.5, step=0.1)
        weights.append(w)

bias = st.sidebar.number_input("Bias", value=0.0, step=0.1)
actual_value = st.sidebar.number_input("Actual Value", value=1.0, step=0.1)
learning_rate = st.sidebar.number_input("Learning Rate", value=0.1, step=0.01, min_value=0.01, max_value=1.0)

activation_function = st.sidebar.selectbox("Activation Function", list(activation_functions.keys()))
activation, activation_derivative, activation_formula = activation_functions[activation_function]

# Calculate the output
weighted_sum = np.dot(inputs, weights) + bias
output = activation(weighted_sum)

# Calculate error and cost
error = actual_value - output
cost = 0.5 * (error ** 2)  # Mean Squared Error

# Calculate weight adjustments
weight_adjustments = []
for i, x in enumerate(inputs):
    adjustment = learning_rate * error * activation_derivative(output) * x
    weight_adjustments.append(adjustment)

bias_adjustment = learning_rate * error * activation_derivative(output)

# Display the diagram
create_perceptron_diagram(inputs, weights, bias, activation_function, output, actual_value, error)

# Display calculations and formulas
st.header("Calculations and Formulas")

st.subheader("Activation Function")
if activation_function == "Step":
    st.latex(r"f(x) = \begin{cases} 1 & \text{if } x \geq 0 \\ 0 & \text{otherwise} \end{cases}")
elif activation_function == "ReLU":
    st.latex(r"f(x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}")
elif activation_function == "Sigmoid":
    st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")

st.subheader("Cost Function (Mean Squared Error)")
st.latex(r"C = \frac{1}{2}(y - \hat{y})^2")
st.write(f"Where y = {actual_value:.4f} (actual value) and ŷ = {output:.4f} (predicted value)")

st.subheader("Minimization Function (Gradient Descent)")
st.latex(r"\Delta w_i = -\eta \frac{\partial C}{\partial w_i} = \eta (y - \hat{y}) f'(z) x_i")
st.latex(r"\Delta b = -\eta \frac{\partial C}{\partial b} = \eta (y - \hat{y}) f'(z)")
st.write("Where η is the learning rate, f'(z) is the derivative of the activation function, and x_i are the inputs")

st.subheader("Calculations")
st.write(f"Weighted Sum: {' + '.join([f'({x:.2f} * {w:.2f})' for x, w in zip(inputs, weights)])} + {bias:.2f} (bias) = {weighted_sum:.4f}")
st.write(f"Output (Predicted Value): {output:.4f}")
st.write(f"Actual Value: {actual_value:.4f}")
st.write(f"Error: {error:.4f}")
st.write(f"Cost (MSE): {cost:.4f}")

st.subheader("Weight Adjustments")
for i, (w, adj) in enumerate(zip(weights, weight_adjustments)):
    st.write(f"Weight {i+1}: {w:.4f} + {adj:.4f} = {w + adj:.4f}")
st.write(f"Bias: {bias:.4f} + {bias_adjustment:.4f} = {bias + bias_adjustment:.4f}")

# Explanation
st.header("How it works")
st.write("""
1. The perceptron takes multiple inputs (x1, x2, ..., xn).
2. Each input is multiplied by its corresponding weight (w1, w2, ..., wn).
3. The weighted inputs are summed together, and a bias term is added.
4. The result is passed through an activation function to produce the output.
5. The error is calculated as the difference between the actual value and the predicted output.
6. The cost (mean squared error) is calculated based on this error.
7. The weights and bias are adjusted using gradient descent to minimize the error:
   - Weight adjustment = learning_rate * error * activation_derivative(output) * input
   - Bias adjustment = learning_rate * error * activation_derivative(output)
8. This process would typically be repeated many times to train the perceptron.
""")
