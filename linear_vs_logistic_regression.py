import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title='Linear vs Logistic Regression', layout='wide')

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
    st.plotly_chart(fig)

st.title('Exploring Linear and Logistic Regression')
st.write('''
This interactive demo lets you experiment with linear and logistic regression to see how they differ in fitting data.

**Linear Regression** is used when the dependent variable is continuous. It fits a straight line to model the relationship between the independent and dependent variables. The line is chosen to minimize the sum of squared errors between the predicted and actual values.

**Example:** Predicting a student's test score based on the number of hours they studied. The test score is a continuous value that could be any number within a range.

**Logistic Regression** is used when the dependent variable is categorical, typically binary (0 or 1). It fits a sigmoidal curve to model the probability that the dependent variable equals 1, given the independent variable. The curve is chosen to maximize the likelihood of the observed data.

**Example:** Predicting whether a customer will buy a product or not based on their age. The outcome is binary - either they buy (1) or they don't (0).

Use the sliders in the sidebar to control the properties of the generated data. The plots will update in real-time to show how the regression lines fit the data.
''')

st.sidebar.header('Parameters')
num_points = st.sidebar.slider('Number of data points', 10, 100, 30)
noise_level = st.sidebar.slider('Noise level', 0.0, 2.0, 0.5)
logistic_x_shift = st.sidebar.slider('Logistic curve shift', -5.0, 5.0, 0.0)
threshold = st.sidebar.slider('Decision Threshold', 0.0, 1.0, 0.5)

x, y_linear, y_logistic = generate_data(num_points, noise_level, logistic_x_shift)

col1, col2 = st.columns(2)

with col1:
    st.header("Linear Regression")
    p = np.polyfit(x, y_linear, 1)
    y_linear_pred = p[0]*x + p[1]
    plot_data(x, y_linear, 'markers', 'Linear Regression', 'blue', y_linear_pred, 'red')
    st.write(f'Slope: {p[0]:.2f}, Intercept: {p[1]:.2f}')
    st.write('The straight line tries to capture the trend in the data. The slope represents the change in the dependent variable for a one unit change in the independent variable. The intercept is the predicted value when the independent variable is zero.')

with col2:
    st.header("Logistic Regression")
    model = LogisticRegression(random_state=0).fit(x.reshape(-1, 1), y_logistic)
    x_logistic_pred = np.linspace(x.min(), x.max(), 100)
    y_logistic_pred = model.predict_proba(x_logistic_pred.reshape(-1, 1))[:,1]
    plot_data(x_logistic_pred, y_logistic_pred, 'lines', 'Logistic Regression', 'orange', threshold=threshold)
    plot_data(x, y_logistic, 'markers', '', 'green')
    st.write('The S-shaped curve models the probability the dependent variable equals 1. As the independent variable increases, the probability approaches 1. As it decreases, the probability approaches 0. The curve is steepest where the probability is closest to 0.5.')
    st.write(f'The decision threshold is set at {threshold:.2f}. Points above this threshold are classified as 1, and points below are classified as 0.')