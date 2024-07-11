import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import f
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools.tools import add_constant

# Set page config
st.set_page_config(page_title="Homoscedasticity Explorer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better appearance
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #e6e6e6;
    border-radius: 4px 4px 0 0;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📊 Homoscedasticity Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the impact of homoscedasticity on linear regression!")

# Helper functions
def generate_data(num_samples, homoscedastic=True):
    X = np.random.rand(num_samples, 1)
    if homoscedastic:
        noise = np.random.normal(0, 0.1, num_samples)
    else:
        noise = np.random.normal(0, X.flatten(), num_samples)
    y = 2 * X.flatten() + 1 + noise
    return X, y

def goldfeld_quandt_test(X, y):
    n = len(X)
    k = int(n / 3)
    X_sorted = np.sort(X.flatten())
    X_low, X_high = X_sorted[:k], X_sorted[-k:]
    y_low = y[np.argsort(X.flatten())[:k]]
    y_high = y[np.argsort(X.flatten())[-k:]]
    
    ssr_low = np.sum((y_low - np.mean(y_low))**2)
    ssr_high = np.sum((y_high - np.mean(y_high))**2)
    
    f_value = ssr_high / ssr_low
    p_value = f.sf(f_value, k-1, k-1)
    
    return f_value, p_value

def plot_data(X, y, regression_line, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Data'))
    fig.add_trace(go.Scatter(x=X.flatten(), y=regression_line, mode='lines', name='Regression Line'))
    fig.update_layout(title=title, xaxis_title='X', yaxis_title='y')
    return fig

# Sidebar
st.sidebar.header("Parameters")
num_samples = st.sidebar.slider("Number of Samples", 50, 500, 100, 50)
data_type = st.sidebar.radio("Data Type", ("Homoscedastic", "Heteroscedastic"))

# Generate data
homoscedastic = data_type == "Homoscedastic"
X, y = generate_data(num_samples, homoscedastic)

# Fit linear regression
model = LinearRegression()
model.fit(X, y)
regression_line = model.predict(X)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🧮 Test Results", "🧠 Quiz", "📚 Learn More"])

with tab1:
    st.header("Data Visualization")
    fig = plot_data(X, y, regression_line, data_type + " Data")
    st.plotly_chart(fig, use_container_width=True)

    # Residual plot
    residuals = y - regression_line
    fig_residuals = go.Figure()
    fig_residuals.add_trace(go.Scatter(x=X.flatten(), y=residuals, mode='markers', name='Residuals'))
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
    fig_residuals.update_layout(title="Residual Plot", xaxis_title='X', yaxis_title='Residuals')
    st.plotly_chart(fig_residuals, use_container_width=True)

with tab2:
    st.header("Homoscedasticity Test Results")

    # Goldfeld-Quandt Test
    f_value, p_value = goldfeld_quandt_test(X, y)
    st.subheader("Goldfeld-Quandt Test")
    st.write(f"F-value: {f_value:.2f}")
    st.write(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        st.error("Reject the null hypothesis. The residuals are heteroscedastic.")
    else:
        st.success("Fail to reject the null hypothesis. The residuals are homoscedastic.")

    # Breusch-Pagan Test
    X_with_constant = add_constant(X)  # Add a constant term to X
    bp_test = het_breuschpagan(residuals, X_with_constant)
    st.subheader("Breusch-Pagan Test")
    st.write(f"LM Statistic: {bp_test[0]:.2f}")
    st.write(f"P-value: {bp_test[1]:.4f}")
    
    if bp_test[1] < 0.05:
        st.error("Reject the null hypothesis. The residuals are heteroscedastic.")
    else:
        st.success("Fail to reject the null hypothesis. The residuals are homoscedastic.")

with tab3:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What is homoscedasticity?",
            "options": [
                "Equal variances of residuals across all levels of predictors",
                "Unequal variances of residuals across all levels of predictors",
                "Equal means of residuals across all levels of predictors",
                "Linear relationship between variables"
            ],
            "correct": 0,
            "explanation": "Homoscedasticity refers to the condition where the variance of the residuals is constant across all levels of the independent variables."
        },
        {
            "question": "What is the consequence of heteroscedasticity in linear regression?",
            "options": [
                "Biased coefficient estimates",
                "Inefficient coefficient estimates",
                "Biased standard errors",
                "All of the above"
            ],
            "correct": 2,
            "explanation": "Heteroscedasticity primarily affects the standard errors of the coefficient estimates, making them biased. This can lead to incorrect inferences about the significance of predictors."
        },
        {
            "question": "Which plot is most useful for detecting heteroscedasticity?",
            "options": [
                "Scatter plot of X vs Y",
                "Histogram of residuals",
                "Q-Q plot of residuals",
                "Residual plot (fitted values vs residuals)"
            ],
            "correct": 3,
            "explanation": "A residual plot, which shows the relationship between fitted values and residuals, is most useful for detecting heteroscedasticity. A fan-shaped pattern in this plot often indicates heteroscedasticity."
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}: {q['question']}")
        user_answer = st.radio(f"Select your answer for Question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")
            st.write(f"Explanation: {q['explanation']}")
        st.write("---")

with tab4:
    st.header("Learn More About Homoscedasticity")
    st.markdown("""
    Homoscedasticity is an important assumption in linear regression that refers to the constant variance of the residuals across all levels of the independent variables.

    Key concepts:
    1. **Homoscedasticity**: The variance of residuals is constant across all levels of the independent variables.
    2. **Heteroscedasticity**: The variance of residuals varies across levels of the independent variables.
    3. **Consequences**: Heteroscedasticity can lead to biased standard errors, affecting hypothesis tests and confidence intervals.

    Detecting heteroscedasticity:
    1. **Visual methods**: Residual plots, scale-location plots
    2. **Statistical tests**: Breusch-Pagan test, White test, Goldfeld-Quandt test

    Dealing with heteroscedasticity:
    1. **Transformation**: Log or square root transformation of the dependent variable
    2. **Weighted Least Squares**: Giving less weight to observations with higher variance
    3. **Robust standard errors**: Using methods that are less sensitive to heteroscedasticity

    Remember, while homoscedasticity is an important assumption, minor violations may not severely impact your analysis. It's important to consider the practical significance of any heteroscedasticity detected in your data.
    """)

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates the concept of homoscedasticity in linear regression. Adjust the parameters and explore the different tabs to learn more!")
