import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
    background-color: #f0f8ff;
}
.stButton>button {
    background-color: #4b0082;
    color: white;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #e6e6fa;
    border-radius: 4px 4px 0 0;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #8a2be2;
    color: white;
}
.highlight {
    background-color: #e6e6fa;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📊 Homoscedasticity Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the impact of homoscedasticity on linear regression!")

# Helper functions
@st.cache_data
def generate_data(num_samples, homoscedastic=True):
    X = np.random.rand(num_samples, 1)
    if homoscedastic:
        noise = np.random.normal(0, 0.1, num_samples)
    else:
        noise = np.random.normal(0, X.flatten() * 0.5, num_samples)
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
    fig = px.scatter(x=X.flatten(), y=y, labels={'x': 'X', 'y': 'y'}, title=title)
    fig.add_trace(go.Scatter(x=X.flatten(), y=regression_line, mode='lines', name='Regression Line'))
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
tab1, tab2, tab3, tab4 = st.tabs(["📚 Learn", "📊 Visualization", "🧮 Test Results", "🧠 Quiz"])

with tab1:
    st.header("📚 Learn About Homoscedasticity")
    
    st.markdown("""
    <div class="highlight">
    <h3>What is Homoscedasticity?</h3>
    <p>Homoscedasticity is a key assumption in linear regression. It means that the variance of the residuals (the differences between observed and predicted values) is constant across all levels of the independent variables.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>Why is Homoscedasticity Important?</h3>
    <ul>
        <li>Ensures the reliability of regression estimates</li>
        <li>Allows for accurate hypothesis testing</li>
        <li>Helps in creating valid confidence intervals</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>What Happens When Data is Heteroscedastic?</h3>
    <ul>
        <li>Standard errors of coefficients may be biased</li>
        <li>Hypothesis tests may be unreliable</li>
        <li>Confidence intervals may be too wide or too narrow</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("📊 Data Visualization")
    
    fig = plot_data(X, y, regression_line, data_type + " Data")
    st.plotly_chart(fig, use_container_width=True)

    # Residual plot
    residuals = y - regression_line
    fig_residuals = px.scatter(x=X.flatten(), y=residuals, labels={'x': 'X', 'y': 'Residuals'}, title="Residual Plot")
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_residuals, use_container_width=True)
    
    st.markdown("""
    <div class="highlight">
    <p>In the residual plot, look for patterns. A random scatter indicates homoscedasticity, while a fan or funnel shape suggests heteroscedasticity.</p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.header("🧮 Homoscedasticity Test Results")

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

    st.markdown("""
    <div class="highlight">
    <p>Both tests check for heteroscedasticity. A p-value less than 0.05 suggests the presence of heteroscedasticity.</p>
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "What does homoscedasticity mean?",
            "options": [
                "The variance of residuals is constant",
                "The mean of residuals is constant",
                "The data is normally distributed",
                "The relationship between variables is linear"
            ],
            "correct": 0,
            "explanation": "Homoscedasticity means that the variance of the residuals is constant across all levels of the independent variables."
        },
        {
            "question": "Which plot is most useful for detecting heteroscedasticity?",
            "options": [
                "Scatter plot of X vs Y",
                "Histogram of Y",
                "Residual plot",
                "Box plot of X"
            ],
            "correct": 2,
            "explanation": "A residual plot, which shows predicted values vs residuals, is most useful for detecting heteroscedasticity. A fan or funnel shape in this plot often indicates heteroscedasticity."
        },
        {
            "question": "What should you look for in a residual plot to indicate homoscedasticity?",
            "options": [
                "A clear pattern",
                "A random scatter of points",
                "A straight line",
                "A normal distribution curve"
            ],
            "correct": 1,
            "explanation": "In a residual plot, a random scatter of points with no clear pattern indicates homoscedasticity."
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}: {q['question']}")
        user_answer = st.radio(f"Select your answer for Question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! Well done!")
            else:
                st.error("Not quite right. Let's learn from this!")
            st.info(f"Explanation: {q['explanation']}")
        st.write("---")

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates the concept of homoscedasticity in linear regression. Adjust the parameters and explore the different tabs to learn more!")
