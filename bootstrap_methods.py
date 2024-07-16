import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache_data
def load_data():
    housing = fetch_california_housing()
    data = pd.DataFrame(housing.data, columns=housing.feature_names)
    data['PRICE'] = housing.target
    return data

def bootstrap_sample(data, size=None):
    if size is None:
        size = len(data)
    return data.sample(n=size, replace=True)

def main():
    st.set_page_config(page_title="Bootstrap Methods Analysis Demo", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
    .stButton>button {
        background-color: #ff6e40;
        color: white !important;
    }
    .stButton>button:hover {
        background-color: #ff9e80;
        color: white !important;
    }
    .info-box {
        background-color: #e6f3ff;
        border-left: 5px solid #3366cc;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff6e40;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔄 Bootstrap Methods Analysis")
    
    tabs = st.tabs(["📚 Introduction", "🔍 Data Exploration", "📊 Bootstrap Sampling", "📈 Bootstrap Regression", "🧠 Quiz"])
    
    with tabs[0]:
        introduction_section()
    
    with tabs[1]:
        data_exploration_section()
    
    with tabs[2]:
        bootstrap_sampling_section()
    
    with tabs[3]:
        bootstrap_regression_section()
    
    with tabs[4]:
        quiz_section()

def introduction_section():
    st.header("Introduction to Bootstrap Methods")
    
    st.markdown("""
    <div class="info-box">
    Bootstrap methods are resampling techniques used to estimate statistics on a population by sampling a dataset with replacement. 
    They can provide measures of accuracy (bias, variance, confidence intervals, prediction error, etc.) to sample estimates.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Key Concepts")
    concepts = {
        "Resampling": "The process of repeatedly drawing samples from a dataset",
        "Sampling with Replacement": "Each sample is returned to the dataset before drawing the next sample",
        "Bootstrap Sample": "A sample of the data obtained through resampling with replacement",
        "Bootstrap Statistic": "A statistic computed from a bootstrap sample",
        "Confidence Interval": "An interval estimate of a population parameter"
    }

    for concept, description in concepts.items():
        st.markdown(f"**{concept}**: {description}")

    st.subheader("Importance of Bootstrap Methods")
    st.markdown("""
    - Provide estimates of the sampling distribution of almost any statistic
    - Allow estimation of standard errors and confidence intervals for complex estimators
    - Can be applied to small sample sizes where parametric inference is impossible or requires unrealistic assumptions
    - Useful for hypothesis testing and validation of models
    - Enable more accurate inferences when the data are not well-behaved or when the sample size is small
    """)

def data_exploration_section():
    st.header("Data Exploration")

    data = load_data()

    st.subheader("California Housing Dataset Overview")
    st.write(data.head())
    st.write(f"Dataset shape: {data.shape}")

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.markdown("""
    The California Housing dataset contains information from the 1990 California census. Here's a brief description of the features:

    1. MedInc: Median income in block group
    2. HouseAge: Median house age in block group
    3. AveRooms: Average number of rooms per household
    4. AveBedrms: Average number of bedrooms per household
    5. Population: Block group population
    6. AveOccup: Average number of household members
    7. Latitude: Block group latitude
    8. Longitude: Block group longitude

    The target variable is PRICE: Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000).
    """)

def bootstrap_sampling_section():
    st.header("Bootstrap Sampling")

    data = load_data()

    st.markdown("""
    <div class="info-box">
    Bootstrap sampling involves creating multiple resamples of your dataset by sampling with replacement. 
    This allows us to estimate the sampling distribution of a statistic and calculate its standard error.
    </div>
    """, unsafe_allow_html=True)

    # Select a feature for demonstration
    feature = st.selectbox("Select a feature for bootstrap sampling", data.columns[:-1])

    # Number of bootstrap samples
    n_bootstraps = st.slider("Number of bootstrap samples", 100, 10000, 1000)

    if st.button("Run Bootstrap Sampling"):
        bootstrap_means = [bootstrap_sample(data[feature]).mean() for _ in range(n_bootstraps)]

        # Original sample statistics
        original_mean = data[feature].mean()
        original_std = data[feature].std()

        # Bootstrap statistics
        bootstrap_mean = np.mean(bootstrap_means)
        bootstrap_std = np.std(bootstrap_means)

        # Confidence Interval
        ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])

        st.subheader("Bootstrap Results")
        st.write(f"Original Sample Mean: {original_mean:.4f}")
        st.write(f"Original Sample Std: {original_std:.4f}")
        st.write(f"Bootstrap Mean: {bootstrap_mean:.4f}")
        st.write(f"Bootstrap Std: {bootstrap_std:.4f}")
        st.write(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")

        # Visualize results
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=bootstrap_means, name="Bootstrap Distribution"))
        fig.add_vline(x=original_mean, line_dash="dash", line_color="red", annotation_text="Original Mean")
        fig.add_vline(x=ci_lower, line_dash="dash", line_color="green", annotation_text="2.5th Percentile")
        fig.add_vline(x=ci_upper, line_dash="dash", line_color="green", annotation_text="97.5th Percentile")
        fig.update_layout(title=f"Bootstrap Distribution of Mean {feature}", 
                          xaxis_title=f"Mean {feature}", 
                          yaxis_title="Frequency")
        st.plotly_chart(fig)

        st.markdown("""
        The histogram shows the distribution of bootstrap sample means. The red dashed line represents the original sample mean, 
        and the green dashed lines represent the 95% confidence interval.
        """)

def bootstrap_regression_section():
    st.header("Bootstrap Regression")

    data = load_data()

    st.markdown("""
    <div class="info-box">
    Bootstrap regression involves applying the bootstrap method to regression analysis. 
    It can be used to estimate the uncertainty of regression coefficients and make inferences about the model.
    </div>
    """, unsafe_allow_html=True)

    # Select features for regression
    features = st.multiselect("Select features for regression", data.columns[:-1], default=data.columns[0])

    # Number of bootstrap samples
    n_bootstraps = st.slider("Number of bootstrap samples", 100, 10000, 1000, key="reg_bootstraps")

    if st.button("Run Bootstrap Regression"):
        X = data[features]
        y = data['PRICE']

        # Original regression
        model = LinearRegression()
        model.fit(X, y)
        original_coef = model.coef_
        original_intercept = model.intercept_
        original_r2 = r2_score(y, model.predict(X))

        # Bootstrap regression
        bootstrap_coefs = []
        bootstrap_intercepts = []
        bootstrap_r2 = []

        for _ in range(n_bootstraps):
            boot_sample = bootstrap_sample(data)
            X_boot = boot_sample[features]
            y_boot = boot_sample['PRICE']
            
            boot_model = LinearRegression()
            boot_model.fit(X_boot, y_boot)
            
            bootstrap_coefs.append(boot_model.coef_)
            bootstrap_intercepts.append(boot_model.intercept_)
            bootstrap_r2.append(r2_score(y_boot, boot_model.predict(X_boot)))

        # Results
        st.subheader("Bootstrap Regression Results")
        st.write("Original Regression Coefficients:")
        for feature, coef in zip(features, original_coef):
            st.write(f"{feature}: {coef:.4f}")
        st.write(f"Original Intercept: {original_intercept:.4f}")
        st.write(f"Original R-squared: {original_r2:.4f}")

        st.write("\nBootstrap Statistics:")
        for i, feature in enumerate(features):
            coef_mean = np.mean([coef[i] for coef in bootstrap_coefs])
            coef_std = np.std([coef[i] for coef in bootstrap_coefs])
            ci_lower, ci_upper = np.percentile([coef[i] for coef in bootstrap_coefs], [2.5, 97.5])
            st.write(f"{feature}:")
            st.write(f"  Mean: {coef_mean:.4f}")
            st.write(f"  Std: {coef_std:.4f}")
            st.write(f"  95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")

        intercept_mean = np.mean(bootstrap_intercepts)
        intercept_std = np.std(bootstrap_intercepts)
        intercept_ci_lower, intercept_ci_upper = np.percentile(bootstrap_intercepts, [2.5, 97.5])
        st.write("Intercept:")
        st.write(f"  Mean: {intercept_mean:.4f}")
        st.write(f"  Std: {intercept_std:.4f}")
        st.write(f"  95% CI: ({intercept_ci_lower:.4f}, {intercept_ci_upper:.4f})")

        r2_mean = np.mean(bootstrap_r2)
        r2_std = np.std(bootstrap_r2)
        r2_ci_lower, r2_ci_upper = np.percentile(bootstrap_r2, [2.5, 97.5])
        st.write("R-squared:")
        st.write(f"  Mean: {r2_mean:.4f}")
        st.write(f"  Std: {r2_std:.4f}")
        st.write(f"  95% CI: ({r2_ci_lower:.4f}, {r2_ci_upper:.4f})")

        # Visualize coefficient distributions
        fig = make_subplots(rows=len(features), cols=1, subplot_titles=features)
        for i, feature in enumerate(features, 1):
            fig.add_trace(go.Histogram(x=[coef[i-1] for coef in bootstrap_coefs], name=feature), row=i, col=1)
            fig.add_vline(x=original_coef[i-1], line_dash="dash", line_color="red", row=i, col=1)
        fig.update_layout(height=300*len(features), title_text="Bootstrap Distributions of Regression Coefficients")
        st.plotly_chart(fig)

        st.markdown("""
        The histograms show the distributions of bootstrap regression coefficients for each feature. 
        The red dashed lines represent the coefficients from the original regression.
        """)

def quiz_section():
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main purpose of bootstrap methods?",
            "options": [
                "To increase the size of the dataset",
                "To estimate statistics and their uncertainties using resampling",
                "To remove outliers from the data",
                "To speed up computation of complex models"
            ],
            "correct": "To estimate statistics and their uncertainties using resampling",
            "explanation": "Bootstrap methods use resampling techniques to estimate statistics on a population and provide measures of accuracy for these estimates."
        },
        {
            "question": "What does 'sampling with replacement' mean in the context of bootstrap?",
            "options": [
                "Replacing missing values in the dataset",
                "Each sample is returned to the dataset before drawing the next sample",
                "Replacing outliers with mean values",
                "Sampling without duplicating any data points"
            ],
            "correct": "Each sample is returned to the dataset before drawing the next sample",
            "explanation": "In bootstrap sampling, 'sampling with replacement' means that after a data point is selected for the bootstrap sample, it is put back into the original dataset, allowing it to be potentially selected again."
        },
        {
            "question": "What is a bootstrap confidence interval?",
            "options": [
                "The range of the original dataset",
                "An interval estimate of a population parameter based on bootstrap samples",
                "The difference between the largest and smallest bootstrap sample",
                "The time it takes to run a bootstrap analysis"
            ],
            "correct": "An interval estimate of a population parameter based on bootstrap samples",
            "explanation": "A bootstrap confidence interval is an interval estimate of a population parameter that is derived from bootstrap samples. It provides a range of plausible values for the parameter, taking into account the uncertainty in the estimate."
        },
        {
            "question": "How can bootstrap methods be applied to regression analysis?",
            "options": [
                "By randomly removing data points from the regression",
                "By resampling the data and refitting the regression model multiple times",
                "By changing the regression algorithm for each iteration",
                "By increasing the number of predictors in the model"
            ],
            "correct": "By resampling the data and refitting the regression model multiple times",
            "explanation": "In bootstrap regression, we create multiple bootstrap samples of the data, fit a regression model to each sample, and analyze the distribution of the resulting model parameters. This helps estimate the uncertainty in the regression coefficients and other model statistics."
        }
    ]
    
    for i, q in enumerate(questions, 1):
        st.subheader(f"Question {i}")
        with st.container():
            st.write(q["question"])
            answer = st.radio("Select your answer:", q["options"], key=f"q{i}")
            if st.button("Check Answer", key=f"check{i}"):
                if answer == q["correct"]:
                    st.success("Correct! 🎉")
                else:
                    st.error(f"Incorrect. The correct answer is: {q['correct']}")
                st.info(f"Explanation: {q['explanation']}")
            st.write("---")

if __name__ == "__main__":
    main()