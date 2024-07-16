import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache_data
def load_data():
    iris = load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])
    data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return data

def main():
    st.set_page_config(page_title="Multivariate Statistics Demo", layout="wide")
    
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
    
    st.title("📊 Multivariate Statistics")
    
    tabs = st.tabs(["📚 Introduction", "🔍 Data Exploration", "📈 Correlation Analysis", "🧮 Principal Component Analysis", "🧠 Quiz"])
    
    with tabs[0]:
        introduction_section()
    
    with tabs[1]:
        data_exploration_section()
    
    with tabs[2]:
        correlation_analysis_section()
    
    with tabs[3]:
        pca_section()
    
    with tabs[4]:
        quiz_section()

def introduction_section():
    st.header("Introduction to Multivariate Statistics")
    
    st.markdown("""
    <div class="info-box">
    Multivariate statistics involves observation and analysis of more than one statistical outcome variable at a time. 
    In design and analysis, the technique is used to perform trade studies across multiple dimensions while taking into 
    account the effects of all variables on the responses of interest.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Key Concepts")
    concepts = {
        "Multivariate Data": "Data that includes multiple variables for each observation",
        "Correlation": "A measure of the linear relationship between two variables",
        "Covariance": "A measure of the joint variability of two variables",
        "Principal Component Analysis (PCA)": "A technique for reducing the dimensionality of multivariate data",
        "Factor Analysis": "A method for describing variability among observed variables in terms of fewer unobserved variables",
        "MANOVA": "Multivariate Analysis of Variance, an extension of ANOVA to multiple dependent variables"
    }

    for concept, description in concepts.items():
        st.markdown(f"**{concept}**: {description}")

    st.subheader("Importance of Multivariate Statistics")
    st.markdown("""
    - Allows for the analysis of complex, multi-dimensional data
    - Helps in understanding relationships between multiple variables simultaneously
    - Useful for data reduction and simplification
    - Enables the discovery of latent patterns in data
    - Applicable in various fields such as biology, psychology, economics, and more
    """)

def data_exploration_section():
    st.header("Data Exploration")

    data = load_data()

    st.subheader("Iris Dataset Overview")
    st.write(data.head())
    st.write(f"Dataset shape: {data.shape}")

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Pairplot
    st.subheader("Pairplot of Iris Features")
    fig = sns.pairplot(data, hue='species')
    st.pyplot(fig)

    st.markdown("""
    The Iris dataset contains 150 samples with 4 features each:
    1. Sepal length
    2. Sepal width
    3. Petal length
    4. Petal width
    
    There are three species of Iris in the dataset:
    - Setosa
    - Versicolor
    - Virginica
    
    The pairplot above shows the relationships between all pairs of features, colored by species.
    """)

def correlation_analysis_section():
    st.header("Correlation Analysis")

    data = load_data()

    st.markdown("""
    <div class="info-box">
    Correlation analysis measures the strength and direction of the linear relationship between two variables. 
    It's a key technique in multivariate statistics for understanding how variables are related to each other.
    </div>
    """, unsafe_allow_html=True)

    # Correlation matrix
    corr_matrix = data.drop(['target', 'species'], axis=1).corr()

    # Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.markdown("""
    The heatmap above shows the correlation coefficients between all pairs of features. 
    - Values close to 1 indicate strong positive correlation
    - Values close to -1 indicate strong negative correlation
    - Values close to 0 indicate weak or no linear correlation
    """)

    # Scatter plot with correlation
    st.subheader("Scatter Plot with Correlation")
    x_var = st.selectbox("Select X variable", data.columns[:-2])
    y_var = st.selectbox("Select Y variable", [col for col in data.columns[:-2] if col != x_var])

    fig = px.scatter(data, x=x_var, y=y_var, color='species', trendline='ols')
    st.plotly_chart(fig)

    correlation = data[x_var].corr(data[y_var])
    st.write(f"Correlation between {x_var} and {y_var}: {correlation:.2f}")

def pca_section():
    st.header("Principal Component Analysis (PCA)")

    data = load_data()

    st.markdown("""
    <div class="info-box">
    Principal Component Analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert 
    a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables 
    called principal components.
    </div>
    """, unsafe_allow_html=True)

    # Perform PCA
    X = data.drop(['target', 'species'], axis=1)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)

    # Scree plot
    st.subheader("Scree Plot")
    explained_variance_ratio = pca.explained_variance_ratio_
    cum_explained_variance_ratio = np.cumsum(explained_variance_ratio)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, len(explained_variance_ratio)+1)), y=explained_variance_ratio, name='Individual'))
    fig.add_trace(go.Scatter(x=list(range(1, len(cum_explained_variance_ratio)+1)), y=cum_explained_variance_ratio, name='Cumulative'))
    fig.update_layout(title='Scree Plot', xaxis_title='Principal Component', yaxis_title='Explained Variance Ratio')
    st.plotly_chart(fig)

    st.markdown("""
    The Scree plot shows the explained variance ratio of each principal component. 
    It helps in determining how many principal components to retain.
    """)

    # PCA visualization
    st.subheader("PCA Visualization")
    fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1], color=data['species'],
                     labels={'x': 'First Principal Component', 'y': 'Second Principal Component'})
    st.plotly_chart(fig)

    st.markdown("""
    This plot shows the data points projected onto the first two principal components. 
    It can reveal clusters or patterns in the data that might not be visible in the original feature space.
    """)

def quiz_section():
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main purpose of Principal Component Analysis (PCA)?",
            "options": [
                "To classify data points",
                "To reduce dimensionality of data",
                "To calculate correlations between variables",
                "To perform hypothesis testing"
            ],
            "correct": "To reduce dimensionality of data",
            "explanation": "PCA is primarily used for dimensionality reduction. It transforms the data into a new coordinate system where the new variables (principal components) are linear combinations of the original variables and are uncorrelated with each other."
        },
        {
            "question": "What does a correlation coefficient of -0.8 between two variables indicate?",
            "options": [
                "A strong positive linear relationship",
                "A weak negative linear relationship",
                "A strong negative linear relationship",
                "No linear relationship"
            ],
            "correct": "A strong negative linear relationship",
            "explanation": "A correlation coefficient of -0.8 indicates a strong negative linear relationship between the two variables. As one variable increases, the other tends to decrease."
        },
        {
            "question": "In a scree plot for PCA, what does the y-axis typically represent?",
            "options": [
                "The number of variables",
                "The explained variance ratio",
                "The correlation coefficient",
                "The number of data points"
            ],
            "correct": "The explained variance ratio",
            "explanation": "In a scree plot for PCA, the y-axis typically represents the explained variance ratio. This shows how much of the total variance in the data is explained by each principal component."
        },
        {
            "question": "What is multivariate data?",
            "options": [
                "Data with only one variable",
                "Data with two variables",
                "Data with multiple variables for each observation",
                "Data with only categorical variables"
            ],
            "correct": "Data with multiple variables for each observation",
            "explanation": "Multivariate data refers to data that includes multiple variables for each observation. This allows for the analysis of relationships and patterns among multiple variables simultaneously."
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