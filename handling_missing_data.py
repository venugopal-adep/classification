import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objects as go

def main():
    st.set_page_config(page_title="Handling Missing Data in Classification", layout="wide")
    
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
        color: white;
    }
    .stButton>button:hover {
        background-color: #ff9e80;
    }
    .quiz-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🧩 Handling Missing Data in Classification")
    
    tabs = st.tabs(["📚 Learn", "🔍 Explore", "📊 Compare Methods", "🧠 Quiz"])
    
    with tabs[0]:
        learn_section()
    
    with tabs[1]:
        explore_section()
    
    with tabs[2]:
        compare_methods_section()
    
    with tabs[3]:
        quiz_section()

def learn_section():
    st.header("Understanding Missing Data")
    
    st.write("""
    Missing data is a common problem in real-world datasets. It can occur due to various reasons such as:
    - Data entry errors
    - Equipment malfunctions
    - Respondents skipping questions in surveys
    - Merging data from different sources
    
    Handling missing data properly is crucial for building accurate and reliable machine learning models.
    """)
    
    st.subheader("Types of Missing Data")
    st.write("""
    1. **Missing Completely at Random (MCAR)**: The probability of missing data is the same for all observations.
    2. **Missing at Random (MAR)**: The probability of missing data depends on other observed variables.
    3. **Missing Not at Random (MNAR)**: The probability of missing data depends on unobserved variables.
    """)
    
    st.subheader("Common Strategies for Handling Missing Data")
    st.write("""
    1. **Deletion Methods**:
       - Listwise deletion (complete case analysis)
       - Pairwise deletion
    
    2. **Imputation Methods**:
       - Mean/Median/Mode imputation
       - Regression imputation
       - Multiple imputation
    
    3. **Advanced Methods**:
       - K-Nearest Neighbors (KNN) imputation
       - Machine learning-based imputation (e.g., using Random Forests)
    
    4. **Using algorithms that handle missing values**:
       - Decision trees
       - Random Forests
    """)

def explore_section():
    st.header("🔍 Explore Missing Data Strategies")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    age = np.random.normal(40, 10, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    education_years = np.random.normal(16, 3, n_samples)
    
    # Introduce missing values
    missing_rate = st.slider("Missing Data Rate", 0.0, 0.5, 0.2, 0.05)
    age[np.random.choice(n_samples, int(n_samples * missing_rate), replace=False)] = np.nan
    income[np.random.choice(n_samples, int(n_samples * missing_rate), replace=False)] = np.nan
    education_years[np.random.choice(n_samples, int(n_samples * missing_rate), replace=False)] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Education_Years': education_years
    })
    
    st.subheader("Sample Data with Missing Values")
    st.write(df.head(10))
    
    st.subheader("Missing Data Visualization")
    fig = go.Figure()
    for column in df.columns:
        fig.add_trace(go.Bar(
            y=[column],
            x=[df[column].isnull().sum() / len(df) * 100],
            orientation='h',
            name=column
        ))
    fig.update_layout(
        title='Percentage of Missing Values by Feature',
        xaxis_title='Percentage of Missing Values',
        yaxis_title='Features',
        barmode='group'
    )
    st.plotly_chart(fig)
    
    st.subheader("Handling Strategies")
    strategy = st.selectbox("Select a strategy", 
                            ["Mean Imputation", "Median Imputation", "Most Frequent Imputation", "Listwise Deletion"])
    
    if strategy == "Listwise Deletion":
        df_clean = df.dropna()
    else:
        imputer_strategy = 'mean' if strategy == "Mean Imputation" else 'median' if strategy == "Median Imputation" else 'most_frequent'
        imputer = SimpleImputer(strategy=imputer_strategy)
        df_clean = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    st.write("Data after handling missing values:")
    st.write(df_clean.head(10))
    
    st.write(f"Original data shape: {df.shape}")
    st.write(f"Cleaned data shape: {df_clean.shape}")

def compare_methods_section():
    st.header("📊 Compare Missing Data Handling Methods")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    age = np.random.normal(40, 10, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    education_years = np.random.normal(16, 3, n_samples)
    target = (age > 40).astype(int)  # Binary classification target
    
    # Introduce missing values
    missing_rate = 0.2
    age[np.random.choice(n_samples, int(n_samples * missing_rate), replace=False)] = np.nan
    income[np.random.choice(n_samples, int(n_samples * missing_rate), replace=False)] = np.nan
    education_years[np.random.choice(n_samples, int(n_samples * missing_rate), replace=False)] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Education_Years': education_years,
        'Target': target
    })
    
    # Split data
    X = df.drop('Target', axis=1)
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    strategies = ["Mean Imputation", "Median Imputation", "Most Frequent Imputation", "Listwise Deletion"]
    results = []
    
    for strategy in strategies:
        if strategy == "Listwise Deletion":
            X_train_clean = X_train.dropna()
            y_train_clean = y_train[X_train_clean.index]
            X_test_clean = X_test.dropna()
            y_test_clean = y_test[X_test_clean.index]
        else:
            imputer_strategy = 'mean' if strategy == "Mean Imputation" else 'median' if strategy == "Median Imputation" else 'most_frequent'
            imputer = SimpleImputer(strategy=imputer_strategy)
            X_train_clean = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
            X_test_clean = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
            y_train_clean = y_train
            y_test_clean = y_test
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_clean, y_train_clean)
        y_pred = model.predict(X_test_clean)
        accuracy = accuracy_score(y_test_clean, y_pred)
        results.append({'Strategy': strategy, 'Accuracy': accuracy})
    
    results_df = pd.DataFrame(results)
    
    fig = px.bar(results_df, x='Strategy', y='Accuracy', title='Comparison of Missing Data Handling Strategies')
    st.plotly_chart(fig)
    
    st.write("Accuracy scores:")
    st.write(results_df)
    
    st.write("""
    💡 **Interpretation:**
    - Different strategies can lead to different model performances.
    - The best strategy may depend on the specific dataset and problem.
    - Consider the nature of your data and the assumptions of each method when choosing a strategy.
    """)

def quiz_section():
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "Which of the following is NOT a type of missing data mechanism?",
            "options": [
                "Missing Completely at Random (MCAR)",
                "Missing at Random (MAR)",
                "Missing Not at Random (MNAR)",
                "Missing Partially at Random (MPAR)"
            ],
            "correct": "Missing Partially at Random (MPAR)",
            "explanation": "The three main types of missing data mechanisms are MCAR, MAR, and MNAR. 'Missing Partially at Random' is not a standard classification of missing data mechanisms."
        },
        {
            "question": "What is the main disadvantage of listwise deletion?",
            "options": [
                "It's computationally expensive",
                "It can introduce bias if data is not MCAR",
                "It always increases the variance of estimates",
                "It can only be used with categorical variables"
            ],
            "correct": "It can introduce bias if data is not MCAR",
            "explanation": "Listwise deletion can introduce bias if the data is not Missing Completely at Random (MCAR). It can also result in a significant loss of data, potentially reducing statistical power."
        },
        {
            "question": "Which imputation method replaces missing values with the most frequent value in the column?",
            "options": [
                "Mean imputation",
                "Median imputation",
                "Most frequent imputation",
                "Regression imputation"
            ],
            "correct": "Most frequent imputation",
            "explanation": "Most frequent imputation (also known as mode imputation) replaces missing values with the most frequent value in the column. This method is particularly useful for categorical variables."
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