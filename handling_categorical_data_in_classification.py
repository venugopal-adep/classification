import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from category_encoders import BinaryEncoder, TargetEncoder, CatBoostEncoder, WOEEncoder, JamesSteinEncoder, MEstimateEncoder
from sklearn.impute import SimpleImputer

def main():
    st.set_page_config(page_title="Handling Categorical Data in Classification", layout="wide")
    
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
    
    st.title("🐱 Handling Categorical Data in Classification")
    
    tabs = st.tabs(["📚 Learn", "🧪 Experiment", "📊 Visualization", "🧠 Quiz"])
    
    with tabs[0]:
        learn_section()
    
    with tabs[1]:
        experiment_section()
    
    with tabs[2]:
        visualization_section()
    
    with tabs[3]:
        quiz_section()

def learn_section():
    st.header("Understanding Categorical Data")
    
    st.write("""
    Categorical data is a type of data that can be divided into groups or categories. In machine learning, 
    we often need to convert categorical data into numerical form for our models to process it effectively.

    There are two main types of categorical data:
    1. **Nominal**: Categories with no inherent order (e.g., color, gender)
    2. **Ordinal**: Categories with a meaningful order (e.g., education level, customer satisfaction)

    Here are various techniques for handling categorical data:
    """)

    encodings = {
        "Label Encoding": "Assigns a unique integer to each category. Suitable for ordinal data.",
        "One-Hot Encoding": "Creates binary columns for each category. Suitable for nominal data with low cardinality.",
        "Binary Encoding": "Encodes categories as binary numbers. Useful for high cardinality nominal data.",
        "Frequency Encoding": "Replaces categories with their frequency in the dataset.",
        "Ordinal Encoding": "Assigns ordered integers to categories. Suitable for ordinal data with known order.",
        "Target Encoding": "Replaces categories with the mean target value for that category.",
        "CatBoost Encoding": "A robust version of target encoding that helps prevent overfitting.",
        "Weight of Evidence (WOE) Encoding": "Calculates log odds of target for each category. Useful for binary classification.",
        "James-Stein Encoding": "A 'smoothed' version of target encoding, reducing impact of low-frequency categories.",
        "M-Estimate Encoding": "Another smoothed version of target encoding, using Bayesian principles.",
        "Helmert Encoding": "Compares each level to the mean of subsequent levels.",
        "Sum Encoding": "Compares each level to the overall mean of the target.",
        "Polynomial Encoding": "Creates polynomial contrasts, useful for ordered categories.",
        "Backward Difference Encoding": "Compares each level to its preceding level.",
        "Leave-One-Out Encoding": "A type of target encoding that excludes the current row to prevent data leakage."
    }

    for method, description in encodings.items():
        st.subheader(f"{method}")
        st.write(description)

    st.write("""
    Choosing the right encoding method depends on the nature of your data, the specific requirements of your machine learning model, 
    and the characteristics of your problem. It's often beneficial to experiment with multiple encoding techniques and evaluate their 
    impact on model performance.
    """)

def experiment_section():
    st.header("🧪 Experiment with Encoding Methods")

    # Sample data
    data = pd.DataFrame({
        'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red', 'Green', 'Yellow'],
        'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small', 'Large', 'Medium'],
        'Target': [0, 1, 1, 0, 1, 1, 0]  # Adding a target variable for target-based encodings
    })

    st.subheader("Original Data")
    st.write(data)

    encoding_method = st.selectbox("Select Encoding Method", 
                                   ["Label Encoding", "One-Hot Encoding", "Frequency Encoding", 
                                    "Ordinal Encoding", "Binary Encoding", "Target Encoding", 
                                    "CatBoost Encoding", "WOE Encoding", "James-Stein Encoding", 
                                    "M-Estimate Encoding"])

    if encoding_method == "Label Encoding":
        st.subheader("Label Encoding")
        le = LabelEncoder()
        data_encoded = data.copy()
        for column in ['Color', 'Size']:
            data_encoded[column] = le.fit_transform(data[column])
        st.write(data_encoded)
        st.write("Note: Label encoding assigns arbitrary numbers. Be cautious with nominal data.")

    elif encoding_method == "One-Hot Encoding":
        st.subheader("One-Hot Encoding")
        data_encoded = pd.get_dummies(data[['Color', 'Size']])
        st.write(data_encoded)
        st.write("Note: One-hot encoding creates new columns for each category.")

    elif encoding_method == "Frequency Encoding":
        st.subheader("Frequency Encoding")
        data_encoded = data.copy()
        for column in ['Color', 'Size']:
            frequency = data[column].value_counts(normalize=True)
            data_encoded[column] = data[column].map(frequency)
        st.write(data_encoded)
        st.write("Note: Frequency encoding replaces categories with their relative frequencies.")

    elif encoding_method == "Ordinal Encoding":
        st.subheader("Ordinal Encoding")
        size_order = ['Small', 'Medium', 'Large']
        oe = OrdinalEncoder(categories=[size_order])
        data_encoded = data.copy()
        data_encoded['Size'] = oe.fit_transform(data[['Size']])
        st.write(data_encoded)
        st.write("Note: Ordinal encoding preserves the order of categories.")

    elif encoding_method == "Binary Encoding":
        st.subheader("Binary Encoding")
        be = BinaryEncoder(cols=['Color', 'Size'])
        data_encoded = be.fit_transform(data[['Color', 'Size']])
        st.write(data_encoded)
        st.write("Note: Binary encoding creates binary columns based on the integer representation of categories.")

    elif encoding_method == "Target Encoding":
        st.subheader("Target Encoding")
        te = TargetEncoder()
        data_encoded = data.copy()
        data_encoded[['Color', 'Size']] = te.fit_transform(data[['Color', 'Size']], data['Target'])
        st.write(data_encoded)
        st.write("Note: Target encoding replaces categories with the mean target value.")

    elif encoding_method == "CatBoost Encoding":
        st.subheader("CatBoost Encoding")
        cbe = CatBoostEncoder()
        data_encoded = data.copy()
        data_encoded[['Color', 'Size']] = cbe.fit_transform(data[['Color', 'Size']], data['Target'])
        st.write(data_encoded)
        st.write("Note: CatBoost encoding is a robust version of target encoding.")

    elif encoding_method == "WOE Encoding":
        st.subheader("Weight of Evidence (WOE) Encoding")
        woe = WOEEncoder()
        data_encoded = data.copy()
        data_encoded[['Color', 'Size']] = woe.fit_transform(data[['Color', 'Size']], data['Target'])
        st.write(data_encoded)
        st.write("Note: WOE encoding calculates log odds of the target for each category.")

    elif encoding_method == "James-Stein Encoding":
        st.subheader("James-Stein Encoding")
        jse = JamesSteinEncoder()
        data_encoded = data.copy()
        data_encoded[['Color', 'Size']] = jse.fit_transform(data[['Color', 'Size']], data['Target'])
        st.write(data_encoded)
        st.write("Note: James-Stein encoding is a smoothed version of target encoding.")

    elif encoding_method == "M-Estimate Encoding":
        st.subheader("M-Estimate Encoding")
        mee = MEstimateEncoder()
        data_encoded = data.copy()
        data_encoded[['Color', 'Size']] = mee.fit_transform(data[['Color', 'Size']], data['Target'])
        st.write(data_encoded)
        st.write("Note: M-Estimate encoding is another smoothed version of target encoding.")

    st.write("""
    Experiment with different encoding methods to see how they transform the data.
    Consider the pros and cons of each method for your specific use case.
    """)

def visualization_section():
    st.header("📊 Visualizing Categorical Data")

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    categories = ['A', 'B', 'C', 'D']
    data = pd.DataFrame({
        'Category': np.random.choice(categories, n_samples),
        'Value': np.random.randn(n_samples)
    })

    st.subheader("Sample Data Distribution")
    fig = px.histogram(data, x='Category', color='Category')
    st.plotly_chart(fig)

    st.write("""
    Visualizing categorical data can help you understand its distribution and relationships.
    This can inform your choice of encoding method and feature engineering strategies.
    """)

    st.subheader("Relationship between Categorical and Numerical Data")
    fig = px.box(data, x='Category', y='Value', color='Category')
    st.plotly_chart(fig)

    st.write("""
    Box plots can reveal how a numerical variable varies across different categories.
    This can be useful for feature selection and understanding the predictive power of categorical variables.
    """)

def quiz_section():
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "Which encoding method is most suitable for high cardinality nominal data?",
            "options": ["Label Encoding", "One-Hot Encoding", "Binary Encoding", "Ordinal Encoding"],
            "correct": "Binary Encoding",
            "explanation": "Binary Encoding is particularly useful for high cardinality nominal data because it creates a compact representation without losing information, unlike One-Hot Encoding which can lead to high dimensionality."
        },
        {
            "question": "What is a potential advantage of using Target Encoding over One-Hot Encoding?",
            "options": [
                "It preserves the order of categories",
                "It can capture the relationship between categories and the target variable",
                "It always results in fewer features",
                "It's computationally less expensive"
            ],
            "correct": "It can capture the relationship between categories and the target variable",
            "explanation": "Target Encoding replaces categories with their mean target value, which can capture the relationship between categories and the target variable. This can be especially useful when certain categories are more predictive of the target than others."
        },
        {
            "question": "Which encoding method is designed to prevent overfitting in target-based encodings?",
            "options": [
                "Label Encoding",
                "Frequency Encoding",
                "CatBoost Encoding",
                "Ordinal Encoding"
            ],
            "correct": "CatBoost Encoding",
            "explanation": "CatBoost Encoding is a robust version of target encoding that helps prevent overfitting. It uses a combination of random permutations and leave-one-out encoding to create a more stable encoding that's less prone to overfitting, especially with small datasets or rare categories."
        },
        {
            "question": "When might Weight of Evidence (WOE) Encoding be particularly useful?",
            "options": [
                "For continuous target variables",
                "For multi-class classification problems",
                "For binary classification problems",
                "For time series data"
            ],
            "correct": "For binary classification problems",
            "explanation": "Weight of Evidence (WOE) Encoding is particularly useful for binary classification problems. It calculates the log odds of the target for each category, which directly relates to the predictive power of the category for a binary outcome."
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