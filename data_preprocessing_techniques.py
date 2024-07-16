import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def main():
    st.set_page_config(page_title="Data Preprocessing Techniques for Classification", layout="wide")
    
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
    
    st.title("🔧 Data Preprocessing Techniques for Classification")
    
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
    st.header("Understanding Data Preprocessing")
    
    st.write("""
    Data preprocessing is a crucial step in any machine learning pipeline, especially for classification tasks. 
    It involves transforming raw data into a clean and meaningful format that can be easily understood by machine learning algorithms.

    Here are various techniques for data preprocessing in classification:
    """)

    techniques = {
        "Handling Missing Values": "Techniques to deal with incomplete data, such as imputation or removal.",
        "Feature Scaling": "Standardization and normalization to bring features to a common scale.",
        "Encoding Categorical Variables": "Converting categorical data into numerical format.",
        "Feature Selection": "Choosing the most relevant features for the classification task.",
        "Dimensionality Reduction": "Reducing the number of features while preserving important information.",
        "Handling Imbalanced Data": "Techniques to address class imbalance in the target variable.",
        "Outlier Detection and Treatment": "Identifying and handling extreme values in the dataset.",
        "Feature Engineering": "Creating new features or transforming existing ones to improve model performance.",
        "Data Augmentation": "Generating synthetic samples to increase the size and diversity of the training set.",
        "Noise Reduction": "Techniques to reduce random variation or error in the data."
    }

    for technique, description in techniques.items():
        st.subheader(f"{technique}")
        st.write(description)

    st.write("""
    Choosing the right preprocessing techniques depends on the nature of your data, the specific requirements of your classification algorithm, 
    and the characteristics of your problem. It's often beneficial to experiment with multiple techniques and evaluate their 
    impact on model performance.
    """)

def experiment_section():
    st.header("🧪 Experiment with Preprocessing Techniques")

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 4)
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    X[:100, 0] = np.nan  # Introduce missing values
    X[800:, 1] = X[800:, 1] * 10  # Introduce outliers

    data = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])
    data['Target'] = y

    st.subheader("Original Data (first 10 rows)")
    st.write(data.head(10))

    preprocessing_method = st.selectbox("Select Preprocessing Method", 
                                        ["Handle Missing Values", "Feature Scaling", 
                                         "Feature Selection", "Dimensionality Reduction", 
                                         "Handle Imbalanced Data"])

    if preprocessing_method == "Handle Missing Values":
        st.subheader("Handling Missing Values")
        imputer = SimpleImputer(strategy='mean')
        data_processed = pd.DataFrame(imputer.fit_transform(data.drop('Target', axis=1)), columns=data.columns[:-1])
        data_processed['Target'] = data['Target']
        st.write(data_processed.head(10))
        st.write("Note: Missing values have been replaced with the mean of the respective feature.")

    elif preprocessing_method == "Feature Scaling":
        scaling_method = st.selectbox("Select Scaling Method", ["StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer"])
        st.subheader(f"Feature Scaling: {scaling_method}")
        
        if scaling_method == "StandardScaler":
            scaler = StandardScaler()
        elif scaling_method == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scaling_method == "RobustScaler":
            scaler = RobustScaler()
        else:
            scaler = Normalizer()
        
        data_processed = pd.DataFrame(scaler.fit_transform(data.drop('Target', axis=1)), columns=data.columns[:-1])
        data_processed['Target'] = data['Target']
        st.write(data_processed.head(10))
        st.write(f"Note: Features have been scaled using {scaling_method}.")

    elif preprocessing_method == "Feature Selection":
        st.subheader("Feature Selection")
        k = st.slider("Select number of features to keep", 1, 4, 2)
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(data.drop('Target', axis=1), data['Target'])
        selected_features = data.columns[:-1][selector.get_support()]
        data_processed = pd.DataFrame(X_selected, columns=selected_features)
        data_processed['Target'] = data['Target']
        st.write(data_processed.head(10))
        st.write(f"Note: {k} most significant features have been selected based on ANOVA F-value.")

    elif preprocessing_method == "Dimensionality Reduction":
        st.subheader("Dimensionality Reduction (PCA)")
        n_components = st.slider("Select number of components", 1, 4, 2)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(data.drop('Target', axis=1))
        data_processed = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        data_processed['Target'] = data['Target']
        st.write(data_processed.head(10))
        st.write(f"Note: Data has been reduced to {n_components} principal components.")

    elif preprocessing_method == "Handle Imbalanced Data":
        st.subheader("Handling Imbalanced Data")
        method = st.radio("Select method", ["Oversampling (SMOTE)", "Undersampling"])
        if method == "Oversampling (SMOTE)":
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(data.drop('Target', axis=1), data['Target'])
        else:
            rus = RandomUnderSampler()
            X_resampled, y_resampled = rus.fit_resample(data.drop('Target', axis=1), data['Target'])
        
        data_processed = pd.DataFrame(X_resampled, columns=data.columns[:-1])
        data_processed['Target'] = y_resampled
        st.write(data_processed.head(10))
        st.write(f"Class distribution before: {dict(data['Target'].value_counts())}")
        st.write(f"Class distribution after: {dict(data_processed['Target'].value_counts())}")

    st.write("""
    Experiment with different preprocessing techniques to see how they transform the data.
    Consider the impact of each method on your classification task.
    """)

def visualization_section():
    st.header("📊 Visualizing Preprocessing Effects")

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X[800:, 1] = X[800:, 1] * 10  # Introduce outliers

    data = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
    data['Target'] = y

    st.subheader("Original Data Distribution")
    fig = px.scatter(data, x='Feature1', y='Feature2', color='Target')
    st.plotly_chart(fig)

    preprocessing_method = st.selectbox("Select Preprocessing Method for Visualization", 
                                        ["Feature Scaling", "Handle Imbalanced Data"])

    if preprocessing_method == "Feature Scaling":
        scaling_method = st.selectbox("Select Scaling Method", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
        
        if scaling_method == "StandardScaler":
            scaler = StandardScaler()
        elif scaling_method == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
        
        data_scaled = pd.DataFrame(scaler.fit_transform(data[['Feature1', 'Feature2']]), columns=['Feature1', 'Feature2'])
        data_scaled['Target'] = data['Target']

        st.subheader(f"Data Distribution after {scaling_method}")
        fig = px.scatter(data_scaled, x='Feature1', y='Feature2', color='Target')
        st.plotly_chart(fig)

    elif preprocessing_method == "Handle Imbalanced Data":
        method = st.radio("Select method", ["Oversampling (SMOTE)", "Undersampling"])
        if method == "Oversampling (SMOTE)":
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(data[['Feature1', 'Feature2']], data['Target'])
        else:
            rus = RandomUnderSampler()
            X_resampled, y_resampled = rus.fit_resample(data[['Feature1', 'Feature2']], data['Target'])
        
        data_resampled = pd.DataFrame(X_resampled, columns=['Feature1', 'Feature2'])
        data_resampled['Target'] = y_resampled

        st.subheader(f"Data Distribution after {method}")
        fig = px.scatter(data_resampled, x='Feature1', y='Feature2', color='Target')
        st.plotly_chart(fig)

    st.write("""
    Visualizing the effects of preprocessing techniques can help you understand how they transform your data.
    This can be crucial in choosing the right preprocessing steps for your classification task.
    """)

def quiz_section():
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "Which preprocessing technique is most suitable for dealing with features on different scales?",
            "options": ["One-Hot Encoding", "Feature Scaling", "SMOTE", "PCA"],
            "correct": "Feature Scaling",
            "explanation": "Feature Scaling (such as standardization or normalization) is used to bring all features to a similar scale. This is important for many machine learning algorithms that are sensitive to the scale of input features."
        },
        {
            "question": "What is the primary purpose of SMOTE in data preprocessing?",
            "options": [
                "To reduce dimensionality",
                "To handle missing values",
                "To balance imbalanced datasets",
                "To remove outliers"
            ],
            "correct": "To balance imbalanced datasets",
            "explanation": "SMOTE (Synthetic Minority Over-sampling Technique) is used to address class imbalance by creating synthetic examples of the minority class, thus balancing the dataset."
        },
        {
            "question": "Which of the following is NOT a common method for handling missing values?",
            "options": [
                "Mean imputation",
                "Median imputation",
                "One-Hot Encoding",
                "Dropping rows with missing values"
            ],
            "correct": "One-Hot Encoding",
            "explanation": "One-Hot Encoding is used for converting categorical variables into a numerical format. It is not a method for handling missing values. Common methods for handling missing values include mean/median imputation, dropping rows or columns, or using more advanced techniques like KNN imputation."
        },
        {
            "question": "What is the main goal of dimensionality reduction techniques like PCA?",
            "options": [
                "To handle missing values",
                "To reduce the number of features while preserving important information",
                "To balance imbalanced datasets",
                "To convert categorical variables to numerical"
            ],
            "correct": "To reduce the number of features while preserving important information",
            "explanation": "Dimensionality reduction techniques like PCA aim to reduce the number of features in a dataset while retaining as much of the important information as possible. This can help in reducing computational complexity and mitigating the curse of dimensionality."
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