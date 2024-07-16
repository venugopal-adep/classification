import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    # Load Titanic dataset
    url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    data = pd.read_csv(url)
    return data

def main():
    st.set_page_config(page_title="Feature Engineering Demo", layout="wide")
    
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
    
    st.title("🛠️ Feature Engineering Demo")
    
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
    st.header("Understanding Feature Engineering")
    
    st.write("""
    Feature engineering is the process of using domain knowledge to extract features from raw data via data mining techniques. 
    These features can be used to improve the performance of machine learning algorithms.

    Key aspects of feature engineering include:
    """)

    aspects = {
        "Feature Creation": "Creating new features from existing ones",
        "Feature Transformation": "Changing the scale or distribution of features",
        "Feature Selection": "Choosing the most relevant features for the model",
        "Feature Encoding": "Converting categorical variables into numerical form",
        "Handling Missing Data": "Imputing or removing missing values",
        "Dimensionality Reduction": "Reducing the number of input variables"
    }

    for aspect, description in aspects.items():
        st.subheader(f"{aspect}")
        st.write(description)

    st.write("""
    Feature engineering is crucial because:
    - It can uncover hidden patterns in the data
    - It can make machine learning algorithms perform better
    - It can lead to more interpretable models
    - It allows domain expertise to be incorporated into the modeling process
    """)

def experiment_section():
    st.header("🧪 Experiment with Feature Engineering")

    data = load_data()

    st.subheader("Original Data Overview")
    st.write(data.head())

    st.write(f"Dataset shape: {data.shape}")

    # Display available columns
    st.write("Available columns:", list(data.columns))

    # Feature Engineering Options
    st.subheader("Select Feature Engineering Techniques")
    
    create_age_group = st.checkbox("Create Age Groups")
    encode_sex = st.checkbox("Encode Sex Feature")
    fill_missing_age = st.checkbox("Fill Missing Age Values")

    # Apply selected feature engineering techniques
    if create_age_group:
        data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 18, 35, 50, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])
    
    if encode_sex:
        data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    
    if fill_missing_age:
        data['Age'] = data['Age'].fillna(data['Age'].median())

    st.subheader("Data After Feature Engineering")
    st.write(data.head())

    # Prepare data for modeling
    features = ['Pclass', 'Sex', 'Age', 'Fare']
    if create_age_group:
        features.append('AgeGroup')

    X = data[features]
    y = data['Survived']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create preprocessing steps
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create and train the model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier())])

    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy:.2f}")

    st.write("""
    Experiment with different feature engineering techniques to see how they affect the model's performance.
    Notice how creating new features or encoding existing ones can impact the accuracy.
    """)

def visualization_section():
    st.header("📊 Visualizing Feature Engineering Effects")

    data = load_data()

    # Feature to visualize
    feature = st.selectbox("Select a feature to visualize", ['Age', 'Fare', 'Pclass', 'Sex'])

    # Create age groups
    data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 18, 35, 50, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])

    # Encode 'Sex' column
    data['Sex_encoded'] = data['Sex'].map({'male': 0, 'female': 1})

    # Visualization options
    viz_type = st.radio("Select visualization type", ["Distribution", "Survival Rate"])

    if viz_type == "Distribution":
        st.subheader(f"Distribution of {feature}")
        if feature in ['Age', 'Fare', 'Pclass']:
            fig = px.histogram(data, x=feature, color='Survived', barmode='group', 
                               labels={'Survived': 'Survived'})
        else:  # For 'Sex' feature
            fig = px.histogram(data, x=feature, color='Survived', barmode='group', 
                               labels={'Survived': 'Survived'}, category_orders={"Sex": ["male", "female"]})
        st.plotly_chart(fig)

    else:
        st.subheader(f"Survival Rate by {feature}")
        if feature in ['Age', 'Fare', 'Pclass']:
            survival_rate = data.groupby(feature)['Survived'].mean().reset_index()
            fig = px.line(survival_rate, x=feature, y='Survived', 
                          labels={'Survived': 'Survival Rate'})
        else:  # For 'Sex' feature
            survival_rate = data.groupby(feature)['Survived'].mean().reset_index()
            fig = px.bar(survival_rate, x=feature, y='Survived', 
                         labels={'Survived': 'Survival Rate'})
        st.plotly_chart(fig)

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    corr = data[['Survived', 'Pclass', 'Sex_encoded', 'Age', 'Fare']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    plt.title("Correlation Heatmap")
    st.pyplot(fig)

    st.write("""
    Visualizing the data helps understand the relationships between features and the target variable.
    The correlation heatmap shows how different features are related to each other and to the survival outcome.
    Note that 'Sex' has been encoded (0 for male, 1 for female) for the correlation calculation.
    """)

def quiz_section():
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main purpose of feature engineering?",
            "options": [
                "To increase the number of features in the dataset",
                "To improve the performance of machine learning algorithms",
                "To remove all categorical variables",
                "To normalize all features"
            ],
            "correct": "To improve the performance of machine learning algorithms",
            "explanation": "Feature engineering aims to create, transform, or select features that can help machine learning algorithms perform better by uncovering hidden patterns or incorporating domain knowledge."
        },
        {
            "question": "Which of the following is NOT a common feature engineering technique?",
            "options": [
                "One-hot encoding",
                "Feature scaling",
                "Creating interaction terms",
                "Removing all outliers"
            ],
            "correct": "Removing all outliers",
            "explanation": "While handling outliers can be part of data preprocessing, automatically removing all outliers is not a standard feature engineering technique. The other options (one-hot encoding, feature scaling, and creating interaction terms) are common feature engineering practices."
        },
        {
            "question": "In the Titanic dataset, what does creating age groups achieve?",
            "options": [
                "It makes the data more visually appealing",
                "It can potentially capture non-linear relationships between age and survival",
                "It's required for all machine learning models",
                "It always improves model accuracy"
            ],
            "correct": "It can potentially capture non-linear relationships between age and survival",
            "explanation": "Creating age groups (e.g., Child, Young Adult, Adult, Senior) can help capture non-linear relationships between age and survival. For instance, very young and very old passengers might have had different survival rates compared to middle-aged passengers."
        },
        {
            "question": "Why might encoding the 'Sex' feature as numerical values (0 and 1) be useful?",
            "options": [
                "It makes the data more readable",
                "It allows the model to treat 'Sex' as a continuous variable",
                "It's required for all machine learning models",
                "It can potentially improve model performance and interpretability"
            ],
            "correct": "It can potentially improve model performance and interpretability",
            "explanation": "Encoding categorical variables like 'Sex' into numerical values (e.g., 0 for male, 1 for female) can help certain models perform better and make the results more interpretable. However, it's important to note that not all models require this encoding, and for some, techniques like one-hot encoding might be more appropriate."
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