import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import plotly.graph_objects as go
import plotly.express as px

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv"
    data = pd.read_csv(url)
    return data

def main():
    st.set_page_config(page_title="Loan Approval Prediction Demo", layout="wide")
    
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
    .quiz-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("💼 Loan Approval Prediction")
    
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
    st.header("Understanding Loan Approval Prediction")
    
    st.write("""
    Loan approval prediction is a crucial application of machine learning in the financial sector. 
    It involves using various features of loan applicants to predict whether their loan application will be approved or not.

    Key aspects of loan approval prediction include:
    """)

    aspects = {
        "Data Collection": "Gathering relevant information about loan applicants",
        "Feature Engineering": "Creating and selecting the most informative features",
        "Model Selection": "Choosing an appropriate machine learning algorithm",
        "Model Training": "Using historical data to train the model",
        "Model Evaluation": "Assessing the model's performance using various metrics",
        "Prediction": "Using the trained model to make predictions on new loan applications"
    }

    for aspect, description in aspects.items():
        st.subheader(f"{aspect}")
        st.write(description)

    st.write("""
    Loan approval prediction is important because:
    - It helps banks automate the loan approval process
    - It can reduce the risk of defaults by identifying high-risk applicants
    - It ensures consistency in the loan approval process
    - It can potentially reduce bias in loan approvals when properly implemented
    """)

def experiment_section():
    st.header("🧪 Experiment with Loan Approval Prediction")

    data = load_data()

    st.subheader("Data Overview")
    st.write(data.head())
    st.write(f"Dataset shape: {data.shape}")

    # Prepare the data
    X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = data['Loan_Status']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps
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
                            ('classifier', RandomForestClassifier(random_state=42))])

    if st.button("Train Model and Show Results"):
        with st.spinner("Training model... This may take a moment."):
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Display results
            st.subheader("Model Performance")
            st.text(classification_report(y_test, y_pred))

            # Display confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), x=['Not Approved', 'Approved'], y=['Not Approved', 'Approved'])
            fig.update_layout(title='Confusion Matrix')
            st.plotly_chart(fig)

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines'))
            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig.update_layout(title=f'ROC Curve (AUC = {roc_auc:.2f})',
                              xaxis_title='False Positive Rate',
                              yaxis_title='True Positive Rate')
            st.plotly_chart(fig)

            # Feature importance
            feature_importance = model.named_steps['classifier'].feature_importances_
            feature_names = (numeric_features.tolist() + 
                            model.named_steps['preprocessor']
                            .named_transformers_['cat']
                            .named_steps['onehot']
                            .get_feature_names_out(categorical_features).tolist())
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(10)

            fig = px.bar(feature_importance_df, x='importance', y='feature', orientation='h', title='Top 10 Important Features')
            st.plotly_chart(fig)

def visualization_section():
    st.header("📊 Visualizing Loan Approval Patterns")

    data = load_data()

    # Loan Status Distribution
    st.subheader("Loan Status Distribution")
    fig = px.pie(data, names='Loan_Status', title='Distribution of Loan Status')
    st.plotly_chart(fig)

    # Loan Amount vs. Applicant Income
    st.subheader("Loan Amount vs. Applicant Income")
    fig = px.scatter(data, x='ApplicantIncome', y='LoanAmount', color='Loan_Status', 
                     title='Loan Amount vs. Applicant Income')
    st.plotly_chart(fig)

    # Loan Status by Education and Property Area
    st.subheader("Loan Status by Education and Property Area")
    fig = px.histogram(data, x='Education', color='Loan_Status', facet_col='Property_Area', 
                       title='Loan Status by Education and Property Area')
    st.plotly_chart(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    corr = numeric_data.corr()
    fig = px.imshow(corr, title='Correlation Heatmap of Numeric Features')
    st.plotly_chart(fig)

def quiz_section():
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main goal of loan approval prediction?",
            "options": [
                "To increase the number of loans given",
                "To predict whether a loan application will be approved or not",
                "To determine the interest rate for a loan",
                "To calculate the credit score of applicants"
            ],
            "correct": "To predict whether a loan application will be approved or not",
            "explanation": "The main goal of loan approval prediction is to use machine learning models to predict whether a loan application is likely to be approved or rejected based on various applicant features."
        },
        {
            "question": "Which of the following is NOT typically a feature used in loan approval prediction?",
            "options": [
                "Applicant Income",
                "Credit History",
                "Loan Amount",
                "Applicant's Favorite Color"
            ],
            "correct": "Applicant's Favorite Color",
            "explanation": "While Applicant Income, Credit History, and Loan Amount are typical features used in loan approval prediction, an applicant's favorite color is not relevant and is not typically used in such models."
        },
        {
            "question": "Why is feature importance analysis useful in loan approval prediction?",
            "options": [
                "It helps in determining the loan interest rate",
                "It identifies which features have the most impact on the prediction",
                "It automatically approves all loan applications",
                "It calculates the exact loan amount to be approved"
            ],
            "correct": "It identifies which features have the most impact on the prediction",
            "explanation": "Feature importance analysis helps identify which features (such as income, credit history, etc.) have the most significant impact on the loan approval prediction. This can provide insights into the model's decision-making process."
        },
        {
            "question": "What does a high AUC (Area Under the Curve) value indicate in the context of loan approval prediction?",
            "options": [
                "The model is not performing well",
                "The model has a good ability to distinguish between approved and rejected loans",
                "The loan amount is too high",
                "The applicant's credit score is excellent"
            ],
            "correct": "The model has a good ability to distinguish between approved and rejected loans",
            "explanation": "A high AUC value (closer to 1) indicates that the model has a good ability to distinguish between positive cases (approved loans) and negative cases (rejected loans). It's a measure of the model's discriminative power."
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