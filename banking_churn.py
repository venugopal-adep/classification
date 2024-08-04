import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os

# Set page config
st.set_page_config(page_title="Customer Churn Prediction App", layout="wide")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("Churn_Modelling.csv")
    return data

data = load_data()

# Function to get feature order
def get_feature_order():
    X = data.drop(['Surname', 'Exited'], axis=1)
    X = pd.get_dummies(X, drop_first=True)
    X = X.rename(columns={'Gender_Male': 'Gender'})
    return X.columns.tolist()

# Main title
st.title("Customer Churn Prediction App")

# Create tabs
tabs = st.tabs(["How to Use", "Explore Data", "Train Model", "Make Predictions", "Tune Model", "Validate Model"])

with tabs[0]:
    st.header("What is this application about?")
    st.write("""
    This app helps banks predict which customers might leave their services (this is called 'churn'). 
    It's like having a crystal ball that tells you which customers might say goodbye!

    Here's what you can do with this app:

    1. **Explore Data**: Look at charts and graphs to understand customer information better.
    2. **Train Model**: Teach the computer to make predictions based on customer data.
    3. **Make Predictions**: Use the trained computer to guess if a new customer might leave.
    4. **Tune Model**: Make the computer even smarter at guessing.
    5. **Validate Model**: Check how good the computer is at making these guesses.

    For example, the app might tell you that customers who are older and have been with the bank for a long time are less likely to leave. Or it might show that customers with low account balances are more likely to go to another bank.

    This information can help banks take care of their customers better and keep them happy!
    """)

with tabs[1]:
    st.header("Explore Customer Data")

    # Show sample data
    st.subheader("Sample Customer Information")
    st.dataframe(data.head())

    # Data info
    st.subheader("Dataset Overview")
    buffer = st.empty()
    buffer.text(data.info())

    # Visualizations
    st.subheader("Visual Insights")

    col1, col2 = st.columns(2)

    with col1:
        # Churn distribution
        fig = px.pie(data, names='Exited', title='Customer Churn Distribution')
        st.plotly_chart(fig, use_container_width=True)

        # Age distribution
        fig = px.histogram(data, x='Age', title='Customer Age Distribution', marginal='box')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Balance vs. EstimatedSalary
        fig = px.scatter(data, x='Balance', y='EstimatedSalary', color='Exited', 
                         title='Balance vs. Estimated Salary', 
                         labels={'Exited': 'Churned'})
        st.plotly_chart(fig, use_container_width=True)

        # Credit Score vs. Age
        fig = px.scatter(data, x='CreditScore', y='Age', color='Exited', 
                         title='Credit Score vs. Age',
                         labels={'Exited': 'Churned'})
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Between Features")
    numeric_data = data.select_dtypes(include=[np.number])
    fig = px.imshow(numeric_data.corr(), title='Correlation Heatmap')
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.header("Train a Prediction Model")

    # Preprocessing
    X = data.drop(['Surname', 'Exited'], axis=1)
    y = data['Exited']
    X = pd.get_dummies(X, drop_first=True)
    X = X.rename(columns={'Gender_Male': 'Gender'})

    feature_order = X.columns.tolist()

    # Model selection
    model_name = st.selectbox("Choose a prediction method", 
                              ["Decision Tree", "Random Forest", "Support Vector Machine", "Logistic Regression", "K-Nearest Neighbors"])

    # Training percentage
    train_size = st.slider("Select percentage of data for training (the rest will be used for testing)", 50, 90, 80)

    # Train model
    if st.button("Train Model"):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size/100, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Select and train model
        if model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif model_name == "Support Vector Machine":
            model = SVC(random_state=42)
        elif model_name == "Logistic Regression":
            model = LogisticRegression(random_state=42)
        else:
            model = KNeighborsClassifier()

        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Display metrics
        st.subheader("Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.2f}")
        col2.metric("Precision", f"{precision:.2f}")
        col3.metric("Recall", f"{recall:.2f}")
        col4.metric("F1-Score", f"{f1:.2f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, title='Confusion Matrix',
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Stayed', 'Churned'],
                        y=['Stayed', 'Churned'])
        st.plotly_chart(fig)

        # Save model and feature order
        filename = f"{model_name.replace(' ', '')}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump((model, scaler, feature_order), file)
        st.success(f"Model saved as {filename}")

with tabs[3]:
    st.header("Make Predictions for New Customers")

    # Load model
    model_name = st.selectbox("Select a trained model", 
                              ["Decision Tree", "Random Forest", "Support Vector Machine", "Logistic Regression", "K-Nearest Neighbors"])
    filename = f"{model_name.replace(' ', '')}.pkl"

    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            model, scaler, feature_order = pickle.load(file)

        # Input features
        st.subheader("Enter Customer Details")
        col1, col2 = st.columns(2)

        with col1:
            credit_score = st.slider("Credit Score", 300, 900, 600)
            age = st.slider("Age", 18, 100, 35)
            tenure = st.slider("Years with Bank", 0, 10, 5)
            balance = st.number_input("Account Balance", 0.0, 250000.0, 50000.0)

        with col2:
            num_products = st.selectbox("Number of Bank Products", [1, 2, 3, 4])
            has_credit_card = st.checkbox("Has Credit Card")
            is_active_member = st.checkbox("Is Active Member")
            estimated_salary = st.number_input("Estimated Yearly Salary", 0.0, 250000.0, 50000.0)

        geography = st.selectbox("Country", ["France", "Spain", "Germany"])
        gender = st.radio("Gender", ["Female", "Male"])

        # Make prediction
        if st.button("Predict Churn"):
            features = pd.DataFrame({
                'CreditScore': [credit_score],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_products],
                'HasCrCard': [int(has_credit_card)],
                'IsActiveMember': [int(is_active_member)],
                'EstimatedSalary': [estimated_salary],
                'Geography_Germany': [1 if geography == 'Germany' else 0],
                'Geography_Spain': [1 if geography == 'Spain' else 0],
                'Gender': [1 if gender == 'Male' else 0]
            })

            # Ensure feature order matches the one used during training
            features = features.reindex(columns=feature_order, fill_value=0)

            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)

            st.subheader("Prediction")
            if prediction[0] == 0:
                st.success("Good news! This customer is likely to stay with the bank.")
            else:
                st.error("Watch out! This customer might leave the bank soon.")
    else:
        st.warning("Please train a model first before making predictions.")

with tabs[4]:
    st.header("Fine-tune the Prediction Model")

    # Preprocessing
    X = data.drop(['Surname', 'Exited'], axis=1)
    y = data['Exited']
    X = pd.get_dummies(X, drop_first=True)
    X = X.rename(columns={'Gender_Male': 'Gender'})

    # Model selection
    model_name = st.selectbox("Choose a model to tune", 
                              ["Decision Tree", "Random Forest", "Support Vector Machine", "Logistic Regression", "K-Nearest Neighbors"])

    # Hyperparameter ranges
    if model_name == "Decision Tree":
        param_grid = {
            'max_depth': st.slider("Maximum tree depth", 1, 20, (1, 10)),
            'min_samples_split': st.slider("Minimum samples to split", 2, 20, (2, 10)),
            'min_samples_leaf': st.slider("Minimum samples in leaf", 1, 20, (1, 5))
        }
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "Random Forest":
        param_grid = {
            'n_estimators': st.slider("Number of trees", 10, 200, (10, 100)),
            'max_depth': st.slider("Maximum tree depth", 1, 20, (1, 10)),
            'min_samples_split': st.slider("Minimum samples to split", 2, 20, (2, 10))
        }
        model = RandomForestClassifier(random_state=42)
    elif model_name == "Support Vector Machine":
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto'] + [0.1, 1, 10]
        }
        model = SVC(random_state=42)
    elif model_name == "Logistic Regression":
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        model = LogisticRegression(random_state=42)
    else:
        param_grid = {
            'n_neighbors': st.slider("Number of neighbors", 1, 20, (1, 10)),
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
        model = KNeighborsClassifier()

    if st.button("Start Tuning"):
        # Perform Grid Search
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X, y)

        st.subheader("Best Settings Found")
        st.write(grid_search.best_params_)

        st.subheader("Best Performance Score")
        st.write(grid_search.best_score_)

with tabs[5]:
    st.header("Validate Model Performance")

    # Preprocessing
    X = data.drop(['Surname', 'Exited'], axis=1)
    y = data['Exited']
    X = pd.get_dummies(X, drop_first=True)
    X = X.rename(columns={'Gender_Male': 'Gender'})

    # Model selection
    model_name = st.selectbox("Choose a model to validate", 
                              ["Decision Tree", "Random Forest", "Support Vector Machine", "Logistic Regression", "K-Nearest Neighbors"])

    if model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "Support Vector Machine":
        model = SVC(random_state=42)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42)
    else:
        model = KNeighborsClassifier()

    # K-fold selection
    k = st.slider("Select number of validation rounds", 2, 10, 5)

    if st.button("Start Validation"):
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=k)

        st.subheader("Validation Results")
        st.write(f"Average Performance Score: {cv_scores.mean():.2f}")
        st.write(f"Score Variation: {cv_scores.std():.2f}")

        # Plot scores
        fig = px.line(x=range(1, k+1), y=cv_scores, markers=True,
                      labels={'x': 'Validation Round', 'y': 'Performance Score'},
                      title='Validation Scores Across Rounds')
        st.plotly_chart(fig)

# Add custom CSS for better aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #4e8cff;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4e8cff;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        color: #4e8cff;
    }
</style>
""", unsafe_allow_html=True)