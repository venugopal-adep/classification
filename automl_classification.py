import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Set page config
st.set_page_config(page_title="AutoML Classification Explorer", layout="wide", initial_sidebar_state="expanded")

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
st.title("🤖 AutoML Classification Explorer")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the power of automated machine learning for classification tasks!")

# Helper functions
def load_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    elif dataset_name == "Digits":
        data = datasets.load_digits()
    X = data.data
    y = data.target
    return X, y, data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Sidebar
st.sidebar.header("Dataset Selection")
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine", "Digits"))

# Load data
X, y, data = load_dataset(dataset_name)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🧮 Model Evaluation", "🏆 Best Model", "📚 Dataset Description"])

with tab1:
    st.header("Data Overview")
    st.write("Dataset shape:", X.shape)
    st.write("Target classes:", data['target_names'])
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    st.write("First 10 rows of the dataset:")
    st.dataframe(df.head(10))

    # Feature importance plot (using Random Forest)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    feature_importance = pd.DataFrame({'feature': data.feature_names, 'importance': rf.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    fig = go.Figure(go.Bar(x=feature_importance['feature'], y=feature_importance['importance']))
    fig.update_layout(title='Feature Importance', xaxis_title='Features', yaxis_title='Importance')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Model Evaluation")

    models = [
        ("Random Forest", RandomForestClassifier()),
        ("SVM", SVC()),
        ("Logistic Regression", LogisticRegression()),
        ("Decision Tree", DecisionTreeClassifier()),
        ("KNN", KNeighborsClassifier()),
        ("Gaussian Naive Bayes", GaussianNB()),
        ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
        ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis()),
        ("Gradient Boosting", GradientBoostingClassifier()),
        ("AdaBoost", AdaBoostClassifier())
    ]

    results = []
    for model_name, model in models:
        model.fit(X_train, y_train)
        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
        results.append([model_name, accuracy, precision, recall, f1])

    df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
    df_results = df_results.sort_values(by=["Accuracy", "F1-Score"], ascending=False)

    st.table(df_results)

    # Performance comparison plot
    fig = go.Figure()
    for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        fig.add_trace(go.Bar(x=df_results["Model"], y=df_results[metric], name=metric))
    fig.update_layout(title='Model Performance Comparison', xaxis_title='Models', yaxis_title='Score', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Best Model")
    best_model_name = df_results.iloc[0]["Model"]
    best_model = [model for model_name, model in models if model_name == best_model_name][0]

    st.write(f"The best-performing model is: **{best_model_name}**")
    st.write(f"Accuracy: {df_results.iloc[0]['Accuracy']:.4f}")
    st.write(f"F1-Score: {df_results.iloc[0]['F1-Score']:.4f}")

    # Feature importance for the best model (if applicable)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({'feature': data.feature_names, 'importance': best_model.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        fig = go.Figure(go.Bar(x=feature_importance['feature'], y=feature_importance['importance']))
        fig.update_layout(title=f'Feature Importance for {best_model_name}', xaxis_title='Features', yaxis_title='Importance')
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Dataset Description")
    st.write(data['DESCR'])

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates automated machine learning for classification tasks. Select a dataset and explore the different tabs to learn more!")
