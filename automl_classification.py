import streamlit as st
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
import pandas as pd

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

def main():
    st.title("AutoML Classification App")
    st.write('**Developed by : Venugopal Adep**')
    
    dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine", "Digits"))
    
    X, y, data = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    st.write("Dataset shape:", X.shape)
    st.write("Target classes:", data['target_names'])
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    st.write("First 10 rows of the dataset:")
    st.write(df.head(10))
    
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
    
    st.subheader("Evaluation Results")
    st.table(df_results)
    
    best_model_name = df_results.iloc[0]["Model"]
    best_model = [model for model_name, model in models if model_name == best_model_name][0]
    
    st.subheader("Best Model")
    st.write(f"The best-performing model is: {best_model_name}")
    st.subheader("Dataset")
    st.write(data['DESCR'])
    
if __name__ == "__main__":
    main()
