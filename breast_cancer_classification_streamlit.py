import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    return df, cancer.target_names

def plot_radar_chart(metrics, title):
    fig = go.Figure(data=go.Scatterpolar(
        r=list(metrics.values()),
        theta=list(metrics.keys()),
        fill='toself'
    ))
    fig.update_layout(title=title, polar=dict(radialaxis=dict(range=[0, 1])))
    return fig

def plot_confusion_matrix(cm, title):
    fig = px.imshow(cm, text_auto=True, aspect="auto", title=title, 
                    color_continuous_scale='Viridis',
                    labels=dict(x="Predicted", y="Actual"))
    return fig

def plot_roc_curve(y_true, y_score, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(title=title, xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    return fig

def train_initial_model(X_train, y_train, initial_params):
    model = SVC(**initial_params, probability=True)
    model.fit(X_train, y_train)
    return model

def perform_gridsearch(X_train, y_train, param_grid):
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
    with st.spinner("Running GridSearchCV... This may take a moment."):
        grid_search.fit(X_train, y_train)
    return grid_search

def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred)
    }

def main():
    st.set_page_config(layout="wide")
    st.title("Breast Cancer Classification")
    st.write("**Developed by : Venugopal Adep**")

    df, target_names = load_data()

    # Sidebar for section selection
    st.sidebar.title("Sections")
    show_dataset = st.sidebar.checkbox("Dataset Overview", value=False)
    show_description = st.sidebar.checkbox("Description of columns", value=False)
    show_stats = st.sidebar.checkbox("Descriptive statistics", value=False)
    show_univariate = st.sidebar.checkbox("Univariate Analysis", value=False)
    show_bivariate = st.sidebar.checkbox("Bivariate Analysis", value=False)
    show_correlation = st.sidebar.checkbox("Correlation analysis", value=False)
    show_initial = st.sidebar.checkbox("Initial Model Performance", value=False)
    show_gridsearch = st.sidebar.checkbox("GridSearchCV Optimization", value=False)
    show_optimized = st.sidebar.checkbox("Optimized Model Performance", value=False)
    show_comparison = st.sidebar.checkbox("Performance Comparison", value=False)

    if show_dataset:
        st.header("Dataset Overview")
        st.write(f"Number of samples: {df.shape[0]}")
        st.write(f"Number of features: {df.shape[1] - 1}")
        st.write("Class distribution:")
        st.write(df['target'].value_counts(normalize=True))
        st.dataframe(df.head())

    if show_description:
        st.header("Description of columns")
        st.write(load_breast_cancer().DESCR)

    if show_stats:
        st.header("Descriptive Statistics")
        st.dataframe(df.describe())

    if show_univariate:
        st.header("Univariate Analysis")
        feature = st.selectbox("Select a feature for histogram", df.columns[:-1])
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=feature, hue='target', kde=True, ax=ax)
        plt.title(f"Distribution of {feature} by Target")
        st.pyplot(fig)

    if show_bivariate:
        st.header("Bivariate Analysis")
        x_feature = st.selectbox("Select X feature", df.columns[:-1], key='x_feature')
        y_feature = st.selectbox("Select Y feature", df.columns[:-1], key='y_feature')
        fig, ax = plt.subplots()
        scatter = ax.scatter(df[x_feature], df[y_feature], c=df['target'], cmap='viridis')
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.legend(*scatter.legend_elements(), title="Classes")
        plt.title(f"{x_feature} vs {y_feature}")
        st.pyplot(fig)

    if show_correlation:
        st.header("Correlation Analysis")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
        plt.title("Correlation Heatmap")
        st.pyplot(fig)

    # Prepare data for modeling
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initial suboptimal parameters
    initial_params = {'C': 1, 'kernel': 'linear', 'gamma': 'scale'}
    model = train_initial_model(X_train_scaled, y_train, initial_params)
    y_pred = model.predict(X_test_scaled)
    y_score = model.predict_proba(X_test_scaled)[:, 1]
    metrics = calculate_metrics(y_test, y_pred)

    if show_initial:
        st.header("Initial Model Performance")
        st.subheader("Initial Parameters:")
        st.json(initial_params)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(plot_radar_chart(metrics, "Initial Model Metrics"), use_container_width=True)
        with col2:
            cm = confusion_matrix(y_test, y_pred)
            st.plotly_chart(plot_confusion_matrix(cm, "Initial Confusion Matrix"), use_container_width=True)
        with col3:
            st.plotly_chart(plot_roc_curve(y_test, y_score, "Initial ROC Curve"), use_container_width=True)

    # GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.1, 1],
        'degree': [2, 3, 4]  # Only used by poly kernel
    }

    if show_gridsearch:
        st.header("GridSearchCV Optimization")
        st.subheader("Parameter Grid:")
        st.json(param_grid)
        grid_search = perform_gridsearch(X_train_scaled, y_train, param_grid)
        
        st.subheader("GridSearchCV Results")
        results_df = pd.DataFrame(grid_search.cv_results_)
        st.dataframe(results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
                     .sort_values('rank_test_score').head(10))
    else:
        # Perform GridSearchCV anyway to have results for other sections
        grid_search = perform_gridsearch(X_train_scaled, y_train, param_grid)

    # Best model results
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test_scaled)
    y_score_best = best_model.predict_proba(X_test_scaled)[:, 1]
    best_metrics = calculate_metrics(y_test, y_pred_best)

    if show_optimized:
        st.header("Optimized Model Performance")
        st.subheader("Best Parameters:")
        st.json(grid_search.best_params_)

        col4, col5, col6 = st.columns(3)
        with col4:
            st.plotly_chart(plot_radar_chart(best_metrics, "Optimized Model Metrics"), use_container_width=True)
        with col5:
            cm_best = confusion_matrix(y_test, y_pred_best)
            st.plotly_chart(plot_confusion_matrix(cm_best, "Optimized Confusion Matrix"), use_container_width=True)
        with col6:
            st.plotly_chart(plot_roc_curve(y_test, y_score_best, "Optimized ROC Curve"), use_container_width=True)

    if show_comparison:
        st.header("Performance Comparison")
        
        # Metrics comparison
        fig = go.Figure()
        x = list(metrics.keys())
        fig.add_trace(go.Bar(x=x, y=list(metrics.values()), name="Initial Model", marker_color='lightblue'))
        fig.add_trace(go.Bar(x=x, y=list(best_metrics.values()), name="Optimized Model", marker_color='darkblue'))
        fig.update_layout(barmode='group', yaxis_range=[0,1], title="Metrics Comparison")
        st.plotly_chart(fig, use_container_width=True)

        # ROC curve comparison
        fig = go.Figure()
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Initial ROC (AUC = {roc_auc:.2f})'))
        
        fpr_best, tpr_best, _ = roc_curve(y_test, y_score_best)
        roc_auc_best = auc(fpr_best, tpr_best)
        fig.add_trace(go.Scatter(x=fpr_best, y=tpr_best, mode='lines', name=f'Optimized ROC (AUC = {roc_auc_best:.2f})'))
        
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig.update_layout(title='ROC Curve Comparison', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig, use_container_width=True)

        # Parameter comparison
        st.subheader("Parameter Comparison")
        param_comparison = pd.DataFrame({
            'Parameter': list(initial_params.keys()) + ['AUC'],
            'Initial Value': list(initial_params.values()) + [roc_auc],
            'Optimized Value': [grid_search.best_params_.get(param, 'N/A') for param in initial_params.keys()] + [roc_auc_best]
        })
        st.table(param_comparison)

if __name__ == "__main__":
    main()