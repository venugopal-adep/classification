import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df, wine.target_names

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
                    color_continuous_scale='Viridis')
    return fig

def train_initial_model(X_train, y_train, initial_params):
    model = SVC(**initial_params)
    model.fit(X_train, y_train)
    return model

def perform_gridsearch(X_train, y_train, param_grid):
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
    with st.spinner("Running GridSearchCV... This may take a moment."):
        grid_search.fit(X_train, y_train)
    #st.success("GridSearchCV completed!")
    return grid_search

def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted'),
        "Recall": recall_score(y_true, y_pred, average='weighted'),
        "F1-score": f1_score(y_true, y_pred, average='weighted')
    }

def main():
    st.set_page_config(layout="wide")
    st.title("Wine Classification")
    st.write("**Developed by : Venugopal Adep**")

    df, target_names = load_data()

    # Sidebar for section selection
    st.sidebar.title("Sections")
    show_dataset = st.sidebar.checkbox("Dataset (tabular format)", value=False)
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
        st.header("Dataset")
        st.dataframe(df)

    if show_description:
        st.header("Description of columns")
        st.write(load_wine().DESCR)

    if show_stats:
        st.header("Descriptive Statistics")
        st.dataframe(df.describe())

    if show_univariate:
        st.header("Univariate Analysis")
        feature = st.selectbox("Select a feature for histogram", df.columns[:-1])
        fig, ax = plt.subplots()
        df[feature].hist(bins=20, ax=ax)
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
        st.pyplot(fig)

    if show_correlation:
        st.header("Correlation Analysis")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Prepare data for modeling
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initial suboptimal parameters
    initial_params = {'C': 0.1, 'kernel': 'linear', 'gamma': 'scale'}
    model = train_initial_model(X_train_scaled, y_train, initial_params)
    y_pred = model.predict(X_test_scaled)
    metrics = calculate_metrics(y_test, y_pred)

    if show_initial:
        st.header("Initial Model Performance")
        st.subheader("Initial Parameters:")
        st.json(initial_params)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_radar_chart(metrics, "Initial Model Metrics"), use_container_width=True)
        with col2:
            cm = confusion_matrix(y_test, y_pred)
            st.plotly_chart(plot_confusion_matrix(cm, "Initial Confusion Matrix"), use_container_width=True)

    # GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 1]
    }

    if show_gridsearch:
        st.header("GridSearchCV Optimization")
        st.subheader("Parameter Grid:")
        st.json(param_grid)
        grid_search = perform_gridsearch(X_train_scaled, y_train, param_grid)
    else:
        # Perform GridSearchCV anyway to have results for other sections
        grid_search = perform_gridsearch(X_train_scaled, y_train, param_grid)

    # Best model results
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test_scaled)
    best_metrics = calculate_metrics(y_test, y_pred_best)

    if show_optimized:
        st.header("Optimized Model Performance")
        st.subheader("Best Parameters:")
        st.json(grid_search.best_params_)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_radar_chart(best_metrics, "Optimized Model Metrics"), use_container_width=True)
        with col4:
            cm_best = confusion_matrix(y_test, y_pred_best)
            st.plotly_chart(plot_confusion_matrix(cm_best, "Optimized Confusion Matrix"), use_container_width=True)

    if show_comparison:
        st.header("Performance Comparison")
        
        # Metrics comparison
        fig = go.Figure()
        x = list(metrics.keys())
        fig.add_trace(go.Bar(x=x, y=list(metrics.values()), name="Initial Model", marker_color='lightblue'))
        fig.add_trace(go.Bar(x=x, y=list(best_metrics.values()), name="Optimized Model", marker_color='darkblue'))
        fig.update_layout(barmode='group', yaxis_range=[0,1], title="Metrics Comparison")
        st.plotly_chart(fig, use_container_width=True)

        # Parameter comparison
        st.subheader("Parameter Comparison")
        param_comparison = pd.DataFrame({
            'Parameter': list(initial_params.keys()),
            'Initial Value': list(initial_params.values()),
            'Optimized Value': [grid_search.best_params_[param] for param in initial_params.keys()]
        })
        st.table(param_comparison)

if __name__ == "__main__":
    main()