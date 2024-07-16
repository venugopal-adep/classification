import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer, load_iris
import shap
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(layout="wide", page_title="Explainable AI with Decision Trees", page_icon="🌳")

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .small-font {
        font-size:18px !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🌳 Explainable AI with Decision Trees 🌳</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("<p class='big-font'>Welcome to the Decision Tree Explainer!</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>Let's explore model interpretability using Decision Trees and SHAP values.</p>", unsafe_allow_html=True)

# Explanation
st.markdown("<p class='medium-font'>What are Decision Trees and SHAP?</p>", unsafe_allow_html=True)
st.markdown("""
<p class='small-font'>
Decision Trees are a popular machine learning algorithm for both classification and regression tasks. They are intuitive and easily interpretable. 

SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance based on cooperative game theory. They help us understand:

- The global importance of features
- How each feature contributes to individual predictions
- How features interact with each other

By combining Decision Trees with SHAP, we can get powerful insights into our models and data.
</p>
""", unsafe_allow_html=True)

# Tabs with custom styling
tab1, tab2, tab3 = st.tabs(["🌳 Model Training & Global Interpretability", "🧠 Local Interpretability", "📊 Feature Interactions"])

# Load data
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Breast Cancer":
        data = load_breast_cancer()
    elif dataset_name == "Iris":
        data = load_iris()
    else:
        raise ValueError("Unknown dataset")
    
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y

with tab1:
    st.markdown("<p class='medium-font'>Model Training & Global Interpretability</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Let's train a Decision Tree model and examine global feature importance using SHAP values.
        </p>
        """, unsafe_allow_html=True)

        dataset = st.selectbox("Select dataset", ["Breast Cancer", "Iris"])
        X, y = load_data(dataset)
        
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", 0, 100, 42)
        
        max_depth = st.slider("Maximum tree depth", 1, 20, 5)
        min_samples_split = st.slider("Minimum samples to split", 2, 20, 2)
        
        if st.button("Train Model and Calculate SHAP Values"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
            
            model.fit(X_train, y_train)
            
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, train_preds)
            test_accuracy = accuracy_score(y_test, test_preds)
            
            st.markdown(f"""
            <p class='small-font'>
            Train Accuracy: {train_accuracy:.4f}<br>
            Test Accuracy: {test_accuracy:.4f}
            </p>
            """, unsafe_allow_html=True)

            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            # Store the number of classes
            n_classes = len(np.unique(y))

    with col2:
        if 'shap_values' in locals():
            st.markdown("<p class='small-font'>Global Feature Importance (SHAP)</p>", unsafe_allow_html=True)
            
            if n_classes == 2:  # Binary classification
                shap_values_plot = shap_values[1]
            else:  # Multi-class classification
                shap_values_plot = np.abs(shap_values).mean(0)
            
            fig = go.Figure(go.Bar(
                y=X.columns,
                x=np.abs(shap_values_plot).mean(0),
                orientation='h'
            ))
            fig.update_layout(
                title='Global Feature Importance',
                xaxis_title='mean(|SHAP value|)',
                yaxis_title='Feature'
            )
            st.plotly_chart(fig)

with tab2:
    st.markdown("<p class='medium-font'>Local Interpretability</p>", unsafe_allow_html=True)
    
    if 'shap_values' in locals():
        st.markdown("""
        <p class='small-font'>
        Let's examine SHAP values for individual predictions to understand local feature importance.
        </p>
        """, unsafe_allow_html=True)

        sample_index = st.slider("Select a sample to explain", 0, len(X_test)-1, 0)
        
        # Print debug information
        st.write("Debug Information:")
        st.write(f"X_test shape: {X_test.shape}")
        st.write(f"shap_values type: {type(shap_values)}")
        if isinstance(shap_values, list):
            st.write(f"shap_values length: {len(shap_values)}")
            st.write(f"shap_values[0] shape: {np.array(shap_values[0]).shape}")
            if len(shap_values) > 1:
                st.write(f"shap_values[1] shape: {np.array(shap_values[1]).shape}")
        else:
            st.write(f"shap_values shape: {np.array(shap_values).shape}")
        st.write(f"Number of features: {len(X_test.columns)}")
        st.write(f"Number of classes: {n_classes}")
        
        # SHAP summary plot
        st.markdown("<p class='small-font'>SHAP Summary Plot</p>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if n_classes == 2:
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
            else:
                shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        st.pyplot(fig)
        plt.clf()  # Clear the current figure
        
        # SHAP waterfall plot
        st.markdown("<p class='small-font'>SHAP Waterfall Plot</p>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if n_classes == 2:
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values_exp = shap_values[1]
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                shap_values_exp = shap_values
                expected_value = explainer.expected_value
            shap.plots.waterfall(shap.Explanation(values=shap_values_exp[sample_index], 
                                                  base_values=expected_value, 
                                                  data=X_test.iloc[sample_index],
                                                  feature_names=X_test.columns),
                                 show=False)
        else:
            predicted_class = model.predict(X_test.iloc[sample_index].to_frame().T)[0]
            shap.plots.waterfall(shap.Explanation(values=shap_values[predicted_class][sample_index], 
                                                  base_values=explainer.expected_value[predicted_class] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
                                                  data=X_test.iloc[sample_index],
                                                  feature_names=X_test.columns),
                                 show=False)
        st.pyplot(fig)
        plt.clf()  # Clear the current figure

with tab3:
    st.markdown("<p class='medium-font'>Feature Interactions</p>", unsafe_allow_html=True)
    
    if 'shap_values' in locals():
        st.markdown("""
        <p class='small-font'>
        Let's explore how features interact with each other using SHAP interaction values.
        </p>
        """, unsafe_allow_html=True)

        # Calculate SHAP interaction values (this can be computationally expensive)
        shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_test)

        # Plot SHAP interaction values
        feature1 = st.selectbox("Select first feature", X.columns)
        feature2 = st.selectbox("Select second feature", X.columns)

        if n_classes == 2:
            interaction_values = shap_interaction_values[1]
        else:
            interaction_values = np.abs(shap_interaction_values).mean(0)

        fig = px.scatter(x=X_test[feature1], y=X_test[feature2], 
                         color=interaction_values[:, X.columns.get_loc(feature1), X.columns.get_loc(feature2)],
                         labels={'x': feature1, 'y': feature2, 'color': 'SHAP interaction value'},
                         title=f'SHAP Interaction Values: {feature1} vs {feature2}')
        st.plotly_chart(fig)

# Conclusion
st.markdown("<p class='big-font'>Congratulations! 🎊</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>You've explored Explainable AI techniques for Decision Tree models. These methods help us understand and interpret our models, making AI more transparent and trustworthy. Keep exploring and applying these concepts to gain deeper insights into your models!</p>", unsafe_allow_html=True)