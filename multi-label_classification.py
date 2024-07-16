import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss, multilabel_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
import urllib.request

# Set page config
st.set_page_config(layout="wide", page_title="Multi-Label Random Forest Explorer", page_icon="🌳")

# Custom CSS for improved visual appeal
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stSelectbox {
        background-color: #e0e0e0;
    }
    .stSlider {
        background-color: #e0e0e0;
    }
    h1 {
        color: #2C3E50;
        text-align: center;
        font-family: 'Helvetica', sans-serif;
    }
    h2 {
        color: #34495E;
        font-family: 'Helvetica', sans-serif;
    }
    p {
        font-family: 'Arial', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🌳 Multi-Label Random Forest Explorer 🌳")

# Introduction
st.markdown("""
Welcome to the Multi-Label Random Forest Explorer! This interactive app allows you to explore 
the power of Random Forest for multi-label classification tasks using the Yeast dataset.
""")

# Load Yeast dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
    data = pd.read_csv(url, sep='\s+', header=None)
    data.columns = ['sequence_name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'class']
    
    # One-hot encode the 'class' column
    class_dummies = pd.get_dummies(data['class'], prefix='class')
    
    X = data.iloc[:, 1:9]
    y = class_dummies
    
    return X, y

# Sidebar
st.sidebar.header("Model Configuration")
n_estimators = st.sidebar.slider("Number of trees", 10, 200, 100, 10)
max_depth = st.sidebar.slider("Maximum tree depth", 1, 20, 10)
min_samples_split = st.sidebar.slider("Minimum samples to split", 2, 20, 2)
test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2, 0.05)

# Main content
tab1, tab2, tab3 = st.tabs(["Model Training", "Feature Importance", "Class Distribution"])

with tab1:
    st.header("Random Forest Model Training")
    
    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                    min_samples_split=min_samples_split, random_state=42)
        model = rf  # No need for MultiOutputClassifier for multi-class problem
        model.fit(X_train_scaled, y_train)
        
        train_preds = model.predict(X_train_scaled)
        test_preds = model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, train_preds)
        test_accuracy = accuracy_score(y_test, test_preds)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Train Accuracy", f"{train_accuracy:.4f}")
        with col2:
            st.metric("Test Accuracy", f"{test_accuracy:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test.idxmax(axis=1), pd.DataFrame(test_preds, columns=y.columns).idxmax(axis=1))
        fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual"), x=y.columns, y=y.columns,
                        title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Feature Importance")
    
    if st.button("Calculate Feature Importance"):
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                    min_samples_split=min_samples_split, random_state=42)
        rf.fit(X, y.idxmax(axis=1))  # Use the class with highest probability
        
        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                     title='Feature Importance', labels={'importance': 'Importance', 'feature': 'Feature'})
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Class Distribution")
    
    class_dist = y.sum().sort_values(ascending=False)
    fig = px.bar(x=class_dist.index, y=class_dist.values, 
                 labels={'x': 'Class', 'y': 'Count'},
                 title='Distribution of Classes in the Yeast Dataset')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    The Yeast dataset contains information about protein localization sites. Each instance belongs to one of several possible classes, 
    representing different localization sites. The chart above shows the distribution of these classes in the dataset.
    """)

# Conclusion
st.markdown("""
## Conclusion

Congratulations! 🎊 You've explored multi-label classification using Random Forest on the Yeast dataset.

Key takeaways:
1. Random Forest can be adapted for multi-label tasks using the MultiOutputClassifier.
2. Feature importance helps identify which attributes are most crucial for prediction.
3. The Yeast dataset has an imbalanced class distribution, which is common in real-world multi-label problems.

Keep exploring and applying these concepts to solve real-world multi-label classification problems!

### Additional Resources
- [Scikit-learn Random Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Multi-label Classification with Scikit-learn](https://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification)
- [UCI Machine Learning Repository: Yeast Dataset](https://archive.ics.uci.edu/ml/datasets/Yeast)

Happy learning and coding! 🌳📊🧠
""")