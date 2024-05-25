import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def plot_decision_boundary(clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = go.Figure(data=go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='RdBu', showscale=False))
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y, colorscale='RdBu', showscale=False)))
    fig.update_layout(title=title, xaxis_title='Feature 1', yaxis_title='Feature 2', width=600, height=500)
    return fig

def main():
    st.title("Bagging and Random Forest Demo")
    st.write('**Developed by : Venugopal Adep**')
    
    st.sidebar.header("Interactive Demo")
    n_estimators = st.sidebar.slider("Number of Estimators", min_value=1, max_value=100, value=10, step=1)
    
    st.header("Explanation")
    st.markdown("""
    Bagging (Bootstrap Aggregating) and Random Forest are ensemble learning methods that combine multiple decision trees to improve prediction accuracy and reduce overfitting.
    
    **Bagging:**
    - Bagging creates multiple subsets of the training data by randomly sampling with replacement (bootstrap samples).
    - It trains a decision tree on each subset independently.
    - The final prediction is obtained by aggregating the predictions of all the trees (majority voting for classification, averaging for regression).
    
    **Random Forest:**
    - Random Forest is an extension of the bagging technique.
    - In addition to bootstrapping the data, it also selects a random subset of features at each split in the decision trees.
    - This further reduces the correlation between the trees and improves the diversity of the ensemble.
    
    In this demo, you can observe the decision boundaries of Bagging and Random Forest classifiers on a toy dataset. Adjust the number of estimators (trees) and see how the decision boundary changes.
    """)
    
    X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    bagging = BaggingClassifier(n_estimators=n_estimators, random_state=42)
    bagging.fit(X_train, y_train)
    bagging_accuracy = accuracy_score(y_test, bagging.predict(X_test))
    
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    rf_accuracy = accuracy_score(y_test, rf.predict(X_test))
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_decision_boundary(bagging, X, y, f"Bagging (Accuracy: {bagging_accuracy:.2f})"), use_container_width=True)
    with col2:
        st.plotly_chart(plot_decision_boundary(rf, X, y, f"Random Forest (Accuracy: {rf_accuracy:.2f})"), use_container_width=True)
    
    st.header("Interpretation")
    st.markdown("""
    - The decision boundaries show how the classifiers separate the two classes (represented by different colors) in the feature space.
    - A more complex and irregular decision boundary indicates a more flexible and potentially overfitting model.
    - As you increase the number of estimators, the decision boundaries become smoother and more stable.
    - Random Forest often performs better than Bagging due to the additional randomness introduced by feature subsampling.
    - The accuracy scores provide a quantitative measure of how well the classifiers perform on the test set.
    """)

if __name__ == '__main__':
    main()
