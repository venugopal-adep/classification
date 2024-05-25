import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set page title
st.set_page_config(page_title="Decision Tree vs Random Forest")

# Title
st.title("Decision Tree vs Random Forest")
st.write('**Developed by : Venugopal Adep**')

# Description
st.write("This interactive application demonstrates the difference between decision trees and random forests using toy datasets.")

# Sidebar for user inputs
st.sidebar.title("Parameters")
dataset = st.sidebar.selectbox("Select dataset", ("Moons", "Circles", "Linearly Separable"))
n_samples = st.sidebar.slider("Number of samples", 50, 500, 200, 50)
noise = st.sidebar.slider("Noise", 0.0, 0.5, 0.2, 0.1)
max_depth = st.sidebar.slider("Maximum depth", 1, 10, 3, 1)
n_estimators = st.sidebar.slider("Number of trees (Random Forest)", 1, 100, 10, 1)

# Generate toy dataset based on user selection
if dataset == "Moons":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
elif dataset == "Circles":
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
else:
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2,
                               random_state=42, n_clusters_per_class=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create decision tree classifier
dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
dt.fit(X_train, y_train)

# Create random forest classifier
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)

# Calculate accuracies
dt_accuracy = accuracy_score(y_test, dt_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Plot the decision boundaries
fig = go.Figure()

# Add scatter points for the training data
fig.add_trace(go.Scatter(x=X_train[:, 0], y=X_train[:, 1], mode='markers', marker=dict(color=y_train, colorscale='Viridis'), name='Training Data'))

# Add scatter points for the testing data
fig.add_trace(go.Scatter(x=X_test[:, 0], y=X_test[:, 1], mode='markers', marker=dict(color=y_test, colorscale='Viridis'), name='Testing Data'))

# Create a mesh grid for decision boundary visualization
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Make predictions on the mesh grid
Z_dt = dt.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_rf = rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Add contour plot for decision tree decision boundary
fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02), z=Z_dt, name='Decision Tree', showscale=False, colorscale='Viridis'))

# Add contour plot for random forest decision boundary
fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02), z=Z_rf, name='Random Forest', showscale=False, colorscale='RdBu'))

# Update layout
fig.update_layout(title=f'Decision Tree vs Random Forest - {dataset} Dataset',
                  xaxis_title='Feature 1', yaxis_title='Feature 2')

# Display the plot
st.plotly_chart(fig)

# Display accuracies
st.write(f"Decision Tree Accuracy: {dt_accuracy:.3f}")
st.write(f"Random Forest Accuracy: {rf_accuracy:.3f}")

# Explain the difference between decision trees and random forests
st.header("Decision Tree vs Random Forest")
st.write("Decision trees and random forests are both popular algorithms for classification and regression tasks. Here are the key differences between them:")

st.subheader("Decision Tree")
st.write("- A decision tree is a single tree-like model that makes decisions based on a series of rules.")
st.write("- It recursively splits the data based on the most informative features until a stopping criterion is met.")
st.write("- Decision trees are easy to interpret and visualize, but they can be prone to overfitting.")

st.subheader("Random Forest")
st.write("- A random forest is an ensemble of multiple decision trees.")
st.write("- It combines the predictions of multiple trees to make the final prediction, which reduces overfitting and improves generalization.")
st.write("- Random forests introduce randomness by using a random subset of features and samples for each tree.")
st.write("- They are more robust and accurate than individual decision trees but less interpretable.")

st.write("In general, random forests tend to perform better than decision trees, especially when dealing with complex datasets. However, decision trees are still useful for their simplicity and interpretability.")
