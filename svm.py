import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def plot_svm(X, y, svc):
    # Create a mesh grid for plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict the class for each point in the mesh grid
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create the plot
    fig = go.Figure(data=[
        go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='RdBu', showscale=False, opacity=0.8),
        go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y, colorscale='RdBu', size=10))
    ])

    fig.update_layout(title='Support Vector Machine Decision Boundary',
                      xaxis_title='Feature 1',
                      yaxis_title='Feature 2',
                      plot_bgcolor='white',
                      height=600,
                      width=800,
                      font=dict(size=14))

    return fig

def main():
    st.title("Support Vector Machines (SVM) Interactive Tool")

    # Explanation of SVM
    st.header("What are Support Vector Machines?")
    st.markdown("""
    Support Vector Machines (SVM) is a supervised machine learning algorithm used for classification and regression tasks. In the case of binary classification, SVM aims to find the optimal hyperplane that separates the two classes with the maximum margin.

    The key concepts in SVM are:
    - **Support Vectors**: The data points closest to the decision boundary. These points have the most influence on the position and orientation of the hyperplane.
    - **Hyperplane**: The decision boundary that separates the two classes. It is a line in 2D space or a plane in higher-dimensional space.
    - **Margin**: The distance between the hyperplane and the closest data points from each class. SVM tries to maximize this margin for better generalization.

    In this interactive tool, you can explore how SVM works on a synthetic dataset and see the impact of different kernel functions and regularization parameters.
    """)

    # Sidebar layout
    st.sidebar.title("Options")

    # Dataset selection
    dataset_options = ['Moons', 'Circles', 'Blobs']
    selected_dataset = st.sidebar.selectbox("Select a dataset", dataset_options)

    # Kernel selection
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    selected_kernel = st.sidebar.selectbox("Select a kernel function", kernel_options)

    # Regularization parameter (C)
    c_value = st.sidebar.slider("Regularization parameter (C)", min_value=0.1, max_value=10.0, step=0.1, value=1.0)

    # Generate the selected dataset
    if selected_dataset == 'Moons':
        X, y = datasets.make_moons(noise=0.3, random_state=0)
    elif selected_dataset == 'Circles':
        X, y = datasets.make_circles(noise=0.2, factor=0.5, random_state=0)
    else:
        X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the SVM model
    svc = SVC(kernel=selected_kernel, C=c_value)
    svc.fit(X_train_scaled, y_train)

    # Plot the decision boundary
    fig = plot_svm(X_train_scaled, y_train, svc)
    st.plotly_chart(fig)

    # Evaluate the model
    y_pred = svc.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Test Accuracy: {accuracy:.2f}")

    # Explanation of the plot
    st.header("Interpreting the Plot")
    st.markdown("""
    The plot shows the decision boundary learned by the SVM model on the selected dataset. The different colors represent the two classes, and the decision boundary is shown as a contour line separating the classes.

    - The data points are represented as scatter points, with their color indicating the true class label.
    - The support vectors, which are the data points closest to the decision boundary, have a significant impact on the position and orientation of the hyperplane.
    - The margin is the distance between the decision boundary and the closest data points from each class. SVM tries to maximize this margin for better generalization.

    You can experiment with different kernel functions and regularization parameters to see how they affect the decision boundary and the model's performance.
    """)

if __name__ == '__main__':
    main()