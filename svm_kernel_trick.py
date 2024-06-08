import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def kernel_transform(X):
    return np.array([[x[0], x[1], x[0]**2 + x[1]**2] for x in X])

def plot_data(X, y, title):
    fig = go.Figure()
    
    colors = ['green' if label == 1 else 'red' for label in y]
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=colors, size=10)))

    fig.update_layout(width=600, height=600, title=title)
    st.plotly_chart(fig)

def plot_3d_data(X, y, title):
    X_transformed = kernel_transform(X)
    
    fig = go.Figure()
    
    colors = ['green' if label == 1 else 'red' for label in y]
    fig.add_trace(go.Scatter3d(x=X_transformed[:, 0], y=X_transformed[:, 1], z=X_transformed[:, 2], 
                               mode='markers', marker=dict(color=colors, size=5)))

    fig.update_layout(width=800, height=800, title=title, 
                      scene=dict(xaxis_title='X1', yaxis_title='X2', zaxis_title='X1^2 + X2^2'))
    st.plotly_chart(fig)

def generate_data(num_points):
    X = np.random.randn(num_points, 2)
    y = np.array([1 if np.linalg.norm(x) < 1 else -1 for x in X])
    return X, y

def main():
    st.title('Support Vector Machines: The Kernel Trick')
    st.write('**Developed by : Venugopal Adep**')
    
    st.markdown("""
    - Not all data is linearly separable, which makes the job of an SVM difficult
    - The kernel trick offers an efficient and less expensive way to transform data from low dimensions to high dimensions
    - This is done by utilizing pairwise comparisons in the original data points
    """)
    
    np.random.seed(0)
    num_points = st.sidebar.slider('Number of data points:', 10, 100, 50, 10)
    test_size = st.sidebar.slider('Test data size (%):', 10, 50, 20, 5)
    
    X, y = generate_data(num_points)
    
    test_size = int(num_points * test_size / 100)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    st.markdown('### Original 2D Data')
    plot_data(X, y, '2D Data Points (Not Linearly Separable)')
    
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(X_train, y_train)
    y_pred_linear = svm_linear.predict(X_test)
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    
    svm_rbf = SVC(kernel='rbf')
    svm_rbf.fit(X_train, y_train)
    y_pred_rbf = svm_rbf.predict(X_test)
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
    
    st.markdown('### Prediction Accuracy')
    st.write(f'Linear SVM Accuracy: {accuracy_linear:.2f}')
    st.write(f'RBF Kernel SVM Accuracy: {accuracy_rbf:.2f}')
    
    st.markdown('### Transformed 3D Data')
    plot_3d_data(X, y, '3D Transformed Data Points (Linearly Separable)')

if __name__ == '__main__':
    main()
