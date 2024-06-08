import streamlit as st
import numpy as np
import plotly.graph_objects as go

def kernel_transform(X):
    return np.array([[x[0], x[1], x[0]**2 + x[1]**2] for x in X])

def plot_data(X, y):
    fig = go.Figure()
    
    colors = ['green' if label == 1 else 'red' for label in y]
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=colors, size=10)))

    fig.update_layout(width=600, height=600, title='2D Data Points (Not Linearly Separable)')
    st.plotly_chart(fig)

def plot_3d_data(X, y):
    X_transformed = kernel_transform(X)
    
    fig = go.Figure()
    
    colors = ['green' if label == 1 else 'red' for label in y]
    fig.add_trace(go.Scatter3d(x=X_transformed[:, 0], y=X_transformed[:, 1], z=X_transformed[:, 2], 
                               mode='markers', marker=dict(color=colors, size=5)))

    fig.update_layout(width=800, height=800, title='3D Transformed Data Points (Linearly Separable)', 
                      scene=dict(xaxis_title='X1', yaxis_title='X2', zaxis_title='X1^2 + X2^2'))
    st.plotly_chart(fig)

def main():
    st.title('Support Vector Machines: The Kernel Trick')
    
    st.markdown("""
    - Not all data is linearly separable, which makes the job of an SVM difficult
    - The kernel trick offers an efficient and less expensive way to transform data from low dimensions to high dimensions
    - This is done by utilizing pairwise comparisons in the original data points
    """)
    
    np.random.seed(0)
    num_points = st.sidebar.slider('Number of data points:', 10, 100, 50, 10)
    
    X = np.random.randn(num_points, 2)
    X[:num_points//2, 1] += 3 * np.abs(X[:num_points//2, 0]) 
    X[num_points//2:, 1] -= 3 * np.abs(X[num_points//2:, 0])
    y = np.array([1] * (num_points//2) + [-1] * (num_points//2))
    
    st.markdown('### Original 2D Data')
    plot_data(X, y)
    
    st.markdown('### Transformed 3D Data')
    plot_3d_data(X, y)

if __name__ == '__main__':
    main()