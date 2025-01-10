import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA

# Title and Description
st.set_page_config(layout="wide")  # Set full-width layout
st.title("Kernel Trick Visualization 🌟")
st.markdown("**Developed by : Venugopal Adep**")
st.markdown("""
This application demonstrates the **Kernel Trick** with side-by-side 2D and 3D visualizations. 
The left plot shows the original dataset in 2D, while the right plot visualizes the kernel-transformed data in 3D.
""")

# Sidebar for Dataset Selection and Kernel Parameters
st.sidebar.header("Options")
dataset = st.sidebar.selectbox(
    "Select Dataset",
    options=["Concentric Circles", "Moons", "Blobs"],
    index=0,
)
gamma = st.sidebar.slider("Gamma (RBF Kernel Parameter)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

# Generate Dataset based on Selection
if dataset == "Concentric Circles":
    X, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
elif dataset == "Moons":
    X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
else:
    X, y = make_blobs(n_samples=300, centers=2, cluster_std=1.5, random_state=42)

# Compute RBF Kernel Transformation
rbf_features = rbf_kernel(X, gamma=gamma)

# Reduce RBF features to 3 dimensions for visualization using PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(rbf_features)

# Prepare DataFrames for Plotly
df_2d = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})
df_3d = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "PC3": X_pca[:, 2], "label": y})

# Display Kernel Function
st.subheader("Kernel Function")
st.latex(r"K(x_i, x_j) = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)")
st.markdown("""
The **Radial Basis Function (RBF)** kernel computes similarity between points in a higher-dimensional space 
without explicitly transforming them. The parameter \( \sigma \) controls how far the influence of a single training example reaches.
""")

# Create 2D Plot (Original Dataset)
fig_2d = go.Figure()
for label in np.unique(y):
    subset = df_2d[df_2d["label"] == label]
    fig_2d.add_trace(go.Scatter(
        x=subset["x1"], y=subset["x2"], mode="markers",
        name=f"Class {label}", marker=dict(size=6)
    ))
fig_2d.update_layout(
    title="Original Dataset (2D)",
    xaxis_title="X1", yaxis_title="X2",
    template="plotly", legend_title="Classes"
)

# Create 3D Plot (Kernel-Transformed Data)
fig_3d = go.Figure()
for label in np.unique(y):
    subset = df_3d[df_3d["label"] == label]
    fig_3d.add_trace(go.Scatter3d(
        x=subset["PC1"], y=subset["PC2"], z=subset["PC3"],
        mode="markers", name=f"Class {label}",
        marker=dict(size=4)
    ))
fig_3d.update_layout(
    title="Kernel-Transformed Dataset (3D)",
    scene=dict(
        xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"
    ),
    template="plotly", legend_title="Classes"
)

# Display Side-by-Side Plots in Streamlit
col1, col2 = st.columns(2)  # Create two columns for side-by-side plots

with col1:
    st.plotly_chart(fig_2d, use_container_width=True)
with col2:
    st.plotly_chart(fig_3d, use_container_width=True)

# Explanation Section
st.subheader("How It Works")
st.markdown("""
- The **left plot** shows the original dataset in its raw form (non-linearly separable).
- The **right plot** demonstrates how the RBF kernel maps the data into a higher-dimensional space where it becomes linearly separable.
- The kernel trick avoids explicitly computing the high-dimensional transformation by using a similarity function.
""")


