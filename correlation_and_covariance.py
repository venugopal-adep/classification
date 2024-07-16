import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy import stats

# Set page config
st.set_page_config(layout="wide", page_title="Correlation and Covariance Explorer", page_icon="📊")

# Custom CSS (unchanged, omitted for brevity)
st.markdown("""
<style>
    # ... (keep the existing CSS)
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>📊 Correlation and Covariance Explorer 📊</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("<p class='big-font'>Welcome to the Correlation and Covariance Explorer!</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>Let's explore how correlation and covariance measure the relationship between two variables.</p>", unsafe_allow_html=True)

# Explanation
st.markdown("<p class='medium-font'>What are Correlation and Covariance?</p>", unsafe_allow_html=True)
st.markdown("""
<p class='small-font'>
Correlation and covariance are statistical measures that describe the relationship between two variables:

- Covariance measures how two variables change together, but its magnitude is unbounded.
- Correlation is a standardized version of covariance, always ranging from -1 to 1.
- Both measure linear relationships between variables.
- Correlation of 1 or -1 indicates a perfect linear relationship, while 0 indicates no linear relationship.

Key formulas:
- Covariance: Cov(X,Y) = E[(X - μ_X)(Y - μ_Y)]
- Pearson Correlation: ρ = Cov(X,Y) / (σ_X * σ_Y)

Where E is the expected value, μ is the mean, and σ is the standard deviation.
</p>
""", unsafe_allow_html=True)

# Tabs with custom styling
tab1, tab2, tab3, tab4 = st.tabs(["📊 Scatter Plot", "🔄 Correlation vs Covariance", "🎲 Interactive Simulation", "🧠 Quiz"])

with tab1:
    st.markdown("<p class='medium-font'>Visualizing Correlation with Scatter Plots</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Adjust the correlation to see how it affects the scatter plot of bivariate data.
        </p>
        """, unsafe_allow_html=True)

        correlation = st.slider("Correlation", -1.0, 1.0, 0.0, 0.1)
        num_points = st.slider("Number of points", 100, 1000, 500, 50)

        # Generate correlated data
        mean = [0, 0]
        cov = [[1, correlation], [correlation, 1]]
        x, y = np.random.multivariate_normal(mean, cov, num_points).T

        st.markdown(f"""
        <p class='small-font'>
        Pearson Correlation: {np.corrcoef(x, y)[0, 1]:.4f}<br>
        Covariance: {np.cov(x, y)[0, 1]:.4f}
        </p>
        """, unsafe_allow_html=True)

    with col2:
        fig = go.Figure(data=go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=5,
                color=y,
                colorscale='Viridis',
                showscale=True
            )
        ))

        fig.update_layout(
            title='Scatter Plot of Correlated Variables',
            xaxis_title='X',
            yaxis_title='Y',
            width=600,
            height=600
        )

        st.plotly_chart(fig)

with tab2:
    st.markdown("<p class='medium-font'>Correlation vs Covariance</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Compare how correlation and covariance change as you adjust the relationship between variables and their scales.
        </p>
        """, unsafe_allow_html=True)

        correlation = st.slider("Correlation", -1.0, 1.0, 0.0, 0.1, key="corr_vs_cov_corr")
        scale_x = st.slider("Scale of X", 0.1, 10.0, 1.0, 0.1)
        scale_y = st.slider("Scale of Y", 0.1, 10.0, 1.0, 0.1)

        # Generate correlated data
        mean = [0, 0]
        cov = [[scale_x**2, correlation*scale_x*scale_y], [correlation*scale_x*scale_y, scale_y**2]]
        x, y = np.random.multivariate_normal(mean, cov, 1000).T

        correlation_computed = np.corrcoef(x, y)[0, 1]
        covariance_computed = np.cov(x, y)[0, 1]

        st.markdown(f"""
        <p class='small-font'>
        Pearson Correlation: {correlation_computed:.4f}<br>
        Covariance: {covariance_computed:.4f}
        </p>
        """, unsafe_allow_html=True)

    with col2:
        fig = go.Figure(data=go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=5,
                color=y,
                colorscale='Viridis',
                showscale=True
            )
        ))

        fig.update_layout(
            title='Scatter Plot with Adjusted Scales',
            xaxis_title='X',
            yaxis_title='Y',
            width=600,
            height=600
        )

        st.plotly_chart(fig)

with tab3:
    st.markdown("<p class='medium-font'>Interactive Correlation and Covariance Simulation</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Generate samples and see how sample correlation and covariance compare to population parameters.
        </p>
        """, unsafe_allow_html=True)

        num_samples = st.slider("Number of samples", 10, 1000, 100, 10)
        true_correlation = st.slider("True Correlation", -1.0, 1.0, 0.0, 0.1, key="sim_corr")
        scale_x = st.slider("Scale of X", 0.1, 10.0, 1.0, 0.1, key="sim_scale_x")
        scale_y = st.slider("Scale of Y", 0.1, 10.0, 1.0, 0.1, key="sim_scale_y")

        if st.button("Run Simulation"):
            mean = [0, 0]
            cov = [[scale_x**2, true_correlation*scale_x*scale_y], [true_correlation*scale_x*scale_y, scale_y**2]]
            samples = np.random.multivariate_normal(mean, cov, num_samples)

            sample_correlation = np.corrcoef(samples.T)[0, 1]
            sample_covariance = np.cov(samples.T)[0, 1]
            true_covariance = cov[0][1]

            st.markdown(f"""
            <p class='small-font'>
            True Correlation: {true_correlation:.4f}<br>
            Sample Correlation: {sample_correlation:.4f}<br>
            True Covariance: {true_covariance:.4f}<br>
            Sample Covariance: {sample_covariance:.4f}
            </p>
            """, unsafe_allow_html=True)

    with col2:
        if 'samples' in locals():
            fig = go.Figure(data=go.Scatter(
                x=samples[:, 0],
                y=samples[:, 1],
                mode='markers',
                marker=dict(
                    size=5,
                    color=samples[:, 1],
                    colorscale='Viridis',
                    showscale=True
                )
            ))

            fig.update_layout(
                title='Scatter Plot of Simulated Samples',
                xaxis_title='X',
                yaxis_title='Y',
                width=600,
                height=600
            )

            st.plotly_chart(fig)

with tab4:
    st.markdown("<p class='medium-font'>Test Your Knowledge!</p>", unsafe_allow_html=True)

    questions = [
        {
            "question": "What is the main difference between correlation and covariance?",
            "options": [
                "Correlation measures linear relationships, covariance measures non-linear relationships",
                "Correlation is bounded between -1 and 1, covariance is unbounded",
                "Correlation is always positive, covariance can be negative",
                "Covariance is easier to interpret than correlation"
            ],
            "correct": 1,
            "explanation": "The main difference is that correlation is standardized and always ranges from -1 to 1, while covariance is unbounded. This makes correlation easier to interpret across different scales."
        },
        {
            "question": "What does a correlation of 0 indicate?",
            "options": [
                "A perfect positive linear relationship",
                "A perfect negative linear relationship",
                "No linear relationship",
                "The variables are identical"
            ],
            "correct": 2,
            "explanation": "A correlation of 0 indicates that there is no linear relationship between the variables. However, there might still be a non-linear relationship."
        },
        {
            "question": "How does changing the scale of variables affect correlation and covariance?",
            "options": [
                "It affects both correlation and covariance equally",
                "It affects covariance but not correlation",
                "It affects correlation but not covariance",
                "It has no effect on either correlation or covariance"
            ],
            "correct": 1,
            "explanation": "Changing the scale of variables affects covariance but not correlation. This is because correlation is standardized by the standard deviations of the variables, making it scale-invariant."
        }
    ]

    score = 0
    for i, q in enumerate(questions):
        st.markdown(f"<p class='small-font'><strong>Question {i+1}:</strong> {q['question']}</p>", unsafe_allow_html=True)
        user_answer = st.radio("Select your answer:", q['options'], key=f"q{i}")
        
        if st.button("Check Answer", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! 🎉")
                score += 1
            else:
                st.error("Incorrect. Try again! 🤔")
            st.info(q['explanation'])
        st.markdown("---")

    if st.button("Show Final Score"):
        st.markdown(f"<p class='big-font'>Your score: {score}/{len(questions)}</p>", unsafe_allow_html=True)
        if score == len(questions):
            st.balloons()

# Conclusion
st.markdown("<p class='big-font'>Congratulations! 🎊</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>You've explored Correlation and Covariance through interactive examples and simulations. These concepts are crucial in understanding relationships between variables in many fields, including statistics, data science, and machine learning. Keep exploring and applying these concepts in various scenarios!</p>", unsafe_allow_html=True)