import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy import stats

# Set page config
st.set_page_config(layout="wide", page_title="Monte Carlo Simulations Explorer", page_icon="🎲")

# Custom CSS (unchanged, omitted for brevity)
st.markdown("""
<style>
    # ... (keep the existing CSS)
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🎲 Monte Carlo Simulations Explorer 🎲</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("<p class='big-font'>Welcome to the Monte Carlo Simulations Explorer!</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>Let's explore the power of Monte Carlo methods in solving various problems.</p>", unsafe_allow_html=True)

# Explanation
st.markdown("<p class='medium-font'>What are Monte Carlo Simulations?</p>", unsafe_allow_html=True)
st.markdown("""
<p class='small-font'>
Monte Carlo simulations are computational algorithms that rely on repeated random sampling to obtain numerical results. Key points:

- Used to solve problems that might be deterministic in principle
- Useful when it's infeasible or impossible to compute an exact result with a deterministic algorithm
- Commonly used in physics, finance, and other fields
- Accuracy generally improves with more samples, but at the cost of increased computation time

Monte Carlo methods are especially useful for simulating phenomena with significant uncertainty in inputs and systems with many coupled degrees of freedom.
</p>
""", unsafe_allow_html=True)

# Tabs with custom styling
tab1, tab2, tab3, tab4 = st.tabs(["π Estimation", "📊 Integration", "💹 Option Pricing", "🧠 Quiz"])

def estimate_pi(n_points):
    points = np.random.rand(n_points, 2)
    inside_circle = np.sum(np.sum(points**2, axis=1) <= 1)
    pi_estimate = 4 * inside_circle / n_points
    return pi_estimate, points

with tab1:
    st.markdown("<p class='medium-font'>Estimating π using Monte Carlo</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        We can estimate π by randomly placing points in a square and determining the ratio of points that fall within an inscribed circle.
        </p>
        """, unsafe_allow_html=True)

        n_points = st.slider("Number of points", 100, 100000, 1000, 100)
        
        if st.button("Run Simulation"):
            pi_estimate, points = estimate_pi(n_points)
            
            st.markdown(f"""
            <p class='small-font'>
            Estimated value of π: {pi_estimate:.6f}<br>
            Error: {abs(pi_estimate - np.pi):.6f}
            </p>
            """, unsafe_allow_html=True)

    with col2:
        if 'points' in locals():
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=points[:, 0], y=points[:, 1],
                mode='markers',
                marker=dict(
                    size=3,
                    color=np.sum(points**2, axis=1),
                    colorscale='Viridis',
                    showscale=True
                )
            ))

            fig.add_shape(type="circle",
                xref="x", yref="y",
                x0=0, y0=0, x1=1, y1=1,
                line_color="Red"
            )

            fig.update_layout(
                title='Monte Carlo Estimation of π',
                xaxis_title='X',
                yaxis_title='Y',
                width=600, height=600,
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )

            st.plotly_chart(fig)

def monte_carlo_integrate(func, a, b, n_points):
    x = np.random.uniform(a, b, n_points)
    y = func(x)
    integral = (b - a) * np.mean(y)
    return integral, x, y

with tab2:
    st.markdown("<p class='medium-font'>Monte Carlo Integration</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Monte Carlo integration is a technique for numerical integration using random numbers.
        </p>
        """, unsafe_allow_html=True)

        function = st.selectbox("Select a function", ["sin(x)", "x^2", "exp(-x^2)"])
        a = st.number_input("Lower bound", value=0.0)
        b = st.number_input("Upper bound", value=np.pi)
        n_points = st.slider("Number of points", 100, 100000, 10000, 100)

        if st.button("Perform Integration"):
            if function == "sin(x)":
                func = np.sin
                analytical = -np.cos(b) + np.cos(a)
            elif function == "x^2":
                func = lambda x: x**2
                analytical = (b**3 - a**3) / 3
            else:  # exp(-x^2)
                func = lambda x: np.exp(-x**2)
                analytical = np.sqrt(np.pi) / 2 * (erf(b) - erf(a))

            integral, x, y = monte_carlo_integrate(func, a, b, n_points)
            
            st.markdown(f"""
            <p class='small-font'>
            Monte Carlo Integration result: {integral:.6f}<br>
            Analytical result: {analytical:.6f}<br>
            Error: {abs(integral - analytical):.6f}
            </p>
            """, unsafe_allow_html=True)

    with col2:
        if 'x' in locals() and 'y' in locals():
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(size=3, color=y, colorscale='Viridis', showscale=True)
            ))

            x_range = np.linspace(a, b, 1000)
            fig.add_trace(go.Scatter(
                x=x_range, y=func(x_range),
                mode='lines', name='Function'
            ))

            fig.update_layout(
                title='Monte Carlo Integration',
                xaxis_title='X',
                yaxis_title='Y',
                width=600, height=600
            )

            st.plotly_chart(fig)

def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

def monte_carlo_option_pricing(S, K, T, r, sigma, n_simulations):
    dt = T / 252  # Assuming 252 trading days in a year
    nudt = (r - 0.5 * sigma ** 2) * dt
    sidt = sigma * np.sqrt(dt)
    
    Z = np.random.standard_normal((n_simulations, int(T * 252)))
    S_T = S * np.exp(np.cumsum(nudt + sidt * Z, axis=1))
    
    option_payoff = np.maximum(S_T[:, -1] - K, 0)
    option_price = np.exp(-r * T) * np.mean(option_payoff)
    
    return option_price, S_T

with tab3:
    st.markdown("<p class='medium-font'>Option Pricing with Monte Carlo</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <p class='small-font'>
        Monte Carlo simulations are widely used in finance for option pricing. Here, we'll price a European call option.
        </p>
        """, unsafe_allow_html=True)

        S = st.number_input("Current stock price", value=100.0, min_value=0.1)
        K = st.number_input("Strike price", value=100.0, min_value=0.1)
        T = st.number_input("Time to maturity (years)", value=1.0, min_value=0.1)
        r = st.number_input("Risk-free rate", value=0.05, min_value=0.0, max_value=1.0)
        sigma = st.number_input("Volatility", value=0.2, min_value=0.01, max_value=1.0)
        n_simulations = st.slider("Number of simulations", 100, 100000, 10000, 100)

        if st.button("Price Option"):
            mc_price, S_T = monte_carlo_option_pricing(S, K, T, r, sigma, n_simulations)
            bs_price = black_scholes(S, K, T, r, sigma)
            
            st.markdown(f"""
            <p class='small-font'>
            Monte Carlo option price: {mc_price:.4f}<br>
            Black-Scholes option price: {bs_price:.4f}<br>
            Difference: {abs(mc_price - bs_price):.4f}
            </p>
            """, unsafe_allow_html=True)

    with col2:
        if 'S_T' in locals():
            fig = go.Figure()
            for i in range(min(100, n_simulations)):  # Plot up to 100 paths
                fig.add_trace(go.Scatter(y=S_T[i], mode='lines', line=dict(width=1), showlegend=False))

            fig.update_layout(
                title='Stock Price Simulations',
                xaxis_title='Time Steps',
                yaxis_title='Stock Price',
                width=600, height=600
            )

            st.plotly_chart(fig)

with tab4:
    st.markdown("<p class='medium-font'>Test Your Knowledge!</p>", unsafe_allow_html=True)

    questions = [
        {
            "question": "What is the primary advantage of Monte Carlo methods?",
            "options": [
                "They always provide exact solutions",
                "They are faster than analytical methods",
                "They can handle complex problems with many variables",
                "They require less computational power"
            ],
            "correct": 2,
            "explanation": "Monte Carlo methods are particularly useful for complex problems with many variables, especially when analytical solutions are difficult or impossible to obtain."
        },
        {
            "question": "How does increasing the number of simulations typically affect Monte Carlo results?",
            "options": [
                "It always improves accuracy but increases computation time",
                "It always reduces accuracy but decreases computation time",
                "It has no effect on accuracy or computation time",
                "It improves accuracy up to a point, then has diminishing returns"
            ],
            "correct": 3,
            "explanation": "Increasing the number of simulations generally improves accuracy, but with diminishing returns. There's a trade-off between accuracy and computation time."
        },
        {
            "question": "In the context of option pricing, what does the Monte Carlo method simulate?",
            "options": [
                "The current stock price",
                "The option's strike price",
                "Possible paths of the stock price",
                "The risk-free interest rate"
            ],
            "correct": 2,
            "explanation": "In option pricing, Monte Carlo methods typically simulate many possible paths that the stock price could take up to the option's expiration date."
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
st.markdown("<p class='small-font'>You've explored Monte Carlo simulations through interactive examples in various fields. These powerful methods have wide-ranging applications in science, engineering, and finance. Keep exploring and applying these concepts to solve complex problems!</p>", unsafe_allow_html=True)