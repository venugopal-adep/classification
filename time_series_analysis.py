import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from math import sqrt

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    data = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
    return data

def main():
    st.set_page_config(page_title="Time Series Analysis Demo", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
    .stButton>button {
        background-color: #ff6e40;
        color: white !important;
    }
    .stButton>button:hover {
        background-color: #ff9e80;
        color: white !important;
    }
    .info-box {
        background-color: #e6f3ff;
        border-left: 5px solid #3366cc;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff6e40;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📈 Time Series Analysis")
    
    tabs = st.tabs(["📚 Introduction", "🔍 Exploration", "📊 Decomposition", "🔮 Forecasting", "🧠 Quiz"])
    
    with tabs[0]:
        introduction_section()
    
    with tabs[1]:
        exploration_section()
    
    with tabs[2]:
        decomposition_section()
    
    with tabs[3]:
        forecasting_section()
    
    with tabs[4]:
        quiz_section()

def introduction_section():
    st.header("Introduction to Time Series Analysis")
    
    st.markdown("""
    <div class="info-box">
    Time Series Analysis is a specific way of analyzing a sequence of data points collected over an interval of time. 
    In Time Series Analysis, analysts record data points at consistent intervals over a set period of time rather than just recording the data points intermittently or randomly.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Key Concepts")
    concepts = {
        "Trend": "The overall direction of the series, whether it's increasing, decreasing, or staying flat",
        "Seasonality": "Regular and predictable patterns that repeat over a specific time period",
        "Cyclicity": "Regular but non-periodic fluctuations",
        "Irregularity": "Random, unpredictable fluctuations in the data",
        "Stationarity": "Statistical properties like mean, variance, and autocorrelation are constant over time"
    }

    for concept, description in concepts.items():
        st.markdown(f"**{concept}**: {description}")

    st.subheader("Importance of Time Series Analysis")
    st.markdown("""
    - Helps in understanding past patterns and behaviors
    - Allows for forecasting future values
    - Crucial for decision making in various fields like finance, economics, and weather forecasting
    - Enables detection of anomalies or outliers in the data
    - Facilitates the identification of cyclical trends and seasonal variations
    """)

    st.subheader("Example Applications")
    st.markdown("""
    1. **Stock Market Analysis**: Predicting future stock prices based on historical data
    2. **Weather Forecasting**: Analyzing temperature and precipitation patterns over time
    3. **Sales Forecasting**: Predicting future sales based on historical sales data
    4. **Economic Indicators**: Analyzing GDP, inflation rates, or unemployment rates over time
    5. **Energy Consumption**: Predicting electricity or gas consumption patterns
    """)

def exploration_section():
    st.header("Data Exploration")

    data = load_data()

    st.subheader("Data Overview")
    st.write(data.head())
    st.write(f"Dataset shape: {data.shape}")

    # Time Series Plot
    st.subheader("Time Series Plot")
    fig = px.line(data, x=data.index, y='Temp', title='Daily Minimum Temperatures in Melbourne')
    st.plotly_chart(fig)

    st.markdown("""
    This plot shows the daily minimum temperatures in Melbourne over time. 
    Look for any visible patterns, trends, or seasonality in the data.
    """)

    # Summary Statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Check for stationarity
    st.subheader("Stationarity Check")
    st.markdown("""
    <div class="info-box">
    A stationary time series is one whose statistical properties such as mean, variance, autocorrelation, etc. 
    are all constant over time. Most statistical forecasting methods are based on the assumption that the 
    time series can be rendered approximately stationary through the use of mathematical transformations.
    </div>
    """, unsafe_allow_html=True)

    result = adfuller(data['Temp'])
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])
    st.write('Critical Values:')
    for key, value in result[4].items():
        st.write(f'{key}: {value}')

    if result[1] <= 0.05:
        st.write("The series is stationary")
    else:
        st.write("The series is not stationary")

def decomposition_section():
    st.header("Time Series Decomposition")

    data = load_data()

    st.markdown("""
    <div class="info-box">
    Time series decomposition involves breaking down a time series into its constituent components. 
    These components are typically: Trend, Seasonal, and Residual.
    </div>
    """, unsafe_allow_html=True)

    # Decompose the time series
    decomposition = seasonal_decompose(data['Temp'], model='additive', period=365)

    # Plot the decomposition
    st.subheader("Time Series Decomposition")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    - **Observed**: The original time series
    - **Trend**: The overall direction of the series
    - **Seasonal**: Regular patterns of ups and downs
    - **Residual**: What's left after removing trend and seasonality
    """)

    # ACF and PACF plots
    st.subheader("ACF and PACF Plots")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(data['Temp'], ax=ax1)
    plot_pacf(data['Temp'], ax=ax2)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    - **ACF (Autocorrelation Function)**: Shows the correlation of the series with itself at different lags
    - **PACF (Partial Autocorrelation Function)**: Shows the direct correlation between an observation and its lag
    These plots help in determining the order of ARIMA models.
    """)

def forecasting_section():
    st.header("Time Series Forecasting")

    data = load_data()

    st.markdown("""
    <div class="info-box">
    Time series forecasting uses historical data to predict future values. 
    We'll use the ARIMA (AutoRegressive Integrated Moving Average) model for this demonstration.
    </div>
    """, unsafe_allow_html=True)

    # Split the data
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    # Fit ARIMA model
    model = ARIMA(train['Temp'], order=(1,1,1))
    results = model.fit()

    # Make predictions
    predictions = results.forecast(len(test))

    # Plot the results
    st.subheader("ARIMA Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Temp'], name='Train'))
    fig.add_trace(go.Scatter(x=test.index, y=test['Temp'], name='Test'))
    fig.add_trace(go.Scatter(x=test.index, y=predictions, name='Forecast'))
    fig.update_layout(title='ARIMA Forecast', xaxis_title='Date', yaxis_title='Temperature')
    st.plotly_chart(fig)

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(test['Temp'], predictions))
    st.write(f'Root Mean Squared Error: {rmse}')

    st.markdown("""
    The RMSE (Root Mean Squared Error) is a measure of the differences between predicted values and observed values. 
    A lower RMSE indicates better fit of the model to the data.
    """)

def quiz_section():
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main goal of Time Series Analysis?",
            "options": [
                "To predict stock prices",
                "To analyze data points collected at consistent time intervals",
                "To manage databases",
                "To design user interfaces"
            ],
            "correct": "To analyze data points collected at consistent time intervals",
            "explanation": "Time Series Analysis is specifically used to analyze sequences of data points collected at consistent time intervals, allowing for trend analysis, forecasting, and more."
        },
        {
            "question": "What does 'seasonality' refer to in Time Series Analysis?",
            "options": [
                "The overall increasing or decreasing pattern in the data",
                "Random fluctuations in the data",
                "Regular and predictable patterns that repeat over a specific time period",
                "The difference between summer and winter temperatures"
            ],
            "correct": "Regular and predictable patterns that repeat over a specific time period",
            "explanation": "In Time Series Analysis, seasonality refers to regular and predictable patterns that repeat over a specific time period, such as daily, monthly, or yearly cycles."
        },
        {
            "question": "What is the purpose of the Augmented Dickey-Fuller test in Time Series Analysis?",
            "options": [
                "To decompose the time series",
                "To check for stationarity in the time series",
                "To forecast future values",
                "To calculate the moving average"
            ],
            "correct": "To check for stationarity in the time series",
            "explanation": "The Augmented Dickey-Fuller test is used to check for stationarity in a time series. Stationarity is an important property for many time series modeling techniques."
        },
        {
            "question": "What does ARIMA stand for in the context of Time Series Analysis?",
            "options": [
                "Automated Regression In Moving Averages",
                "AutoRegressive Integrated Moving Average",
                "Average Rate In Multivariate Analysis",
                "Accelerated Reasoning In Mathematical Applications"
            ],
            "correct": "AutoRegressive Integrated Moving Average",
            "explanation": "ARIMA stands for AutoRegressive Integrated Moving Average. It's a popular model used for time series forecasting that combines autoregression, differencing, and moving average components."
        }
    ]
    
    for i, q in enumerate(questions, 1):
        st.subheader(f"Question {i}")
        with st.container():
            st.write(q["question"])
            answer = st.radio("Select your answer:", q["options"], key=f"q{i}")
            if st.button("Check Answer", key=f"check{i}"):
                if answer == q["correct"]:
                    st.success("Correct! 🎉")
                else:
                    st.error(f"Incorrect. The correct answer is: {q['correct']}")
                st.info(f"Explanation: {q['explanation']}")
            st.write("---")

if __name__ == "__main__":
    main()