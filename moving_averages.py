import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

@st.cache_data
def load_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5 years of data
    data = yf.download('^GSPC', start=start_date, end=end_date)
    return data

def calculate_ma(data, window):
    return data['Close'].rolling(window=window).mean()

def main():
    st.set_page_config(page_title="Moving Averages Analysis Demo", layout="wide")
    
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
    
    st.title("📈 Moving Averages Analysis")
    
    tabs = st.tabs(["📚 Introduction", "🔍 Exploration", "📊 Visualization", "💡 Applications", "🧠 Quiz"])
    
    with tabs[0]:
        introduction_section()
    
    with tabs[1]:
        exploration_section()
    
    with tabs[2]:
        visualization_section()
    
    with tabs[3]:
        applications_section()
    
    with tabs[4]:
        quiz_section()

def introduction_section():
    st.header("Introduction to Moving Averages")
    
    st.markdown("""
    <div class="info-box">
    A Moving Average (MA) is a widely used indicator in technical analysis that helps smooth out price action 
    by filtering out the "noise" from random short-term price fluctuations. It is a trend-following, or lagging, 
    indicator because it is based on past prices.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Key Concepts")
    concepts = {
        "Simple Moving Average (SMA)": "The unweighted mean of the previous n data points",
        "Exponential Moving Average (EMA)": "A type of moving average that gives more weight to recent prices",
        "Crossovers": "When a shorter-term MA crosses a longer-term MA, potentially signaling a trend change",
        "Support and Resistance": "MAs can act as support (price floor) or resistance (price ceiling)",
        "Trend Identification": "The direction of an MA can help identify the overall trend"
    }

    for concept, description in concepts.items():
        st.markdown(f"**{concept}**: {description}")

    st.subheader("Importance of Moving Averages")
    st.markdown("""
    - Help identify trend direction
    - Smooth out price action and filter out noise
    - Can be used to identify support and resistance levels
    - Useful in various trading strategies
    - Help in understanding the overall market sentiment
    """)

def exploration_section():
    st.header("Data Exploration")

    data = load_data()

    st.subheader("S&P 500 Data Overview")
    st.write(data.head())
    st.write(f"Dataset shape: {data.shape}")

    # Basic statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Closing price plot
    st.subheader("S&P 500 Closing Price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title='S&P 500 Closing Price', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    st.markdown("""
    This plot shows the daily closing price of the S&P 500 index. 
    Observe the overall trend and any significant price movements.
    """)

def visualization_section():
    st.header("Moving Averages Visualization")

    data = load_data()

    st.markdown("""
    <div class="info-box">
    Moving averages help smooth out price action by filtering out the noise from random price fluctuations. 
    Let's visualize different moving averages and see how they interact with the price.
    </div>
    """, unsafe_allow_html=True)

    # Allow user to select MA periods
    ma_short = st.slider("Select Short-term MA period", 5, 50, 20)
    ma_long = st.slider("Select Long-term MA period", 20, 200, 50)

    # Calculate MAs
    data['MA_Short'] = calculate_ma(data, ma_short)
    data['MA_Long'] = calculate_ma(data, ma_long)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_Short'], mode='lines', name=f'{ma_short}-day MA'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_Long'], mode='lines', name=f'{ma_long}-day MA'))
    fig.update_layout(title='S&P 500 with Moving Averages', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    st.markdown(f"""
    This chart shows the S&P 500 closing price along with a {ma_short}-day and {ma_long}-day moving average.
    - The shorter MA ({ma_short}-day) is more responsive to recent price changes.
    - The longer MA ({ma_long}-day) shows the longer-term trend.
    - Crossovers between these MAs can potentially signal trend changes.
    """)

def applications_section():
    st.header("Applications of Moving Averages")

    data = load_data()

    st.markdown("""
    <div class="info-box">
    Moving averages have various applications in technical analysis and trading. 
    Let's explore some common use cases.
    </div>
    """, unsafe_allow_html=True)

    # Trend Identification
    st.subheader("Trend Identification")
    ma_trend = st.slider("Select MA period for trend identification", 20, 200, 50)
    data['MA_Trend'] = calculate_ma(data, ma_trend)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_Trend'], mode='lines', name=f'{ma_trend}-day MA'))
    fig.update_layout(title='Trend Identification with Moving Average', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    st.markdown(f"""
    The {ma_trend}-day moving average helps identify the overall trend:
    - When the price is above the MA, it generally indicates an uptrend.
    - When the price is below the MA, it generally indicates a downtrend.
    """)

    # Golden Cross and Death Cross
    st.subheader("Golden Cross and Death Cross")
    data['MA_50'] = calculate_ma(data, 50)
    data['MA_200'] = calculate_ma(data, 200)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], mode='lines', name='50-day MA'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_200'], mode='lines', name='200-day MA'))
    fig.update_layout(title='Golden Cross and Death Cross', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    st.markdown("""
    Golden Cross and Death Cross are significant MA crossovers:
    - Golden Cross: 50-day MA crosses above 200-day MA (bullish signal)
    - Death Cross: 50-day MA crosses below 200-day MA (bearish signal)
    These crossovers can potentially signal long-term trend changes.
    """)

def quiz_section():
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What is a Simple Moving Average (SMA)?",
            "options": [
                "The weighted mean of the previous n data points",
                "The unweighted mean of the previous n data points",
                "The median of the previous n data points",
                "The mode of the previous n data points"
            ],
            "correct": "The unweighted mean of the previous n data points",
            "explanation": "A Simple Moving Average (SMA) is calculated by taking the arithmetic mean of a given set of values over a specified period. It gives equal weight to all observations in the period."
        },
        {
            "question": "What does it typically indicate when a shorter-term MA crosses above a longer-term MA?",
            "options": [
                "A potential bearish trend",
                "A potential bullish trend",
                "No change in trend",
                "A decrease in trading volume"
            ],
            "correct": "A potential bullish trend",
            "explanation": "When a shorter-term MA crosses above a longer-term MA, it's often interpreted as a bullish signal, potentially indicating the start of an uptrend. This is known as a 'golden cross' when it involves specific moving averages."
        },
        {
            "question": "What is the main advantage of an Exponential Moving Average (EMA) over a Simple Moving Average (SMA)?",
            "options": [
                "It's easier to calculate",
                "It gives more weight to recent prices",
                "It's always more accurate",
                "It uses less data"
            ],
            "correct": "It gives more weight to recent prices",
            "explanation": "The main advantage of an EMA is that it gives more weight to recent prices, making it more responsive to new information. This can be beneficial in fast-moving markets."
        },
        {
            "question": "What is a 'Death Cross' in the context of moving averages?",
            "options": [
                "When the 50-day MA crosses above the 200-day MA",
                "When the 50-day MA crosses below the 200-day MA",
                "When the price crosses below both the 50-day and 200-day MA",
                "When trading volume reaches zero"
            ],
            "correct": "When the 50-day MA crosses below the 200-day MA",
            "explanation": "A 'Death Cross' occurs when the 50-day moving average crosses below the 200-day moving average. It's considered a bearish signal that potentially indicates the start of a downtrend."
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