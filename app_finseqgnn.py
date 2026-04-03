# ==============================
# FINseqGNN + s STYLE DASHBOARD
# ==============================

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(layout="wide")
st.title(" Real-Time Stock Dashboard (FINseqGNN Ready)")

# ==============================
# FUNCTIONS
# ==============================

def fetch_stock_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        return data
    except:
        return pd.DataFrame()

def process_data(data):
    if data.empty:
        return data

    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')

    data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data

def calculate_metrics(data):
    last_close = float(data['Close'].iloc[-1])
    prev_close = float(data['Close'].iloc[0])

    change = last_close - prev_close
    pct_change = (change / prev_close) * 100

    high = float(data['High'].max())
    low = float(data['Low'].min())
    volume = int(data['Volume'].sum())

    return last_close, change, pct_change, high, low, volume

def add_indicators(data):
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['EMA_20'] = data['Close'].ewm(span=20).mean()
    return data

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("Chart Parameters")

ticker = st.sidebar.text_input("Ticker", "AAPL")
time_period = st.sidebar.selectbox("Time Period", ['1d', '1wk', '1mo', '1y', 'max'])
chart_type = st.sidebar.selectbox("Chart Type", ['Candlestick', 'Line'])

indicators = st.sidebar.multiselect(
    "Technical Indicators",
    ['SMA 20', 'EMA 20']
)

interval_mapping = {
    '1d': '1m',
    '1wk': '30m',
    '1mo': '1d',
    '1y': '1wk',
    'max': '1wk'
}

# ==============================
# MAIN DASHBOARD
# ==============================
if st.sidebar.button("Update"):

    data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])

    if data.empty:
        st.error("No data found. Check ticker.")
        st.stop()

    data = process_data(data)
    data = add_indicators(data)

    # ==============================
    # METRICS (FIXED)
    # ==============================
    last_close, change, pct_change, high, low, volume = calculate_metrics(data)

    st.metric(
        label=f"{ticker} Price",
        value=f"{last_close:.2f} USD",
        delta=f"{change:.2f} ({pct_change:.2f}%)"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("High", f"{high:.2f} USD")
    col2.metric("Low", f"{low:.2f} USD")
    col3.metric("Volume", f"{volume:,}")

    # ==============================
    # CHART
    # ==============================
    fig = go.Figure()

    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=data['Datetime'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        ))
    else:
        fig = px.line(data, x='Datetime', y='Close')

    # Add indicators
    for ind in indicators:
        if ind == 'SMA 20':
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_20'], name="SMA 20"))
        if ind == 'EMA 20':
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'], name="EMA 20"))

    fig.update_layout(
        template="plotly_dark",
        height=600,
        title=f"{ticker} Chart"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # DATA TABLE
    # ==============================
    st.subheader("📄 Historical Data")
    st.dataframe(data.tail())

# ==============================
# SIDEBAR LIVE STOCKS (FIXED ERROR)
# ==============================
st.sidebar.header("📊 Real-Time Stock Prices")

symbols = ['AAPL', 'GOOG', 'AMZN', 'MSFT']

for s in symbols:
    try:
        d = yf.download(s, period="1d", interval="1m", progress=False)

        if not d.empty:
            price = float(d['Close'].iloc[-1])
            open_price = float(d['Open'].iloc[0])

            change = price - open_price
            pct = (change / open_price) * 100

            st.sidebar.metric(
                s,
                f"{price:.2f} USD",
                f"{change:.2f} ({pct:.2f}%)"
            )
        else:
            st.sidebar.write(f"{s}: No Data")

    except:
        st.sidebar.write(f"{s}: Error")

# ==============================
# ABOUT
# ==============================
st.sidebar.subheader("About")
st.sidebar.info(
    "This dashboard integrates real-time stock data, technical indicators, "
    "and is extendable to FINseqGNN prediction models."
)
