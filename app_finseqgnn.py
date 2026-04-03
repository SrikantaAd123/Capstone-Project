import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("📊 Real Time Stock Dashboard (With Z-Score)")

# ==============================
# SAFE VALUE
# ==============================
def safe(val):
    if isinstance(val, pd.Series):
        val = val.values[0]
    return float(val)

# ==============================
# FETCH DATA
# ==============================
def fetch(ticker, period):

    df = yf.download(ticker, period=period, interval="1d", progress=False)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if 'Date' not in df.columns:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])

    df = df[['Date','Open','High','Low','Close','Volume']]
    df = df.dropna()

    return df

# ==============================
# INDICATORS
# ==============================
def add_indicators(df):
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    return df

# ==============================
# Z-SCORE CALCULATION
# ==============================
def compute_zscore(df):
    df = df.copy()
    df['Z'] = (df['Close'] - df['Close'].rolling(60).mean()) / df['Close'].rolling(60).std()
    return df.dropna()

# ==============================
# TREND LOGIC
# ==============================
def get_trend(df):
    if df['Close'].iloc[-1] > df['Close'].iloc[-5]:
        return "📈 UPTREND"
    else:
        return "📉 DOWNTREND"

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("Chart Parameters")

ticker = st.sidebar.text_input("Ticker", "AAPL")

period = st.sidebar.selectbox(
    "Time Period",
    ["3mo","6mo","1y","2y"]
)

chart_type = st.sidebar.selectbox(
    "Chart Type",
    ["Candlestick","Line"]
)

indicators = st.sidebar.multiselect(
    "Technical Indicators",
    ["SMA20","EMA20"],
    default=["SMA20"]
)

run = st.sidebar.button("Update")

# ==============================
# MAIN
# ==============================
if run:

    df = fetch(ticker, period)

    if df.empty:
        st.error("Invalid ticker or no data")
        st.stop()

    df = add_indicators(df)

    # ==============================
    # METRICS
    # ==============================
    price = safe(df['Close'].iloc[-1])
    prev = safe(df['Close'].iloc[-2])

    change = price - prev
    pct = (change / prev) * 100 if prev != 0 else 0

    high = safe(df['High'].max())
    low = safe(df['Low'].min())
    volume = int(safe(df['Volume'].iloc[-1]))

    trend = get_trend(df)

    # ==============================
    # Z-SCORE METRIC
    # ==============================
    zdf = compute_zscore(df)

    if len(zdf) > 0:
        z_val = float(zdf['Z'].iloc[-1])
    else:
        z_val = 0

    z_trend = "📈 Bullish" if z_val > 0 else "📉 Bearish"

    # ==============================
    # DISPLAY METRICS
    # ==============================
    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric("Price", f"{price:.2f} USD", f"{change:+.2f} ({pct:.2f}%)")
    c2.metric("High", f"{high:.2f}")
    c3.metric("Low", f"{low:.2f}")
    c4.metric("Volume", f"{volume:,}")
    c5.metric("Z-score", f"{z_val:.3f}", z_trend)

    st.subheader(trend)

    # ==============================
    # MAIN CHART
    # ==============================
    fig = go.Figure()

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='#22c55e',
            decreasing_line_color='#ef4444'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            line=dict(color='cyan', width=2)
        ))

    # Indicators
    if "SMA20" in indicators:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['SMA20'],
            line=dict(color='yellow', width=2),
            name="SMA20"
        ))

    if "EMA20" in indicators:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['EMA20'],
            line=dict(color='orange', width=2),
            name="EMA20"
        ))

    fig.update_layout(
        template="plotly_dark",
        height=550,
        margin=dict(l=10,r=10,t=30,b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # Z-SCORE TREND GRAPH
    # ==============================
    st.subheader("Z-Score Trend")

    if len(zdf) > 0:

        fig_z = go.Figure()

        fig_z.add_trace(go.Scatter(
            x=zdf['Date'],
            y=zdf['Z'],
            line=dict(color='orange', width=2),
            name="Z-score"
        ))

        fig_z.update_layout(
            template="plotly_dark",
            height=300,
            yaxis=dict(title="Z-score"),
            margin=dict(l=10,r=10,t=30,b=10)
        )

        st.plotly_chart(fig_z, use_container_width=True)

    else:
        st.info("Not enough data for Z-score")

# ==============================
# LIVE PRICES
# ==============================
st.sidebar.subheader("Real-Time Stock Prices")

for s in ["AAPL","GOOG","AMZN"]:
    try:
        d = yf.download(s, period="1d", interval="1m", progress=False)

        if not d.empty:
            p = float(d['Close'].iloc[-1])
            st.sidebar.metric(s, f"{p:.2f} USD")

    except:
        st.sidebar.write(f"{s}: Loading...")
