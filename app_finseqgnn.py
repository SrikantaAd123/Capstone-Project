import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📈 Real Time Stock Dashboard")

# ---------------------------
# HELPERS
# ---------------------------
def safe(val):
    if isinstance(val, pd.Series):
        val = val.values[0]
    return float(val)

def fetch(ticker, period):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        return df
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

def add_indicators(df):
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    return df.dropna()

def zscore(df):
    out = df.copy()
    for col in ['Open','High','Low','Close','Volume']:
        out[col] = (out[col] - out[col].rolling(60).mean()) / out[col].rolling(60).std()
    return out.dropna()

# ---------------------------
# SIDEBAR (LIKE IMAGE)
# ---------------------------
st.sidebar.header("Chart Parameters")

ticker = st.sidebar.text_input("Ticker", "AAPL")
period = st.sidebar.selectbox("Time Period", ["3mo","6mo","1y","2y"])
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick","Line"])

indicators = st.sidebar.multiselect(
    "Technical Indicators",
    ["SMA20","EMA20"],
    default=["SMA20"]
)

run = st.sidebar.button("Update")

# ---------------------------
# MAIN
# ---------------------------
if run:
    df = fetch(ticker, period)

    if df.empty:
        st.error("No data")
        st.stop()

    df = add_indicators(df)

    # ---------------------------
    # METRICS
    # ---------------------------
    price = safe(df['Close'].iloc[-1])
    prev = safe(df['Close'].iloc[-2])
    change = price - prev
    pct = (change/prev)*100 if prev!=0 else 0

    high = safe(df['High'].max())
    low = safe(df['Low'].min())
    volume = int(safe(df['Volume'].iloc[-1]))

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Last Price", f"{price:.2f} USD", f"{change:+.2f} ({pct:.2f}%)")
    c2.metric("High", f"{high:.2f}")
    c3.metric("Low", f"{low:.2f}")
    c4.metric("Volume", f"{volume:,}")

    # ---------------------------
    # MAIN CHART (CLEAN)
    # ---------------------------
    fig = go.Figure()

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='#22c55e',
            decreasing_line_color='#ef4444',
            name="Candlestick"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            line=dict(color='cyan', width=2),
            name="Price"
        ))

    # Add indicators
    if "SMA20" in indicators:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['SMA20'],
            mode='lines',
            line=dict(color='yellow', width=1.5),
            name="SMA20"
        ))

    if "EMA20" in indicators:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['EMA20'],
            mode='lines',
            line=dict(color='orange', width=1.5),
            name="EMA20"
        ))

    fig.update_layout(
        template="plotly_dark",
        height=520,
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price"),
        margin=dict(l=10,r=10,t=30,b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Z-SCORE TREND (VISIBLE)
    # ---------------------------
    st.subheader("Z-Score Trend")

    zdf = zscore(df[['Open','High','Low','Close','Volume']])

    if len(zdf) > 0:
        fig_z = go.Figure()

        fig_z.add_trace(go.Scatter(
            x=df['Date'].iloc[-len(zdf):],
            y=zdf['Close'],
            mode='lines',
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

# ---------------------------
# LIVE PRICES (SIDEBAR)
# ---------------------------
st.sidebar.subheader("Real-Time Prices")

for s in ["AAPL","GOOG","AMZN"]:
    try:
        d = yf.download(s, period="1d", interval="1m", progress=False)
        if not d.empty:
            p = float(d['Close'].iloc[-1])
            st.sidebar.metric(s, f"{p:.2f} USD")
    except:
        st.sidebar.write(f"{s}: error")
