
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 Stock Prediction Dashboard (Smart Indicators + Z-Score + Forecast)")

# ==============================
# FETCH DATA (FIXED)
# ==============================
def fetch(ticker, period):
    df = yf.download(ticker, period=period, interval="1d", progress=False)

    if df.empty:
        return df

    # Fix multi-index issue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if 'Date' not in df.columns:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date','Open','High','Low','Close','Volume']]

    return df.dropna()

# ==============================
# INDICATORS (FIXED)
# ==============================
def add_indicators(df):
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()

    # 🔥 IMPORTANT FIX
    df = df.dropna().reset_index(drop=True)

    return df

# ==============================
# Z-SCORE
# ==============================
def compute_zscore(df):
    df['Z'] = (df['Close'] - df['Close'].rolling(60).mean()) / df['Close'].rolling(60).std()
    return df.dropna().reset_index(drop=True)

# ==============================
# TREND (FIXED)
# ==============================
def get_trend(df):
    latest = df.iloc[-1]

    close = float(latest['Close'])
    sma = float(latest['SMA20'])
    ema = float(latest['EMA20'])

    if close > sma and ema > sma:
        return "📈 STRONG UPTREND", "🟢 BULLISH"
    elif close < sma and ema < sma:
        return "📉 STRONG DOWNTREND", "🔴 BEARISH"
    else:
        return "⚠ SIDEWAYS", "🟡 NEUTRAL"

# ==============================
# FINAL SIGNAL (FIXED)
# ==============================
def final_signal(df, z):
    latest = df.iloc[-1]

    close = float(latest['Close'])
    sma = float(latest['SMA20'])
    ema = float(latest['EMA20'])

    score = 0

    if close > sma:
        score += 2
    else:
        score -= 2

    if ema > sma:
        score += 2
    else:
        score -= 2

    if z < -1:
        score += 2
    elif z > 1.5:
        score -= 3

    if score >= 4:
        return "🚀 STRONG BUY"
    elif score >= 2:
        return "📈 BUY"
    elif score >= 0:
        return "😐 HOLD"
    elif score >= -2:
        return "📉 SELL"
    else:
        return "💀 STRONG SELL"

# ==============================
# FUTURE PREDICTION
# ==============================
def predict_future(df, days=10):
    df = df.copy()
    df['t'] = range(len(df))

    m, b = np.polyfit(df['t'], df['Close'], 1)

    future_t = np.arange(len(df), len(df)+days)
    future_prices = m * future_t + b

    future_dates = pd.date_range(
        start=df['Date'].iloc[-1] + pd.Timedelta(days=1),
        periods=days
    )

    return pd.DataFrame({'Date': future_dates, 'Predicted': future_prices})

# ==============================
# UI
# ==============================
ticker = st.text_input("Enter Ticker", "AAPL")
period = st.selectbox("Select Period", ["3mo","6mo","1y"])

df = fetch(ticker, period)

if df.empty:
    st.error("No data found")
    st.stop()

df = add_indicators(df)
zdf = compute_zscore(df)

if zdf.empty:
    st.error("Not enough data for Z-score")
    st.stop()

z_val = float(zdf['Z'].iloc[-1])

trend, market = get_trend(df)
signal = final_signal(df, z_val)

future_df = predict_future(df)

# ==============================
# DISPLAY
# ==============================
price = float(df['Close'].iloc[-1])
prev = float(df['Close'].iloc[-2]) if len(df) > 1 else price

change = price - prev
pct = (change / prev) * 100 if prev != 0 else 0

st.subheader(f"{trend} | Market: {market}")
st.subheader(f"📊 FINAL SIGNAL: {signal}")

# ==============================
# CHART
# ==============================
fig = go.Figure()

fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Actual Price"))
fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA20'], name="SMA20"))
fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA20'], name="EMA20"))

# Future Prediction
fig.add_trace(go.Scatter(
    x=future_df['Date'],
    y=future_df['Predicted'],
    name="🔮 Future Prediction",
    line=dict(dash='dash', width=3)
))

fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

# ==============================
# METRICS
# ==============================
c1, c2, c3 = st.columns(3)

c1.metric("Price", f"{price:.2f}", f"{change:+.2f} ({pct:.2f}%)")
c2.metric("Z-score", f"{z_val:.3f}")
c3.metric("Volume", f"{int(df['Volume'].iloc[-1]):,}")

# ==============================
# Z-SCORE GRAPH
# ==============================
st.subheader("Z-Score Trend")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=zdf['Date'], y=zdf['Z']))
fig2.update_layout(template="plotly_dark", height=300)

st.plotly_chart(fig2, use_container_width=True)

# ==============================
# FUTURE TABLE
# ==============================
st.subheader("🔮 Future Prediction (Next 10 Days)")
st.dataframe(future_df)
