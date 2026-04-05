import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 Stock Prediction Dashboard (Smart Indicators + Z-Score + Forecast)")

# ==============================
# FETCH DATA
# ==============================
def fetch(ticker, period):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    df = df.reset_index()
    df = df[['Date','Open','High','Low','Close','Volume']]
    return df.dropna()

# ==============================
# INDICATORS
# ==============================
def add_indicators(df):
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    return df

# ==============================
# Z-SCORE
# ==============================
def compute_zscore(df):
    df['Z'] = (df['Close'] - df['Close'].rolling(60).mean()) / df['Close'].rolling(60).std()
    return df.dropna()

# ==============================
# TREND
# ==============================
def get_trend(df):
    latest = df.iloc[-1]
    if latest['Close'] > latest['SMA20'] and latest['EMA20'] > latest['SMA20']:
        return "📈 STRONG UPTREND", "🟢 BULLISH"
    elif latest['Close'] < latest['SMA20'] and latest['EMA20'] < latest['SMA20']:
        return "📉 STRONG DOWNTREND", "🔴 BEARISH"
    else:
        return "⚠ SIDEWAYS", "🟡 NEUTRAL"

# ==============================
# SIGNAL
# ==============================
def final_signal(df, z):
    latest = df.iloc[-1]
    score = 0

    if latest['Close'] > latest['SMA20']:
        score += 2
    else:
        score -= 2

    if latest['EMA20'] > latest['SMA20']:
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
    df['t'] = range(len(df))
    m, b = np.polyfit(df['t'], df['Close'], 1)

    future_t = np.arange(len(df), len(df)+days)
    future_prices = m * future_t + b

    future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days)

    return pd.DataFrame({'Date': future_dates, 'Predicted': future_prices})

# ==============================
# UI
# ==============================
ticker = st.text_input("Ticker", "AAPL")
period = st.selectbox("Period", ["3mo","6mo","1y"])

df = fetch(ticker, period)
df = add_indicators(df)
zdf = compute_zscore(df)

z_val = zdf['Z'].iloc[-1]
trend, market = get_trend(df)
signal = final_signal(df, z_val)

future_df = predict_future(df)

# ==============================
# DISPLAY
# ==============================
price = df['Close'].iloc[-1]

st.subheader(f"{trend} | {market}")
st.subheader(f"📊 FINAL SIGNAL: {signal}")

fig = go.Figure()

fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Actual"))
fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA20'], name="SMA20"))
fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA20'], name="EMA20"))

fig.add_trace(go.Scatter(
    x=future_df['Date'],
    y=future_df['Predicted'],
    name="🔮 Future",
    line=dict(dash='dash')
))

fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Z-Score Trend")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=zdf['Date'], y=zdf['Z']))
fig2.update_layout(template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)
