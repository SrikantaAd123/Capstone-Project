import streamlit as st
import yfinance as yf
import pandas as pd
import torch
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

st.title("📈 AI Stock Prediction Dashboard (FINseqGNN)")

# ==============================
# SIDEBAR CONTROLS
# ==============================
st.sidebar.header("⚙️ Controls")

stocks = st.sidebar.multiselect(
    "Select Stocks",
    ["AAPL", "GOOG", "ABBV", "GLD", "SLV"],
    default=["AAPL"]
)

interval = st.sidebar.selectbox(
    "Interval",
    ["1d", "1h", "5m"],
    index=0
)

refresh = st.sidebar.button("🔄 Refresh Data")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return torch.load("../outputs/best_model.pth", map_location="cpu")

model = load_model()
model.eval()

SEQ = 8

# ==============================
# FUNCTION: PROCESS STOCK
# ==============================
def process_stock(ticker):

    data = yf.download(ticker, period="6mo", interval=interval)

    if data.empty:
        return None, None, None

    data = data[['Open','High','Low','Close','Volume']]

    # Z-score normalization
    for col in data.columns:
        data[col] = (data[col] - data[col].rolling(60).mean()) / data[col].rolling(60).std()

    data = data.dropna()

    # Build sequences
    X = []
    for i in range(len(data) - SEQ):
        X.append(data.iloc[i:i+SEQ].values)

    if len(X) == 0:
        return data, None, None

    X = np.array(X)
    X_tensor = torch.tensor(X).float()

    # Prediction
    with torch.no_grad():
        pred = model(X_tensor)

    latest = pred[-1].item()
    trend = "📈 UPTREND" if latest > 0 else "📉 DOWNTREND"

    return data, latest, trend

# ==============================
# MAIN DISPLAY
# ==============================
results = []

for ticker in stocks:

    st.subheader(f"📊 {ticker}")

    data, latest, trend = process_stock(ticker)

    if data is None:
        st.warning("No data available")
        continue

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Trend", trend if trend else "N/A")

    with col2:
        st.metric("Z-score", round(latest,4) if latest else 0)

    with col3:
        st.metric("Last Price", round(data['Close'].iloc[-1],2))

    # ==============================
    # CANDLESTICK CHART
    # ==============================
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    ))

    # Moving Average
    data['MA20'] = data['Close'].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA20'],
        line=dict(color='blue'),
        name="MA20"
    ))

    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # VOLUME CHART
    # ==============================
    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name="Volume"
    ))

    fig2.update_layout(height=200)

    st.plotly_chart(fig2, use_container_width=True)

    # Save results
    if latest:
        results.append({
            "Stock": ticker,
            "Z-score": round(latest,4),
            "Trend": trend
        })

# ==============================
# SUMMARY TABLE
# ==============================
if len(results) > 0:
    st.write("## 📋 Summary Table")
    df = pd.DataFrame(results)
    st.dataframe(df)

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "📥 Download Predictions",
        csv,
        "predictions.csv",
        "text/csv"
    )

# ==============================
# MODEL COMPARISON (OPTIONAL)
# ==============================
st.write("## ⚖️ Model Comparison (Demo)")

compare_data = pd.DataFrame({
    "Model": ["LSTM", "ALSTM", "GAT", "FINseqGNN"],
    "MSE": [0.0012, 0.0008, 0.0005, 0.0003]
})

fig3 = go.Figure()

fig3.add_trace(go.Bar(
    x=compare_data["Model"],
    y=compare_data["MSE"]
))

st.plotly_chart(fig3)
