import streamlit as st
import yfinance as yf
import pandas as pd
import torch
import numpy as np
import plotly.graph_objects as go

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Multi Stock Predictor", layout="wide")

st.title("📊 Multi-Stock Prediction using FINseqGNN")

# ===============================
# SELECT STOCKS
# ===============================
stocks = st.multiselect(
    "Select Stocks",
    ["AAPL", "GOOG", "ABBV"],
    default=["AAPL"]
)

# ===============================
# LOAD MODEL
# ===============================
model = torch.load("../outputs/best_model.pth", map_location="cpu")
model.eval()

SEQ = 8

# ===============================
# PROCESS EACH STOCK
# ===============================
results = []

for ticker in stocks:

    # -----------------------------
    # DOWNLOAD DATA
    # -----------------------------
    data = yf.download(ticker, period="6mo")

    data = data[['Open','High','Low','Close','Volume']]

    # -----------------------------
    # Z SCORE NORMALIZATION
    # -----------------------------
    for col in data.columns:
        data[col] = (data[col] - data[col].rolling(60).mean()) / data[col].rolling(60).std()

    data = data.dropna()

    # -----------------------------
    # BUILD INPUT
    # -----------------------------
    X = []

    for i in range(len(data) - SEQ):
        X.append(data.iloc[i:i+SEQ].values)

    if len(X) == 0:
        continue

    X = np.array(X)
    X_tensor = torch.tensor(X).float()

    # -----------------------------
    # PREDICTION
    # -----------------------------
    with torch.no_grad():
        pred = model(X_tensor)

    latest = pred[-1].item()

    trend = "📈 UPTREND" if latest > 0 else "📉 DOWNTREND"

    results.append({
        "Stock": ticker,
        "Z-score": round(latest, 4),
        "Trend": trend
    })

    # ===============================
    # DISPLAY PER STOCK
    # ===============================
    st.subheader(f"📌 {ticker} Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Trend", trend)
        st.metric("Z-score", round(latest, 4))

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=data['Close'], name=f"{ticker} Price"))
        st.plotly_chart(fig)

    st.dataframe(data.tail())

# ===============================
# SUMMARY TABLE
# ===============================
if len(results) > 0:
    st.write("## 📋 Summary")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results)
