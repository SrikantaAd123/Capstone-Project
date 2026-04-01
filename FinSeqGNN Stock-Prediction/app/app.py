
import streamlit as st
import yfinance as yf
import pandas as pd
import torch
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="ETF Predictor", layout="wide")

st.title("📈 ETF Trend Prediction using FINseqGNN")

# -----------------------------
# SELECT ETF
# -----------------------------
ticker = st.selectbox("Select ETF", ["GLD", "SLV"])

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
SEQ = 8
X = []

for i in range(len(data)-SEQ):
    X.append(data.iloc[i:i+SEQ].values)

X = np.array(X)

X_tensor = torch.tensor(X).float()

# -----------------------------
# LOAD MODEL
# -----------------------------
model = torch.load("../outputs/best_model.pth", map_location="cpu")

model.eval()

with torch.no_grad():
    pred = model(X_tensor)

latest = pred[-1].item()

# -----------------------------
# TREND
# -----------------------------
trend = "📈 UPTREND" if latest > 0 else "📉 DOWNTREND"

# -----------------------------
# DISPLAY
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted Trend", trend)
    st.metric("Z-score Prediction", round(latest,4))

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data['Close'], name="Price"))
    st.plotly_chart(fig)

st.write("### Latest Data")
st.dataframe(data.tail())
