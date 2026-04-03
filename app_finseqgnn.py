import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go

# ==============================
# PAGE CONFIG (DARK DASHBOARD)
# ==============================
st.set_page_config(page_title="FINseqGNN Dashboard", layout="wide")

st.markdown("""
    <style>
    body {background-color: #0e1117; color: white;}
    </style>
""", unsafe_allow_html=True)

st.title("📊 Real-Time FINseqGNN Stock Prediction Dashboard")

# ==============================
# SIDEBAR CONTROLS
# ==============================
st.sidebar.header("⚙️ Chart Parameters")

ticker = st.sidebar.selectbox("Ticker", ["AAPL", "GOOG", "ABBV", "AMZN"])
period = st.sidebar.selectbox("Time Period", ["1mo","3mo","6mo","1y"])
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line"])

run = st.sidebar.button("🔮 Run Prediction")

# ==============================
# DATA FETCH
# ==============================
@st.cache_data
def load_data(ticker, period):
    df = yf.download(ticker, period=period)
    return df[['Open','High','Low','Close','Volume']]

# ==============================
# PREPROCESSING
# ==============================
def normalize(df):
    for col in df.columns:
        df[col] = (df[col] - df[col].rolling(60).mean()) / df[col].rolling(60).std()
    return df.dropna()

def create_sequences(df, seq_len=8):
    X = []
    for i in range(len(df)-seq_len):
        X.append(df.iloc[i:i+seq_len].values)
    return np.array(X)

# ==============================
# MODEL (YOUR LSTM BASE)
# ==============================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze()

def predict(X):
    model = LSTMModel()
    model.eval()
    with torch.no_grad():
        return model(X)

# ==============================
# MAIN EXECUTION
# ==============================
if run:

    df = load_data(ticker, period)

    # METRICS (REAL TIME)
    latest = df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 Price", f"{latest['Close']:.2f} USD")
    col2.metric("📈 High", f"{df['High'].max():.2f}")
    col3.metric("📉 Low", f"{df['Low'].min():.2f}")
    col4.metric("📊 Volume", f"{int(df['Volume'].iloc[-1]):,}")

    # ==============================
    # CHART
    # ==============================
    fig = go.Figure()

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        ))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close']))

    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # MODEL PREDICTION
    # ==============================
    norm_df = normalize(df)

    if len(norm_df) > 70:
        X = create_sequences(norm_df)

        if len(X) > 0:
            X_tensor = torch.tensor(X, dtype=torch.float32)

            if X_tensor.dim() == 2:
                X_tensor = X_tensor.unsqueeze(0)

            preds = predict(X_tensor)
            z = preds[-1].item()

            trend = "📈 UPTREND" if z > 0 else "📉 DOWNTREND"

            st.subheader("🔮 Model Prediction")
            st.metric("Predicted Trend", trend)
            st.metric("Z-score", round(z, 4))
        else:
            st.warning("Not enough sequence data")
    else:
        st.warning("Not enough data for prediction")

# ==============================
# REAL TIME STOCK LIST
# ==============================
st.sidebar.subheader("📊 Live Stocks")

for s in ["AAPL","GOOG","AMZN"]:
    d = yf.download(s, period="1d")
    price = d['Close'].iloc[-1]
    st.sidebar.write(f"{s}: {price:.2f} USD")
