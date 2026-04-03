import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="FINseqGNN Dashboard", layout="wide")

st.title("📊 Real-Time FINseqGNN Stock Prediction Dashboard")

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("⚙️ Chart Parameters")

ticker = st.sidebar.selectbox(
    "Select Asset",
    ["AAPL", "GOOG", "ABBV", "AMZN", "GLD", "SLV"]
)

period = st.sidebar.selectbox(
    "Time Period",
    ["1mo", "3mo", "6mo", "1y"]
)

chart_type = st.sidebar.selectbox(
    "Chart Type",
    ["Candlestick", "Line"]
)

run = st.sidebar.button("🔮 Run Prediction")

# ==============================
# DATA FUNCTION
# ==============================
@st.cache_data
def load_data(ticker, period):
    df = yf.download(ticker, period=period, progress=False)
    return df[['Open','High','Low','Close','Volume']]

# ==============================
# PREPROCESSING
# ==============================
def normalize(df):
    df = df.copy()
    for col in df.columns:
        df[col] = (df[col] - df[col].rolling(60).mean()) / df[col].rolling(60).std()
    return df.dropna()

def create_sequences(df, seq_len=8):
    X = []
    for i in range(len(df) - seq_len):
        X.append(df.iloc[i:i+seq_len].values)
    return np.array(X)

# ==============================
# MODEL
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

    if X.dim() != 3:
        raise ValueError(f"LSTM expects 3D input, got {X.dim()}D")

    with torch.no_grad():
        preds = model(X)

    return preds

# ==============================
# MAIN DASHBOARD
# ==============================
if run:
    try:
        df = load_data(ticker, period)

        if len(df) == 0:
            st.error("No data available")
            st.stop()

        latest = df.iloc[-1]

        # ==============================
        # METRICS
        # ==============================
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("💰 Price", f"{latest['Close']:.2f} USD")
        col2.metric("📈 High", f"{df['High'].max():.2f}")
        col3.metric("📉 Low", f"{df['Low'].min():.2f}")
        col4.metric("📊 Volume", f"{int(latest['Volume']):,}")

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
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name="Close Price"
            ))

        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # ==============================
        # PREDICTION (FINseqGNN LOGIC)
        # ==============================
        norm_df = normalize(df)

        if len(norm_df) < 70:
            st.warning("Not enough data after normalization")
        else:
            X = create_sequences(norm_df)

            if len(X) == 0:
                st.warning("Sequence creation failed")
            else:
                X_tensor = torch.tensor(X, dtype=torch.float32)

                if X_tensor.dim() == 2:
                    X_tensor = X_tensor.unsqueeze(0)

                preds = predict(X_tensor)
                z = preds[-1].item()

                trend = "📈 UPTREND" if z > 0 else "📉 DOWNTREND"

                st.subheader("🔮 Model Prediction")
                st.metric("Trend", trend)
                st.metric("Z-score", round(z, 4))

        # ==============================
        # DATA TABLE
        # ==============================
        st.subheader("📄 Latest Data")
        st.dataframe(df.tail())

    except Exception as e:
        st.error(f"Error: {str(e)}")

# ==============================
# SIDEBAR LIVE STOCKS (FIXED)
# ==============================
st.sidebar.subheader("📊 Live Market")

stocks = ["AAPL", "GOOG", "AMZN"]

for s in stocks:
    try:
        d = yf.download(s, period="1d", progress=False)

        if len(d) > 0:
            price = float(d['Close'].iloc[-1])
            st.sidebar.metric(s, f"{price:.2f} USD")
        else:
            st.sidebar.write(f"{s}: No Data")

    except:
        st.sidebar.write(f"{s}: Error")
