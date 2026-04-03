# ==============================
# FINAL FINseqGNN DASHBOARD (PRO VERSION)
# ==============================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 Real-Time Stock Dashboard + Z-Score Prediction")

# ==============================
# SAFE VALUE EXTRACTOR
# ==============================
def safe(val):
    if isinstance(val, pd.Series):
        val = val.values[0]
    return float(val)

# ==============================
# DATA FETCH
# ==============================
def fetch_data(ticker, period="1y"):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    return df[['Open','High','Low','Close','Volume']]

# ==============================
# Z-SCORE NORMALIZATION
# ==============================
def zscore(df):
    df = df.copy()
    for col in df.columns:
        df[col] = (df[col] - df[col].rolling(60).mean()) / df[col].rolling(60).std()
    return df.dropna()

# ==============================
# SEQUENCE CREATION
# ==============================
def create_seq(df, seq=8):
    X = []
    for i in range(len(df)-seq):
        X.append(df.iloc[i:i+seq].values)
    return np.array(X)

# ==============================
# MODEL (IMPROVED)
# ==============================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze()

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("⚙️ Parameters")

ticker = st.sidebar.selectbox(
    "Select Asset",
    ["AAPL","GOOG","ABBV","AMZN","GLD","SLV"]
)

period = st.sidebar.selectbox(
    "Period",
    ["3mo","6mo","1y","2y"]
)

chart_type = st.sidebar.selectbox(
    "Chart",
    ["Candlestick","Line"]
)

run = st.sidebar.button("🚀 Run")

# ==============================
# MAIN
# ==============================
if run:

    df = fetch_data(ticker, period)

    if df.empty:
        st.error("No data found")
        st.stop()

    df.reset_index(inplace=True)

    # ==============================
    # METRICS (SAFE)
    # ==============================
    price = safe(df['Close'].iloc[-1])
    high = safe(df['High'].max())
    low = safe(df['Low'].min())
    volume = int(safe(df['Volume'].iloc[-1]))

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Price", f"{price:.2f} USD")
    c2.metric("High", f"{high:.2f}")
    c3.metric("Low", f"{low:.2f}")
    c4.metric("Volume", f"{volume:,}")

    # ==============================
    # CHART
    # ==============================
    fig = go.Figure()

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        ))
    else:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close']))

    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # Z-SCORE + PREDICTION
    # ==============================
    st.subheader("🔮 Z-Score Prediction")

    norm_df = zscore(df[['Open','High','Low','Close','Volume']])

    # FIX: ensure enough data
    if len(norm_df) >= 80:

        X = create_seq(norm_df)

        if len(X) > 0:

            X_tensor = torch.tensor(X, dtype=torch.float32)

            # FIX SHAPE (IMPORTANT)
            if X_tensor.dim() != 3:
                X_tensor = X_tensor.view(-1, 8, 5)

            model = Model()
            model.eval()

            with torch.no_grad():
                preds = model(X_tensor)

            z_val = float(preds[-1].item())

            trend = "📈 UPTREND" if z_val > 0 else "📉 DOWNTREND"

            cc1,cc2 = st.columns(2)
            cc1.metric("Z-score", f"{z_val:.4f}")
            cc2.metric("Trend", trend)

        else:
            st.warning("Sequence creation failed")

    else:
        st.warning("Need more data (increase period)")

    # ==============================
    # TABLE
    # ==============================
    st.subheader("📄 Data")
    st.dataframe(df.tail())

# ==============================
# LIVE MARKET (FIXED)
# ==============================
st.sidebar.subheader("📊 Live Market")

stocks = ["AAPL","GOOG","AMZN"]

for s in stocks:
    try:
        d = yf.download(s, period="1d", interval="1m", progress=False)

        if not d.empty:
            p = safe(d['Close'].iloc[-1])
            st.sidebar.metric(s, f"{p:.2f} USD")

    except:
        st.sidebar.write(f"{s}: Error")
