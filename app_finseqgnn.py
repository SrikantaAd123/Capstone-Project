import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(layout="wide")
st.title("📊 FINseqGNN Dashboard (Z-Score Prediction)")

# ==============================
# FUNCTIONS
# ==============================

def fetch_stock_data(ticker, period, interval):
    return yf.download(ticker, period=period, interval=interval, progress=False)

def process_data(df):
    df = df.copy()
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'Datetime'}, inplace=True)
    return df

# -------- SAFE VALUE EXTRACTOR --------
def safe_scalar(val):
    if isinstance(val, pd.Series):
        val = val.values[0]
    return float(val)

# -------- Z-SCORE --------
def zscore_normalize(df):
    df = df.copy()
    for col in ['Open','High','Low','Close','Volume']:
        df[col] = (df[col] - df[col].rolling(60).mean()) / df[col].rolling(60).std()
    return df.dropna()

# -------- SEQUENCES --------
def create_sequences(df, seq_len=8):
    X = []
    for i in range(len(df) - seq_len):
        X.append(df.iloc[i:i+seq_len].values)
    return np.array(X)

# -------- MODEL --------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze()

def predict_zscore(X):
    model = LSTMModel()
    model.eval()
    with torch.no_grad():
        return model(X)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("⚙️ Parameters")

ticker = st.sidebar.text_input("Ticker", "AAPL")
period = st.sidebar.selectbox("Time Period", ['1mo','3mo','6mo','1y'])
chart_type = st.sidebar.selectbox("Chart Type", ['Candlestick','Line'])

run = st.sidebar.button("Run Analysis")

interval_map = {
    '1mo': '1d',
    '3mo': '1d',
    '6mo': '1d',
    '1y': '1wk'
}

# ==============================
# MAIN
# ==============================
if run:

    df = fetch_stock_data(ticker, period, interval_map[period])

    if df.empty:
        st.error("No data found")
        st.stop()

    df = process_data(df)

    # ==============================
    # SAFE METRICS (FIXED)
    # ==============================
    last_price = safe_scalar(df['Close'].iloc[-1])
    high = safe_scalar(df['High'].max())
    low = safe_scalar(df['Low'].min())
    volume = int(safe_scalar(df['Volume'].iloc[-1]))

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Price", f"{last_price:.2f} USD")
    col2.metric("High", f"{high:.2f}")
    col3.metric("Low", f"{low:.2f}")
    col4.metric("Volume", f"{volume:,}")

    # ==============================
    # CHART
    # ==============================
    fig = go.Figure()

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df['Datetime'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        ))
    else:
        fig = px.line(df, x='Datetime', y='Close')

    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # Z-SCORE PREDICTION
    # ==============================
    st.subheader("🔮 Z-Score Prediction")

    norm_df = zscore_normalize(df[['Open','High','Low','Close','Volume']])

    if len(norm_df) < 70:
        st.warning("Not enough data for prediction")
    else:
        X = create_sequences(norm_df)

        if len(X) > 0:
            X_tensor = torch.tensor(X, dtype=torch.float32)

            if X_tensor.dim() == 2:
                X_tensor = X_tensor.unsqueeze(0)

            preds = predict_zscore(X_tensor)
            z = float(preds[-1].item())

            trend = "📈 UPTREND" if z > 0 else "📉 DOWNTREND"

            c1, c2 = st.columns(2)
            c1.metric("Z-score", f"{z:.4f}")
            c2.metric("Trend", trend)
        else:
            st.warning("Sequence error")

    # ==============================
    # DATA TABLE
    # ==============================
    st.subheader("📄 Data")
    st.dataframe(df.tail())

# ==============================
# SIDEBAR LIVE STOCKS (SAFE)
# ==============================
st.sidebar.subheader("📊 Live Market")

stocks = ["AAPL","GOOG","AMZN","MSFT"]

for s in stocks:
    try:
        d = yf.download(s, period="1d", interval="1m", progress=False)

        if not d.empty:
            price = safe_scalar(d['Close'].iloc[-1])
            st.sidebar.metric(s, f"{price:.2f} USD")
    except:
        st.sidebar.write(f"{s}: Error")
