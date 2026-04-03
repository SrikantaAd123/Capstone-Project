import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import numpy as np
import torch

##########################################################
# PART 1: DATA FETCH + PREPROCESSING (YOUR COLAB LOGIC)
##########################################################

def fetch_data(ticker, period="6mo"):
    df = yf.download(ticker, period=period)
    df = df[['Open','High','Low','Close','Volume']]
    return df

def zscore_normalize(df):
    for col in df.columns:
        df[col] = (df[col] - df[col].rolling(60).mean()) / df[col].rolling(60).std()
    return df.dropna()

def create_sequences(df, seq_len=8):
    X = []
    for i in range(len(df) - seq_len):
        X.append(df.iloc[i:i+seq_len].values)
    return np.array(X)

##########################################################
# PART 2: MODEL (LSTM BASELINE FROM YOUR COLAB)
##########################################################

class PriceLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(5, 64, batch_first=True)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze()

def predict_model(X):
    model = PriceLSTM()
    model.eval()
    with torch.no_grad():
        preds = model(X)
    return preds

##########################################################
# PART 3: STREAMLIT DASHBOARD
##########################################################

st.set_page_config(layout="wide")
st.title("📊 FINseqGNN Stock Prediction Dashboard")

# Sidebar
st.sidebar.header("Parameters")
ticker = st.sidebar.selectbox("Select Stock", ["AAPL", "GOOG", "ABBV"])
period = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y"])

##########################################################
# MAIN EXECUTION
##########################################################

if st.sidebar.button("Run Prediction"):

    # Fetch + preprocess
    data = fetch_data(ticker, period)
    data = zscore_normalize(data)

    # Create sequences
    X = create_sequences(data)
    X_tensor = torch.tensor(X).float()

    # Predict
    preds = predict_model(X_tensor)
    latest_pred = preds[-1].item()

    # Trend
    trend = "📈 UPTREND" if latest_pred > 0 else "📉 DOWNTREND"

    ##########################################################
    # UI OUTPUT
    ##########################################################

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Trend", trend)
        st.metric("Z-score", round(latest_pred, 4))

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=data['Close'], name="Close Price"))
        fig.update_layout(title=f"{ticker} Price Chart")
        st.plotly_chart(fig)

    st.subheader("Recent Data")
    st.dataframe(data.tail())
