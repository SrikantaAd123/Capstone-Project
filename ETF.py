import streamlit as st
import yfinance as yf
import numpy as np
import torch
import plotly.graph_objects as go

##########################################################
# FUNCTIONS
##########################################################

def fetch_data(ticker):
    df = yf.download(ticker, period="6mo")
    return df[['Open','High','Low','Close','Volume']]

def normalize(df):
    for col in df.columns:
        df[col] = (df[col] - df[col].rolling(60).mean()) / df[col].rolling(60).std()
    return df.dropna()

def sequences(df):
    X = []
    for i in range(len(df)-8):
        X.append(df.iloc[i:i+8].values)
    return np.array(X)

##########################################################
# MODEL
##########################################################

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(5, 32, batch_first=True)
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze()

##########################################################
# APP
##########################################################

st.title("📈 ETF Prediction (GLD / SLV)")

ticker = st.selectbox("Select ETF", ["GLD", "SLV"])

data = fetch_data(ticker)
data = normalize(data)

X = sequences(data)
X_tensor = torch.tensor(X).float()

model = Model()
model.eval()

with torch.no_grad():
    pred = model(X_tensor)

latest = pred[-1].item()
trend = "📈 UP" if latest > 0 else "📉 DOWN"

st.metric("Trend", trend)
st.metric("Z-score", round(latest,4))

fig = go.Figure()
fig.add_trace(go.Scatter(y=data['Close']))
st.plotly_chart(fig)
