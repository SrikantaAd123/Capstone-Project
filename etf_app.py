import streamlit as st
import yfinance as yf
import numpy as np
import torch
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📈 ETF Prediction Dashboard (Gold & Silver)")

ticker = st.selectbox("Select ETF", ["GLD", "SLV"])

df = yf.download(ticker, period="6mo")

st.metric("Current Price", f"{df['Close'].iloc[-1]:.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], fill='tozeroy'))
fig.update_layout(template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

# ==============================
# SIMPLE MODEL
# ==============================
def normalize(df):
    for col in df.columns:
        df[col] = (df[col] - df[col].rolling(60).mean()) / df[col].rolling(60).std()
    return df.dropna()

def sequences(df):
    X = []
    for i in range(len(df)-8):
        X.append(df.iloc[i:i+8].values)
    return np.array(X)

norm = normalize(df[['Open','High','Low','Close','Volume']])

if len(norm) > 70:
    X = sequences(norm)

    if len(X) > 0:
        X_tensor = torch.tensor(X, dtype=torch.float32)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(5, 32, batch_first=True)
                self.fc = torch.nn.Linear(32,1)

            def forward(self,x):
                _,(h,_) = self.lstm(x)
                return self.fc(h[-1])

        model = Model()
        model.eval()

        with torch.no_grad():
            pred = model(X_tensor)

        z = pred[-1].item()
        trend = "UP" if z > 0 else "DOWN"

        st.metric("Predicted Trend", trend)
        st.metric("Z-score", round(z,4))
