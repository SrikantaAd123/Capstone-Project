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
st.set_page_config(layout="wide")
st.title("📊 FINseqGNN Dashboard (Stock + ETF + Z-Score)")

# ==============================
# SAFE VALUE
# ==============================
def safe(val):
    if isinstance(val, pd.Series):
        val = val.values[0]
    return float(val)

# ==============================
# DATA FETCH
# ==============================
def fetch_data(ticker, period):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    return df[['Open','High','Low','Close','Volume']]

# ==============================
# Z-SCORE
# ==============================
def zscore(df):
    df = df.copy()
    for col in df.columns:
        df[col] = (df[col] - df[col].rolling(60).mean()) / df[col].rolling(60).std()
    return df.dropna()

# ==============================
# SEQUENCE
# ==============================
def create_seq(df, seq=8):
    X = []
    for i in range(len(df) - seq):
        X.append(df.iloc[i:i+seq].values)
    return np.array(X)

# ==============================
# MODEL
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
st.sidebar.header("⚙️ Settings")

ticker = st.sidebar.selectbox(
    "Select Asset",
    ["AAPL","GOOG","ABBV","AMZN","GLD","SLV"]
)

period = st.sidebar.selectbox(
    "Period",
    ["3mo","6mo","1y","2y"]
)

chart_type = st.sidebar.selectbox(
    "Chart Type",
    ["Candlestick","Line"]
)

run = st.sidebar.button("🚀 Run Analysis")

# ==============================
# MAIN
# ==============================
if run:

    df = fetch_data(ticker, period)

    if df.empty:
        st.error("No data found")
        st.stop()

    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df.dropna()

    # ==============================
    # METRICS
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
    # MAIN CHART (FIXED)
    # ==============================
    fig = go.Figure()

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'].astype(float),
            high=df['High'].astype(float),
            low=df['Low'].astype(float),
            close=df['Close'].astype(float),
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Close'].astype(float),
            mode='lines',
            line=dict(color='cyan', width=2)
        ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # Z-SCORE PREDICTION
    # ==============================
    st.subheader("🔮 Z-Score Prediction")

    norm_df = zscore(df[['Open','High','Low','Close','Volume']])

    if len(norm_df) >= 80:

        X = create_seq(norm_df)

        if len(X) > 0:

            X_tensor = torch.tensor(X, dtype=torch.float32)

            if X_tensor.dim() != 3:
                X_tensor = X_tensor.view(-1, 8, 5)

            model = Model()
            model.eval()

            with torch.no_grad():
                preds = model(X_tensor)

            z_val = float(preds[-1].item())

            trend = "📈 UPTREND" if z_val > 0 else "📉 DOWNTREND"

            c1,c2 = st.columns(2)
            c1.metric("Z-score", f"{z_val:.4f}")
            c2.metric("Trend", trend)

            # ==============================
            # Z-SCORE GRAPH (FIXED)
            # ==============================
            fig_z = go.Figure()

            fig_z.add_trace(go.Scatter(
                x=df['Date'].iloc[-len(norm_df):],
                y=norm_df['Close'],
                mode='lines',
                line=dict(color='orange', width=2)
            ))

            fig_z.update_layout(
                template="plotly_dark",
                height=300,
                title="Z-score Trend"
            )

            st.plotly_chart(fig_z, use_container_width=True)

        else:
            st.warning("Sequence creation failed")

    else:
        st.warning("Need more data for Z-score")

    # ==============================
    # ETF STYLE CHART
    # ==============================
    if ticker in ["GLD","SLV"]:

        st.subheader("📉 ETF Style Chart")

        fig_etf = go.Figure()

        fig_etf.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Close'].astype(float),
            fill='tozeroy',
            line=dict(color='red', width=2)
        ))

        fig_etf.update_layout(
            template="plotly_dark",
            height=350,
            showlegend=False
        )

        st.plotly_chart(fig_etf, use_container_width=True)

    # ==============================
    # TABLE
    # ==============================
    st.subheader("📄 Data")
    st.dataframe(df.tail())

# ==============================
# LIVE MARKET
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
