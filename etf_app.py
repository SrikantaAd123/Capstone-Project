import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 ETF Dashboard (Gold & Silver)")

# ---------------------------
# HELPERS
# ---------------------------
def safe(val):
    if isinstance(val, pd.Series):
        val = val.values[0]
    return float(val)

def fetch(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return df
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

# ---------------------------
# UI
# ---------------------------
ticker = st.selectbox("Select ETF", ["GLD", "SLV"])

range_map = {
    "1D": ("1d","1m"),
    "5D": ("5d","5m"),
    "1M": ("1mo","1h"),
    "6M": ("6mo","1d"),
    "1Y": ("1y","1d"),
    "MAX": ("max","1d")
}

cols = st.columns([1,3])
with cols[0]:
    time_range = st.radio("Range", list(range_map.keys()), horizontal=False)
with cols[1]:
    st.write("")  # spacer

period, interval = range_map[time_range]
df = fetch(ticker, period, interval)

if df.empty:
    st.error("No data available")
    st.stop()

# ---------------------------
# METRICS
# ---------------------------
price = safe(df['Close'].iloc[-1])
open_p = safe(df['Open'].iloc[0])
change = price - open_p
pct = (change / open_p) * 100 if open_p != 0 else 0.0

high = safe(df['High'].max())
low = safe(df['Low'].min())
volume = int(safe(df['Volume'].sum()))

st.markdown(f"""
### {ticker}
## ${price:.2f}
<span style='color:{"#22c55e" if change>0 else "#ef4444"}; font-size:18px'>
{change:+.2f} ({pct:.2f}%) Today
</span>
""", unsafe_allow_html=True)

# ---------------------------
# RED AREA CHART (LIKE IMAGE)
# ---------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Close'].astype(float),
    mode='lines',
    fill='tozeroy',
    line=dict(color='#ef4444', width=2),
    name="Price"
))

fig.update_layout(
    template="plotly_dark",
    height=420,
    showlegend=False,
    margin=dict(l=0,r=0,t=10,b=0),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# INFO GRID (LIKE GOOGLE UI)
# ---------------------------
c1, c2 = st.columns(2)

with c1:
    st.write(f"Open: {open_p:.2f}")
    st.write(f"Day High: {high:.2f}")
    st.write(f"Year Low: {low:.2f}")

with c2:
    st.write(f"Volume: {volume:,}")
    st.write(f"Day Low: {low:.2f}")
    st.write(f"Year High: {high:.2f}")
