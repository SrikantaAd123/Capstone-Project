import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("📊 ETF Dashboard (Gold & Silver)")

# ==============================
# SELECT ETF
# ==============================
ticker = st.selectbox("Select ETF", ["GLD", "SLV"])

# ==============================
# FETCH DATA
# ==============================
df = yf.download(ticker, period="1d", interval="1m", progress=False)

if df.empty:
    st.error("No data available")
    st.stop()

# ==============================
# SAFE VALUE EXTRACTION
# ==============================
def safe(val):
    if isinstance(val, pd.Series):
        val = val.values[0]
    return float(val)

# ==============================
# METRICS
# ==============================
price = safe(df['Close'].iloc[-1])
open_p = safe(df['Open'].iloc[0])

change = price - open_p
pct = (change / open_p) * 100

high = safe(df['High'].max())
low = safe(df['Low'].min())
volume = int(safe(df['Volume'].sum()))

# ==============================
# HEADER STYLE (LIKE GOOGLE UI)
# ==============================
st.markdown(f"""
### {ticker} ETF
## ${price:.2f}
<span style='color:{"green" if change>0 else "red"}'>
{change:+.2f} ({pct:.2f}%) Today
</span>
""", unsafe_allow_html=True)

# ==============================
# TIME FILTER BUTTONS
# ==============================
time_range = st.radio("Select Range", ["1D","5D","1M","6M","1Y"], horizontal=True)

# ==============================
# FETCH DATA BASED ON RANGE
# ==============================
range_map = {
    "1D": ("1d","1m"),
    "5D": ("5d","5m"),
    "1M": ("1mo","1h"),
    "6M": ("6mo","1d"),
    "1Y": ("1y","1d")
}

period, interval = range_map[time_range]
df = yf.download(ticker, period=period, interval=interval, progress=False)

# ==============================
# CHART (RED STYLE LIKE IMAGE)
# ==============================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=df['Close'],
    fill='tozeroy',
    line=dict(color='red'),
    name="Price"
))

fig.update_layout(
    template="plotly_dark",
    height=400,
    showlegend=False,
    margin=dict(l=0,r=0,t=0,b=0)
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# INFO GRID (LIKE IMAGE)
# ==============================
col1, col2 = st.columns(2)

with col1:
    st.write(f"Open: {open_p:.2f}")
    st.write(f"Day High: {high:.2f}")
    st.write(f"Year Low: {safe(df['Low'].min()):.2f}")

with col2:
    st.write(f"Volume: {volume:,}")
    st.write(f"Day Low: {low:.2f}")
    st.write(f"Year High: {safe(df['High'].max()):.2f}")
