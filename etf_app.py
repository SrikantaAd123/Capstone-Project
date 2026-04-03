import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 ETF Dashboard (Gold & Silver)")

# ==============================
# SAFE VALUE
# ==============================
def safe(val):
    if isinstance(val, pd.Series):
        val = val.values[0]
    return float(val)

# ==============================
# FETCH DATA (FIXED VERSION)
# ==============================
def fetch(ticker, period, interval):

    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        return df

    # -------- FIX 1: Flatten columns --------
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # -------- FIX 2: Reset index safely --------
    df = df.reset_index()

    # -------- FIX 3: Rename index column --------
    if 'Date' not in df.columns:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # -------- FIX 4: Convert datetime --------
    df['Date'] = pd.to_datetime(df['Date'])

    # -------- FIX 5: Select required columns safely --------
    required_cols = ['Open','High','Low','Close','Volume']
    df = df[['Date'] + required_cols]

    # -------- FIX 6: Drop NaN --------
    df = df.dropna()

    return df

# ==============================
# UI
# ==============================
ticker = st.selectbox("Select ETF", ["GLD", "SLV"])

range_map = {
    "1D": ("1d","1m"),
    "5D": ("5d","5m"),
    "1M": ("1mo","1h"),
    "6M": ("6mo","1d"),
    "1Y": ("1y","1d"),
    "MAX": ("max","1d")
}

time_range = st.radio("Range", list(range_map.keys()), horizontal=True)

period, interval = range_map[time_range]

df = fetch(ticker, period, interval)

if df.empty:
    st.error("No data available")
    st.stop()

# ==============================
# METRICS
# ==============================
price = safe(df['Close'].iloc[-1])
open_p = safe(df['Open'].iloc[0])

change = price - open_p
pct = (change / open_p) * 100 if open_p != 0 else 0

high = safe(df['High'].max())
low = safe(df['Low'].min())
volume = int(df['Volume'].sum())

# ==============================
# HEADER
# ==============================
st.markdown(f"""
### {ticker}
## ${price:.2f}
<span style='color:{"green" if change>0 else "red"}'>
{change:+.2f} ({pct:.2f}%) Today
</span>
""", unsafe_allow_html=True)

# ==============================
# RED AREA CHART
# ==============================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Close'].astype(float),
    mode='lines',
    fill='tozeroy',
    line=dict(color='red', width=2)
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

# ==============================
# INFO GRID
# ==============================
col1, col2 = st.columns(2)

with col1:
    st.write(f"Open: {open_p:.2f}")
    st.write(f"Day High: {high:.2f}")
    st.write(f"Year Low: {low:.2f}")

with col2:
    st.write(f"Volume: {volume:,}")
    st.write(f"Day Low: {low:.2f}")
    st.write(f"Year High: {high:.2f}")
