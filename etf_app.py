import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 ETF Dashboard (Gold & Silver + Z-Score)")

# ==============================
# SAFE VALUE
# ==============================
def safe(val):
    if isinstance(val, pd.Series):
        val = val.values[0]
    return float(val)

# ==============================
# FETCH DATA (ROBUST)
# ==============================
def fetch(ticker, period, interval):

    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if 'Date' not in df.columns:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])

    df = df[['Date','Open','High','Low','Close','Volume']]
    df = df.dropna()

    return df

# ==============================
# Z-SCORE
# ==============================
def compute_zscore(df):
    df = df.copy()
    df['Z'] = (df['Close'] - df['Close'].rolling(60).mean()) / df['Close'].rolling(60).std()
    return df.dropna()

# ==============================
# TREND LOGIC
# ==============================
def get_trend(df):
    if df['Close'].iloc[-1] > df['Close'].iloc[-5]:
        return "📈 UPTREND"
    else:
        return "📉 DOWNTREND"

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

trend = get_trend(df)

# ==============================
# Z-SCORE METRIC
# ==============================
zdf = compute_zscore(df)

if len(zdf) > 0:
    z_val = float(zdf['Z'].iloc[-1])
else:
    z_val = 0

z_trend = "📈 Bullish" if z_val > 0 else "📉 Bearish"

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
# SHOW TREND
# ==============================
st.subheader(trend)

# ==============================
# MAIN CHART
# ==============================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Close'],
    mode='lines',
    fill='tozeroy',
    line=dict(color='red', width=2)
))

fig.update_layout(
    template="plotly_dark",
    height=420,
    showlegend=False,
    margin=dict(l=0,r=0,t=10,b=0)
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# METRICS GRID
# ==============================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Open", f"{open_p:.2f}")
c2.metric("High", f"{high:.2f}")
c3.metric("Low", f"{low:.2f}")
c4.metric("Volume", f"{volume:,}")

# ==============================
# Z-SCORE DISPLAY
# ==============================
st.subheader("Z-Score Analysis")

st.metric("Z-score", f"{z_val:.3f}", z_trend)

# ==============================
# Z-SCORE GRAPH
# ==============================
if len(zdf) > 0:

    fig_z = go.Figure()

    fig_z.add_trace(go.Scatter(
        x=zdf['Date'],
        y=zdf['Z'],
        mode='lines',
        line=dict(color='orange', width=2)
    ))

    fig_z.update_layout(
        template="plotly_dark",
        height=300,
        yaxis=dict(title="Z-score"),
        margin=dict(l=10,r=10,t=30,b=10)
    )

    st.plotly_chart(fig_z, use_container_width=True)

else:
    st.info("Not enough data for Z-score")
