```python
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 ETF Dashboard (Gold & Silver + Smart Signals)")

# ==============================
# SAFE VALUE
# ==============================
def safe(val):
    if isinstance(val, pd.Series):
        val = val.values[0]
    return float(val)

# ==============================
# FETCH DATA
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
# INDICATORS
# ==============================
def add_indicators(df):
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    return df

# ==============================
# Z-SCORE
# ==============================
def compute_zscore(df):
    df = df.copy()
    df['Z'] = (df['Close'] - df['Close'].rolling(60).mean()) / df['Close'].rolling(60).std()
    return df.dropna()

# ==============================
# TREND + MARKET TYPE
# ==============================
def get_trend(df):
    latest = df.iloc[-1]

    if latest['Close'] > latest['SMA20'] and latest['EMA20'] > latest['SMA20']:
        return "📈 STRONG UPTREND", "🟢 BULLISH"
    elif latest['Close'] < latest['SMA20'] and latest['EMA20'] < latest['SMA20']:
        return "📉 STRONG DOWNTREND", "🔴 BEARISH"
    else:
        return "⚠ SIDEWAYS", "🟡 NEUTRAL"

# ==============================
# FINAL SMART SIGNAL
# ==============================
def final_signal(df, z_val):
    latest = df.iloc[-1]

    score = 0

    # Trend strength
    if latest['Close'] > latest['SMA20']:
        score += 2
    else:
        score -= 2

    if latest['EMA20'] > latest['SMA20']:
        score += 2
    else:
        score -= 2

    # Z-score filter (IMPORTANT)
    if z_val < -1:
        score += 2
    elif z_val > 1.5:
        score -= 3

    # Decision
    if score >= 4:
        return "🚀 STRONG BUY (Bullish + Undervalued)"
    elif score >= 2:
        return "📈 BUY (Bullish)"
    elif score >= 0:
        return "😐 HOLD (Neutral)"
    elif score >= -2:
        return "📉 SELL (Bearish)"
    else:
        return "💀 STRONG SELL (Overbought + Bearish)"

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
# ADD INDICATORS
# ==============================
df = add_indicators(df)

# ==============================
# PRICE LOGIC
# ==============================
price = safe(df['Close'].iloc[-1])
prev = safe(df['Close'].iloc[-2]) if len(df) > 1 else price

change = price - prev
pct = (change / prev) * 100 if prev != 0 else 0

high = safe(df['High'].max())
low = safe(df['Low'].min())
volume = int(df['Volume'].sum())

trend, market_type = get_trend(df)

# ==============================
# Z-SCORE
# ==============================
zdf = compute_zscore(df)
z_val = float(zdf['Z'].iloc[-1]) if len(zdf) > 0 else 0

# Improved Z-score interpretation
if z_val > 2:
    z_signal = "🚨 Strong Overbought"
elif z_val > 1:
    z_signal = "⚠ Overbought"
elif z_val < -2:
    z_signal = "💎 Strong Oversold"
elif z_val < -1:
    z_signal = "🔥 Oversold"
else:
    z_signal = "😐 Neutral"

# ==============================
# FINAL SIGNAL
# ==============================
signal = final_signal(df, z_val)

# ==============================
# HEADER
# ==============================
st.markdown(f"""
### {ticker}
## ${price:.2f}
<span style='color:{"green" if change>0 else "red"}'>
{change:+.2f} ({pct:.2f}%)
</span>
""", unsafe_allow_html=True)

st.subheader(f"{trend} | Market: {market_type}")
st.subheader(f"📊 FINAL SIGNAL: {signal}")

# ==============================
# MAIN CHART
# ==============================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Close'],
    mode='lines',
    fill='tozeroy',
    line=dict(width=2)
))

fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA20'], name="SMA20"))
fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA20'], name="EMA20"))

fig.update_layout(
    template="plotly_dark",
    height=420,
    margin=dict(l=0,r=0,t=10,b=0)
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# METRICS
# ==============================
c1, c2, c3, c4 = st.columns(4)

c1.metric("High", f"{high:.2f}")
c2.metric("Low", f"{low:.2f}")
c3.metric("Volume", f"{volume:,}")
c4.metric("Z-score", f"{z_val:.3f}", z_signal)

# ==============================
# Z-SCORE GRAPH
# ==============================
st.subheader("Z-Score Trend")

if len(zdf) > 0:
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(
        x=zdf['Date'],
        y=zdf['Z']
    ))
    fig_z.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_z, use_container_width=True)
else:
    st.info("Not enough data")
```
