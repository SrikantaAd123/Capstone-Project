import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("📊 Smart Stock Dashboard (AI + Indicators)")

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
def fetch(ticker, period):
    df = yf.download(ticker, period=period, interval="1d", progress=False)

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
# TREND
# ==============================
def get_trend(df):
    latest = df.iloc[-1]

    if latest['Close'] > latest['SMA20'] and latest['EMA20'] > latest['SMA20']:
        return "📈 STRONG UPTREND"
    elif latest['Close'] < latest['SMA20'] and latest['EMA20'] < latest['SMA20']:
        return "📉 STRONG DOWNTREND"
    else:
        return "⚠ SIDEWAYS"

# ==============================
# FINAL PREDICTION
# ==============================
def final_prediction(df, z_val):
    latest = df.iloc[-1]
    score = 0

    if latest['Close'] > latest['SMA20']:
        score += 1
    else:
        score -= 1

    if latest['EMA20'] > latest['SMA20']:
        score += 1
    else:
        score -= 1

    if z_val < -1:
        score += 1
    elif z_val > 1:
        score -= 1

    if score >= 2:
        return "🚀 STRONG BUY"
    elif score == 1:
        return "📈 BUY"
    elif score == 0:
        return "😐 HOLD"
    elif score == -1:
        return "📉 SELL"
    else:
        return "💀 STRONG SELL"

# ==============================
# REAL-TIME PRICE (FIXED)
# ==============================
@st.cache_data(ttl=60)
def get_price(ticker):
    try:
        stock = yf.Ticker(ticker)

        # Try real-time (1m)
        df = stock.history(period="1d", interval="1m")

        # Fallback if market closed
        if df.empty:
            df = stock.history(period="5d")

        if not df.empty:
            price = float(df['Close'].iloc[-1])

            if len(df) > 1:
                prev = float(df['Close'].iloc[-2])
            else:
                prev = price

            change = price - prev
            pct = (change / prev) * 100 if prev != 0 else 0

            return round(price, 2), round(change, 2), round(pct, 2)

    except:
        return None, None, None

    return None, None, None

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("⚙ Parameters")

tickers = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,GOOG,AMZN")
ticker_list = [t.strip().upper() for t in tickers.split(",")]

period = st.sidebar.selectbox("Time Period", ["3mo","6mo","1y","2y"])
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick","Line"])

indicators = st.sidebar.multiselect(
    "Indicators",
    ["SMA20","EMA20"],
    default=["SMA20","EMA20"]
)

run = st.sidebar.button("Update")

# ==============================
# MAIN
# ==============================
if run:

    for ticker in ticker_list:

        st.markdown("---")
        st.header(f"📊 {ticker}")

        df = fetch(ticker, period)

        if df.empty:
            st.warning(f"{ticker} data not found")
            continue

        df = add_indicators(df)
        zdf = compute_zscore(df)

        price = safe(df['Close'].iloc[-1])
        prev = safe(df['Close'].iloc[-2])

        change = price - prev
        pct = (change / prev) * 100 if prev != 0 else 0

        high = safe(df['High'].max())
        low = safe(df['Low'].min())
        volume = int(safe(df['Volume'].iloc[-1]))

        trend = get_trend(df)

        z_val = float(zdf['Z'].iloc[-1]) if len(zdf) > 0 else 0

        # Z-score signal
        if z_val > 1:
            z_signal = "⚠ Overbought"
        elif z_val < -1:
            z_signal = "🔥 Oversold"
        else:
            z_signal = "😐 Neutral"

        signal = final_prediction(df, z_val)

        # ==============================
        # METRICS
        # ==============================
        c1,c2,c3,c4,c5 = st.columns(5)

        c1.metric("Price", f"{price:.2f} USD", f"{change:+.2f} ({pct:.2f}%)")
        c2.metric("High", f"{high:.2f}")
        c3.metric("Low", f"{low:.2f}")
        c4.metric("Volume", f"{volume:,}")
        c5.metric("Z-score", f"{z_val:.3f}", z_signal)

        st.subheader(trend)
        st.subheader(f"📊 FINAL SIGNAL: {signal}")

        # ==============================
        # CHART
        # ==============================
        fig = go.Figure()

        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines'
            ))

        if "SMA20" in indicators:
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['SMA20'],
                name="SMA20"
            ))

        if "EMA20" in indicators:
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['EMA20'],
                name="EMA20"
            ))

        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

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

# ==============================
# SIDEBAR LIVE PRICES (FIXED)
# ==============================
st.sidebar.subheader("📊 Live Prices")

for s in ticker_list:
    price, change, pct = get_price(s)

    if price is not None:
        arrow = "🔼" if change >= 0 else "🔽"

        st.sidebar.metric(
            label=s,
            value=f"{price} USD",
            delta=f"{arrow} {change} ({pct}%)"
        )
    else:
        st.sidebar.metric(s, "Fetching...", "")
