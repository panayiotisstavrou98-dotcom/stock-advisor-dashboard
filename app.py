import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Stock Advisor",
    page_icon="📈",
    layout="wide",
)

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Settings")

default_tickers = "AAPL, MSFT, NVDA, GOOGL"
tickers_str = st.sidebar.text_input("Tickers (comma-separated)", value=default_tickers)

period = st.sidebar.selectbox("Lookback period", ["6mo", "1y", "2y"], index=1)

min_score = st.sidebar.slider("Minimum score to show", 0, 100, 0, 5)

run_btn = st.sidebar.button("🔄 Run analysis")

st.sidebar.markdown("---")
st.sidebar.caption("This is an educational tool, not investment advice.")


# ---------- DATA + SIGNALS ----------
def fetch_data(tickers, period: str):
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, period=period, auto_adjust=True, progress=False)
            if not df.empty:
                data[t] = df
        except Exception:
            pass
    return data


def compute_signal(df):
    if len(df) < 60:
        return None

    close = df["Close"]
    last = float(close.iloc[-1])

    m6 = float(last / float(close.iloc[0]) - 1)

    high_52w = float(close.max())
    dist_high = float(last / high_52w - 1)

    ret = close.pct_change()
    vol_30 = float(ret[-30:].std() * (252 ** 0.5))

    score = 0
    if m6 > 0.10:
        score += 2
    elif m6 > 0:
        score += 1
    elif m6 < -0.10:
        score -= 2
    else:
        score -= 1

    if dist_high > -0.05:
        score += 1
    elif dist_high < -0.20:
        score -= 1

    if vol_30 > 0.6:
        score -= 1
    elif vol_30 < 0.25:
        score += 1

    if score >= 3:
        signal = "Strong Buy"
    elif score == 2:
        signal = "Buy"
    elif score in [0, 1]:
        signal = "Hold"
    elif score == -1:
        signal = "Reduce"
    else:
        signal = "Sell"

    score_scaled = (score + 5) * 10

    return {
        "last_price": last,
        "momentum_6m": m6,
        "dist_52w_high": dist_high,
        "vol_30d": vol_30,
        "score": score_scaled,
        "signal": signal,
    }


def build_signals_table(tickers, period):
    raw = fetch_data(tickers, period)
    rows = []
    charts = {}

    for t, df in raw.items():
        res = compute_signal(df)
        if res is None:
            continue

        rows.append({
            "Ticker": t,
            "Last Price": res["last_price"],
            "6M Momentum": res["momentum_6m"],
            "Dist 52W High": res["dist_52w_high"],
            "30D Vol": res["vol_30d"],
            "Score": res["score"],
            "Signal": res["signal"],
        })
        charts[t] = df

    if not rows:
        return pd.DataFrame(), charts

    df_sig = pd.DataFrame(rows)
    df_sig = df_sig.sort_values("Score", ascending=False).reset_index(drop=True)
    return df_sig, charts


# ---------- MAIN LAYOUT ----------
st.title("📈 Stock Advisor Dashboard")

tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
if not tickers:
    st.info("Enter at least one ticker in the sidebar to begin.")
    st.stop()

if not run_btn:
    st.info("Set your tickers and click **Run analysis** in the sidebar.")
    st.stop()

with st.spinner("Fetching data and computing signals..."):
    df_sig, charts = build_signals_table(tickers, period)

if df_sig.empty:
    st.error("No signals computed. Try different tickers or a longer period.")
    st.stop()

df_view = df_sig[df_sig["Score"] >= min_score].copy()
if df_view.empty:
    st.warning("No stocks meet the minimum score filter.")
    st.stop()

# ---------- TOP METRICS ----------
col1, col2, col3, col4 = st.columns(4)
n_strong_buy = (df_view["Signal"] == "Strong Buy").sum()
n_buy = (df_view["Signal"] == "Buy").sum()
avg_score = df_view["Score"].mean()
n_total = len(df_view)

col1.metric("Strong Buy", n_strong_buy)
col2.metric("Buy", n_buy)
col3.metric("Avg Score", f"{avg_score:.1f}")
col4.metric("Stocks shown", n_total)

st.markdown("---")

# ---------- TABLE + DETAILS ----------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Signals")
    st.dataframe(
        df_view.style.format({
            "Last Price": "{:.2f}",
            "6M Momentum": "{:.1%}",
            "Dist 52W High": "{:.1%}",
            "30D Vol": "{:.1%}",
            "Score": "{:.0f}",
        }),
        use_container_width=True,
        height=400,
    )

with right_col:
    st.subheader("Details")

    selected_ticker = st.selectbox(
        "Pick a stock to inspect",
        df_view["Ticker"].tolist(),
    )

    row = df_view[df_view["Ticker"] == selected_ticker].iloc[0]

    st.markdown(f"### {selected_ticker}")
    st.write(f"**Signal:** {row['Signal']}  |  **Score:** {row['Score']:.0f}")
    st.write(
        f"**Last price:** {row['Last Price']:.2f}  \n"
        f"**6M momentum:** {row['6M Momentum']:.1%}  \n"
        f"**Dist. from 52W high:** {row['Dist 52W High']:.1%}  \n"
        f"**30D volatility:** {row['30D Vol']:.1%}"
    )

    df_price = charts.get(selected_ticker)
    if df_price is not None and not df_price.empty:
        st.line_chart(df_price["Close"].tail(120), height=200, use_container_width=True)

    st.caption(
        "Heuristic rules combining trend (6M momentum), proximity to highs, and recent volatility. "
        "Later you can replace this block with a full factor / ML engine."
    )
