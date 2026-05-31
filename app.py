import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

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

    # .squeeze() ensures we always get a 1-D Series regardless of yfinance version
    close = df["Close"].squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    last     = float(close.iloc[-1])
    m6       = float(last / float(close.iloc[0]) - 1)
    high_52w = float(close.max())
    dist_high = float(last / high_52w - 1)

    ret    = close.pct_change().dropna()
    vol_30 = float(ret.tail(30).std() * (252 ** 0.5))

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
        "last_price":    last,
        "momentum_6m":   m6,
        "dist_52w_high": dist_high,
        "vol_30d":       vol_30,
        "score":         score_scaled,
        "signal":        signal,
    }


def build_signals_table(tickers, period):
    raw = fetch_data(tickers, period)
    rows, charts = [], {}

    for t, df in raw.items():
        res = compute_signal(df)
        if res is None:
            continue
        rows.append({
            "Ticker":        t,
            "Last Price":    res["last_price"],
            "6M Momentum":   res["momentum_6m"],
            "Dist 52W High": res["dist_52w_high"],
            "30D Vol":       res["vol_30d"],
            "Score":         res["score"],
            "Signal":        res["signal"],
        })
        charts[t] = df

    if not rows:
        return pd.DataFrame(), charts

    df_sig = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    return df_sig, charts


# ---------- MAIN ----------
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

# --- KPIs ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("🟢 Strong Buy", int((df_view["Signal"] == "Strong Buy").sum()))
col2.metric("🟡 Buy",        int((df_view["Signal"] == "Buy").sum()))
col3.metric("Avg Score",     f"{df_view['Score'].mean():.1f}")
col4.metric("Stocks shown",  len(df_view))

st.markdown("---")

# --- TABLE + DETAILS ---
left_col, right_col = st.columns([2, 1])

SIGNAL_ICON = {"Strong Buy": "🟢", "Buy": "🟡", "Hold": "🔵", "Reduce": "🟠", "Sell": "🔴"}

with left_col:
    st.subheader("Signals")
    display = df_view.copy()
    display["Signal"] = display["Signal"].apply(lambda s: f"{SIGNAL_ICON.get(s,'')} {s}")
    st.dataframe(
        display.style.format({
            "Last Price":    "{:.2f}",
            "6M Momentum":   "{:.1%}",
            "Dist 52W High": "{:.1%}",
            "30D Vol":       "{:.1%}",
            "Score":         "{:.0f}",
        }),
        use_container_width=True,
        height=420,
    )

with right_col:
    st.subheader("Details")
    sel = st.selectbox("Pick a stock to inspect", df_view["Ticker"].tolist())
    row = df_view[df_view["Ticker"] == sel].iloc[0]

    icon = SIGNAL_ICON.get(row["Signal"], "")
    st.markdown(f"### {sel}  {icon} {row['Signal']}")
    st.markdown(f"""
| Metric | Value |
|---|---|
| Last Price | **{row['Last Price']:.2f}** |
| 6M Momentum | {row['6M Momentum']:.1%} |
| Dist. 52W High | {row['Dist 52W High']:.1%} |
| 30D Volatility | {row['30D Vol']:.1%} |
| **Score** | **{row['Score']:.0f} / 100** |
""")

    df_price = charts.get(sel)
    if df_price is not None and not df_price.empty:
        price_series = df_price["Close"].squeeze()
        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.iloc[:, 0]
        st.line_chart(price_series.tail(252), height=220, use_container_width=True)

    st.caption(
        "Score = momentum (6M return + dist from high + volatility). "
        "Value & quality factors coming in next version."
    )
