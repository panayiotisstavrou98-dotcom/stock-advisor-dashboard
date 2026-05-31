import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Stock Advisor", page_icon="📈", layout="wide")

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
default_tickers = "AAPL, MSFT, NVDA, GOOGL, ASML, TSLA, AMZN, META"
tickers_str = st.sidebar.text_input("Tickers (comma-separated)", value=default_tickers)
period      = st.sidebar.selectbox("Lookback period", ["6mo", "1y", "2y"], index=1)
min_score   = st.sidebar.slider("Min composite score", 0, 100, 0, 5)

st.sidebar.markdown("### ⚖️ Factor weights")
w_mom  = st.sidebar.slider("Momentum weight",  0, 100, 30, 5)
w_val  = st.sidebar.slider("Value weight",      0, 100, 25, 5)
w_qual = st.sidebar.slider("Quality weight",    0, 100, 25, 5)
w_risk = st.sidebar.slider("Risk weight",       0, 100, 20, 5)
st.sidebar.caption(f"Total weight: {w_mom+w_val+w_qual+w_risk} (should be 100)")

run_btn = st.sidebar.button("🔄 Run analysis")
st.sidebar.markdown("---")
st.sidebar.caption("Educational tool only — not investment advice.")


# ─── SIGNAL ICONS ────────────────────────────────────────────────────────────
ICON = {"Strong Buy": "🟢", "Buy": "🟡", "Hold": "🔵", "Reduce": "🟠", "Sell": "🔴"}

def signal_from_score(s):
    if s >= 75: return "Strong Buy"
    if s >= 60: return "Buy"
    if s >= 45: return "Hold"
    if s >= 30: return "Reduce"
    return "Sell"


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def safe_series(df, col):
    """Return a 1-D Series from df[col], handling MultiIndex yfinance output."""
    s = df[col].squeeze()
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.dropna()


def norm(val, lo, hi):
    """Normalise val to 0-100 given expected [lo, hi] range."""
    return float(min(max((val - lo) / (hi - lo) * 100, 0), 100))


# ─── FACTOR MODULES ──────────────────────────────────────────────────────────
def momentum_score(close):
    """Score 0-100 based on 1M/6M/12M returns and MA trend."""
    last = float(close.iloc[-1])
    m1   = float(close.iloc[-1] / close.iloc[-21]  - 1) if len(close) >= 21  else 0
    m6   = float(close.iloc[-1] / close.iloc[-126] - 1) if len(close) >= 126 else 0
    m12  = float(close.iloc[-1] / close.iloc[0]    - 1)
    ma50  = float(close.tail(50).mean())
    ma200 = float(close.tail(200).mean()) if len(close) >= 200 else float(close.mean())
    trend = (10 if last > ma50 else 0) + (10 if last > ma200 else 0)
    raw   = m1 * 0.20 + m6 * 0.40 + m12 * 0.40
    base  = norm(raw, -0.50, 0.80)
    return min(base + trend, 100), {"m1": m1, "m6": m6, "m12": m12,
                                     "above_ma50": last > ma50, "above_ma200": last > ma200}


def value_score(info):
    """
    Score 0-100 based on P/E, P/S, EV/EBITDA, FCF yield.
    Lower multiples = higher score.
    """
    pe   = info.get("trailingPE")     or info.get("forwardPE")
    ps   = info.get("priceToSalesTrailing12Months")
    ev_e = info.get("enterpriseToEbitda")
    fcfy = info.get("freeCashflow") and info.get("marketCap") and \
           (info["freeCashflow"] / info["marketCap"])

    parts = []
    if pe  and pe  > 0: parts.append(norm(pe,  5, 60))
    if ps  and ps  > 0: parts.append(norm(ps,  0.5, 20))
    if ev_e and ev_e > 0: parts.append(norm(ev_e, 3, 40))
    if fcfy:             parts.append(norm(-fcfy, -0.15, 0.0))  # high FCF yield = low score → invert

    if not parts:
        return 50, {}   # neutral if no data

    raw_avg = float(np.mean(parts))
    score   = 100 - raw_avg   # cheaper = higher score
    return max(min(score, 100), 0), {
        "pe": pe, "ps": ps, "ev_ebitda": ev_e, "fcf_yield": fcfy
    }


def quality_score(info):
    """
    Score 0-100 based on ROE, profit margin, debt/equity, current ratio.
    """
    roe     = info.get("returnOnEquity")   # 0.20 = 20%
    margin  = info.get("profitMargins")     # 0.15 = 15%
    de      = info.get("debtToEquity")      # 50 = 50%
    cr      = info.get("currentRatio")      # >1.5 is healthy

    parts = []
    if roe    is not None: parts.append(norm(roe,    -0.10, 0.40))
    if margin is not None: parts.append(norm(margin, -0.05, 0.35))
    if de     is not None: parts.append(100 - norm(de, 0, 200))  # lower debt = better
    if cr     is not None: parts.append(norm(cr, 0.5, 3.0))

    if not parts:
        return 50, {}

    return max(min(float(np.mean(parts)), 100), 0), {
        "roe": roe, "profit_margin": margin, "debt_equity": de, "current_ratio": cr
    }


def risk_score(close):
    """
    Score 0-100 where HIGH score = LOW risk.
    Based on 30D volatility and distance from 52W high.
    """
    ret      = close.pct_change().dropna()
    vol_30   = float(ret.tail(30).std() * (252 ** 0.5))
    high_52w = float(close.max())
    dist_hi  = float(close.iloc[-1] / high_52w - 1)   # 0 = at high, -0.3 = 30% below

    vol_s  = 100 - norm(vol_30, 0.10, 0.80)
    dist_s = norm(dist_hi, -0.50, 0.0)
    score  = vol_s * 0.6 + dist_s * 0.4
    return max(min(score, 100), 0), {"vol_30d": vol_30, "dist_52w_high": dist_hi}


def sentiment_score(info):
    """
    Proxy sentiment from analyst recommendation + earnings surprise.
    Returns 0-100.
    """
    rec  = info.get("recommendationMean")   # 1=Strong Buy, 5=Sell
    eps_s = info.get("earningsSurprisePercent") or \
            info.get("earningsQuarterlyGrowth")

    parts = []
    if rec:   parts.append(norm(rec, 1, 5))    # 1 = best → we'll invert below
    if eps_s: parts.append(norm(eps_s, -0.30, 0.50))

    if not parts:
        return 50, {"recommendation": None, "earnings_surprise": None}

    raw = float(np.mean(parts))
    # invert analyst rec (lower mean = more bullish)
    score = 100 - raw if rec else raw
    return max(min(score, 100), 0), {
        "recommendation": rec, "earnings_surprise": eps_s
    }


# ─── COMPOSITE ENGINE ────────────────────────────────────────────────────────
def analyse_ticker(ticker, period, w_mom, w_val, w_qual, w_risk):
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, auto_adjust=True)
        if df.empty or len(df) < 60:
            return None, None

        close = safe_series(df, "Close")
        info  = tk.info or {}

        ms, m_meta  = momentum_score(close)
        vs, v_meta  = value_score(info)
        qs, q_meta  = quality_score(info)
        rs, r_meta  = risk_score(close)
        ss, s_meta  = sentiment_score(info)

        total_w = w_mom + w_val + w_qual + w_risk
        if total_w == 0: total_w = 100
        comp = (ms * w_mom + vs * w_val + qs * w_qual + rs * w_risk) / total_w

        sig = signal_from_score(comp)

        row = {
            "Ticker":      ticker,
            "Last Price":  float(close.iloc[-1]),
            "Mom Score":   round(ms, 1),
            "Value Score": round(vs, 1),
            "Qual Score":  round(qs, 1),
            "Risk Score":  round(rs, 1),
            "Sentiment":   round(ss, 1),
            "Composite":   round(comp, 1),
            "Signal":      sig,
            # detail fields
            "1M Mom":      m_meta["m1"],
            "6M Mom":      m_meta["m6"],
            "12M Mom":     m_meta["m12"],
            "Above MA50":  "✅" if m_meta["above_ma50"]  else "❌",
            "Above MA200": "✅" if m_meta["above_ma200"] else "❌",
            "P/E":         v_meta.get("pe"),
            "P/S":         v_meta.get("ps"),
            "EV/EBITDA":   v_meta.get("ev_ebitda"),
            "FCF Yield":   v_meta.get("fcf_yield"),
            "ROE":         q_meta.get("roe"),
            "Margin":      q_meta.get("profit_margin"),
            "D/E":         q_meta.get("debt_equity"),
            "Curr Ratio":  q_meta.get("current_ratio"),
            "30D Vol":     r_meta["vol_30d"],
            "Dist 52W Hi": r_meta["dist_52w_high"],
            "Analyst Rec": s_meta.get("recommendation"),
            "EPS Surprise":s_meta.get("earnings_surprise"),
        }
        return row, df
    except Exception:
        return None, None


# ─── MAIN ────────────────────────────────────────────────────────────────────
st.title("📈 Stock Advisor Dashboard")

tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
if not tickers:
    st.info("Enter tickers in the sidebar.")
    st.stop()

if not run_btn:
    st.info("Configure settings in the sidebar, then click **🔄 Run analysis**.")
    st.stop()

# ── fetch & score
rows, charts = [], {}
progress = st.progress(0, text="Analysing stocks...")
for i, t in enumerate(tickers):
    progress.progress((i + 1) / len(tickers), text=f"Analysing {t}...")
    row, df = analyse_ticker(t, period, w_mom, w_val, w_qual, w_risk)
    if row:
        rows.append(row)
        charts[t] = df
progress.empty()

if not rows:
    st.error("No data returned. Check tickers or try a longer period.")
    st.stop()

df_all  = pd.DataFrame(rows).sort_values("Composite", ascending=False).reset_index(drop=True)
df_view = df_all[df_all["Composite"] >= min_score].copy()

if df_view.empty:
    st.warning("No stocks pass the minimum score filter.")
    st.stop()

# ── KPIs
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("🟢 Strong Buy", int((df_view["Signal"]=="Strong Buy").sum()))
c2.metric("🟡 Buy",        int((df_view["Signal"]=="Buy").sum()))
c3.metric("🔵 Hold",       int((df_view["Signal"]=="Hold").sum()))
c4.metric("Avg Composite", f"{df_view['Composite'].mean():.1f}")
c5.metric("Stocks shown",  len(df_view))

st.markdown("---")

# ── TABLE TABS
tab1, tab2, tab3, tab4 = st.tabs(["📊 Signals", "💰 Value & Quality", "⚡ Momentum", "⚠️ Risk & Sentiment"])

TABLE_COLS = ["Ticker","Last Price","Mom Score","Value Score","Qual Score","Risk Score","Sentiment","Composite","Signal"]
FMT_BASE   = {"Last Price":"{:.2f}","Mom Score":"{:.1f}","Value Score":"{:.1f}",
               "Qual Score":"{:.1f}","Risk Score":"{:.1f}","Sentiment":"{:.1f}","Composite":"{:.1f}"}

def styled_table(df, cols, fmt):
    display = df[cols].copy()
    if "Signal" in display.columns:
        display["Signal"] = display["Signal"].apply(lambda s: f"{ICON.get(s,'')} {s}")
    return display.style.format(fmt)

with tab1:
    st.subheader("All factor scores + composite signal")
    st.dataframe(styled_table(df_view, TABLE_COLS, FMT_BASE), use_container_width=True, height=420)

with tab2:
    cols2 = ["Ticker","Value Score","P/E","P/S","EV/EBITDA","FCF Yield","Qual Score","ROE","Margin","D/E","Curr Ratio"]
    fmt2  = {"Value Score":"{:.1f}","P/E":"{:.1f}","P/S":"{:.1f}","EV/EBITDA":"{:.1f}",
             "FCF Yield":"{:.1%}","Qual Score":"{:.1f}","ROE":"{:.1%}","Margin":"{:.1%}","D/E":"{:.1f}","Curr Ratio":"{:.2f}"}
    available2 = [c for c in cols2 if c in df_view.columns]
    st.dataframe(df_view[available2].style.format({k:v for k,v in fmt2.items() if k in available2}),
                 use_container_width=True, height=420)

with tab3:
    cols3 = ["Ticker","Mom Score","1M Mom","6M Mom","12M Mom","Above MA50","Above MA200"]
    fmt3  = {"Mom Score":"{:.1f}","1M Mom":"{:.1%}","6M Mom":"{:.1%}","12M Mom":"{:.1%}"}
    available3 = [c for c in cols3 if c in df_view.columns]
    st.dataframe(df_view[available3].style.format({k:v for k,v in fmt3.items() if k in available3}),
                 use_container_width=True, height=420)

with tab4:
    cols4 = ["Ticker","Risk Score","30D Vol","Dist 52W Hi","Sentiment","Analyst Rec","EPS Surprise"]
    fmt4  = {"Risk Score":"{:.1f}","30D Vol":"{:.1%}","Dist 52W Hi":"{:.1%}",
             "Sentiment":"{:.1f}","Analyst Rec":"{:.2f}","EPS Surprise":"{:.1%}"}
    available4 = [c for c in cols4 if c in df_view.columns]
    st.dataframe(df_view[available4].style.format({k:v for k,v in fmt4.items() if k in available4}),
                 use_container_width=True, height=420)

st.markdown("---")

# ── DETAIL PANEL
st.subheader("🔍 Stock Deep Dive")
sel = st.selectbox("Select a ticker", df_view["Ticker"].tolist())
row = df_view[df_view["Ticker"] == sel].iloc[0]
icon = ICON.get(row["Signal"], "")

d1, d2, d3 = st.columns(3)

with d1:
    st.markdown(f"### {sel}  {icon} {row['Signal']}")
    st.metric("Composite Score", f"{row['Composite']:.0f} / 100")
    st.markdown(f"""
| Factor | Score |
|---|---|
| 🚀 Momentum | {row['Mom Score']:.1f} |
| 💰 Value | {row['Value Score']:.1f} |
| ⭐ Quality | {row['Qual Score']:.1f} |
| 🛡️ Risk | {row['Risk Score']:.1f} |
| 📰 Sentiment | {row['Sentiment']:.1f} |
""")

with d2:
    st.markdown("**Momentum detail**")
    st.write(f"• 1M return: {row['1M Mom']:.1%}")
    st.write(f"• 6M return: {row['6M Mom']:.1%}")
    st.write(f"• 12M return: {row['12M Mom']:.1%}")
    st.write(f"• Above MA50: {row['Above MA50']}")
    st.write(f"• Above MA200: {row['Above MA200']}")
    st.markdown("**Value detail**")
    st.write(f"• P/E: {row['P/E']}")
    st.write(f"• P/S: {row['P/S']}")
    st.write(f"• EV/EBITDA: {row['EV/EBITDA']}")
    if row['FCF Yield']: st.write(f"• FCF Yield: {row['FCF Yield']:.1%}")

with d3:
    st.markdown("**Quality detail**")
    if row['ROE']:    st.write(f"• ROE: {row['ROE']:.1%}")
    if row['Margin']: st.write(f"• Net margin: {row['Margin']:.1%}")
    if row['D/E']:    st.write(f"• Debt/Equity: {row['D/E']:.1f}")
    if row['Curr Ratio']: st.write(f"• Current ratio: {row['Curr Ratio']:.2f}")
    st.markdown("**Risk & Sentiment**")
    st.write(f"• 30D Volatility: {row['30D Vol']:.1%}")
    st.write(f"• Dist. 52W High: {row['Dist 52W Hi']:.1%}")
    if row['Analyst Rec']: st.write(f"• Analyst rec mean: {row['Analyst Rec']:.2f} (1=Strong Buy)")
    if row['EPS Surprise']: st.write(f"• EPS Surprise: {row['EPS Surprise']:.1%}")

# Price chart
df_price = charts.get(sel)
if df_price is not None and not df_price.empty:
    price_series = safe_series(df_price, "Close")
    st.line_chart(price_series.tail(252), height=250, use_container_width=True)

st.caption(
    "Composite = weighted blend of Momentum, Value, Quality, Risk scores. "
    "Weights adjustable in the sidebar. Data via Yahoo Finance."
)
