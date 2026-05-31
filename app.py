import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Advisor", page_icon="📈", layout="wide")

# ─── PERSISTENT STATE ─────────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = set()
if "owned" not in st.session_state:
    st.session_state["owned"] = {}   # ticker -> qty, avg_price
if "results" not in st.session_state:
    st.session_state["results"] = None
if "charts" not in st.session_state:
    st.session_state["charts"] = {}

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

default_tickers = "AAPL, MSFT, NVDA, GOOGL, ASML, TSLA, AMZN, META"
tickers_str = st.sidebar.text_input("Tickers (comma-separated)", value=default_tickers)
period      = st.sidebar.selectbox("Lookback period", ["6mo", "1y", "2y"], index=1)
min_score   = st.sidebar.slider("Min composite score", 0, 100, 0, 5)

st.sidebar.markdown("### ⚖️ Factor weights")
w_mom  = st.sidebar.slider("Momentum",  0, 100, 30, 5)
w_val  = st.sidebar.slider("Value",     0, 100, 25, 5)
w_qual = st.sidebar.slider("Quality",   0, 100, 25, 5)
w_risk = st.sidebar.slider("Risk",      0, 100, 20, 5)
st.sidebar.caption(f"Total: {w_mom+w_val+w_qual+w_risk} (should be 100)")

st.sidebar.markdown("### 🚨 Hard risk caps")
max_vol   = st.sidebar.slider("Max 30D vol before cap (annualised)", 0.20, 1.50, 0.80, 0.05)
max_dd    = st.sidebar.slider("Max drawdown from 52W high before cap", -0.60, -0.05, -0.25, 0.05)

st.sidebar.markdown("### 📰 News API (optional)")news_key = st.sidebar.text_input("NewsAPI key (leave blank to skip)", type="password")

run_btn = st.sidebar.button("🔄 Run analysis")
st.sidebar.markdown("---")
st.sidebar.caption("Educational tool only — not investment advice.")

# ─── HELPERS ────────────────────────────────────────────────────────────────
ICON = {"Strong Buy": "🟢", "Buy": "🟡", "Hold": "🔵", "Reduce": "🟠", "Sell": "🔴"}

def signal_from_score(s):
    if s >= 75: return "Strong Buy"
    if s >= 60: return "Buy"
    if s >= 45: return "Hold"
    if s >= 30: return "Reduce"
    return "Sell"

def safe_series(df, col):
    s = df[col].squeeze()
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.dropna()

def norm(val, lo, hi):
    if val is None or (hi - lo) == 0: return 50
    return float(min(max((val - lo) / (hi - lo) * 100, 0), 100))

# ─── SECTOR DATA (for relative value) ──────────────────────────────────────
SECTOR_MEDIAN_PE = {
    "Technology": 28, "Consumer Cyclical": 22, "Healthcare": 20,
    "Financial Services": 14, "Communication Services": 18,
    "Industrials": 20, "Consumer Defensive": 22, "Energy": 12,
    "Basic Materials": 15, "Real Estate": 35, "Utilities": 18,
    "default": 20
}

# ─── FACTOR MODULES ──────────────────────────────────────────────────────────
def momentum_score(close):
    last   = float(close.iloc[-1])
    m1     = float(close.iloc[-1]/close.iloc[-21]  - 1) if len(close)>=21  else 0
    m6     = float(close.iloc[-1]/close.iloc[-126] - 1) if len(close)>=126 else 0
    m12    = float(close.iloc[-1]/close.iloc[0]    - 1)
    ma50   = float(close.tail(50).mean())
    ma200  = float(close.tail(200).mean()) if len(close)>=200 else float(close.mean())
    trend  = (10 if last>ma50 else 0) + (10 if last>ma200 else 0)
    raw    = m1*0.20 + m6*0.40 + m12*0.40
    base   = norm(raw, -0.50, 0.80)
    return min(base+trend, 100), {"m1":m1,"m6":m6,"m12":m12,
                                   "above_ma50":last>ma50,"above_ma200":last>ma200}

def value_score(info, sector="default"):
    pe    = info.get("trailingPE") or info.get("forwardPE")
    ps    = info.get("priceToSalesTrailing12Months")
    ev_e  = info.get("enterpriseToEbitda")
    mktcap= info.get("marketCap")
    fcf   = info.get("freeCashflow")
    fcfy  = (fcf/mktcap) if (fcf and mktcap and mktcap>0) else None

    # Sector-relative P/E scoring
    sector_pe = SECTOR_MEDIAN_PE.get(sector, SECTOR_MEDIAN_PE["default"])
    parts = []
    if pe and pe>0:
        # score relative to sector median: at median=50, half=80, double=20
        rel_pe = pe / sector_pe
        parts.append(max(min((2 - rel_pe) * 50, 100), 0))
    if ps  and ps>0:  parts.append(norm(ps,   0.5, 20)  and 100-norm(ps, 0.5, 20))
    if ev_e and ev_e>0: parts.append(100-norm(ev_e, 3, 40))
    if fcfy:          parts.append(norm(fcfy, -0.05, 0.15))

    score = float(np.mean(parts)) if parts else 50
    return max(min(score,100),0), {"pe":pe,"ps":ps,"ev_ebitda":ev_e,
                                    "fcf_yield":fcfy,"sector":sector,"sector_median_pe":sector_pe}

def quality_score(info):
    roe     = info.get("returnOnEquity")
    margin  = info.get("profitMargins")
    de      = info.get("debtToEquity")
    cr      = info.get("currentRatio")
    # earnings stability proxy: operating cash flow / net income
    ocf     = info.get("operatingCashflow")
    ni      = info.get("netIncomeToCommon")
    accrual = None
    if ocf and ni and ni != 0:
        accrual = ocf / abs(ni)  # >1 = cash earnings > reported (good)

    parts = []
    if roe    is not None: parts.append(norm(roe,    -0.10, 0.40))
    if margin is not None: parts.append(norm(margin, -0.05, 0.35))
    if de     is not None: parts.append(100-norm(de, 0, 200))
    if cr     is not None: parts.append(norm(cr, 0.5, 3.0))
    if accrual is not None: parts.append(norm(accrual, 0.5, 2.5))  # earnings quality

    score = float(np.mean(parts)) if parts else 50
    return max(min(score,100),0), {"roe":roe,"profit_margin":margin,
                                    "debt_equity":de,"current_ratio":cr,"accrual_ratio":accrual}

def risk_score_fn(close):
    ret     = close.pct_change().dropna()
    vol_30  = float(ret.tail(30).std() * (252**0.5))
    high_52 = float(close.max())
    dist_hi = float(close.iloc[-1]/high_52 - 1)
    # max drawdown in period
    roll_max = close.cummax()
    dd       = ((close - roll_max)/roll_max)
    max_dd   = float(dd.min())
    vol_s   = 100-norm(vol_30, 0.10, 0.80)
    dist_s  = norm(dist_hi, -0.50, 0.0)
    dd_s    = 100-norm(abs(max_dd), 0, 0.60)
    score   = vol_s*0.4 + dist_s*0.3 + dd_s*0.3
    return max(min(score,100),0), {"vol_30d":vol_30,"dist_52w_high":dist_hi,"max_dd":max_dd}

def sentiment_score_fn(info, ticker, news_key):
    rec   = info.get("recommendationMean")
    eps_s = info.get("earningsQuarterlyGrowth")
    news_sent = 50  # neutral default
    headlines = []

    if news_key:
        try:
            url = (f"https://newsapi.org/v2/everything?"
                   f"q={ticker}&language=en&sortBy=publishedAt"
                   f"&from={(datetime.now()-timedelta(days=7)).strftime('%Y-%m-%d')}"
                   f"&pageSize=10&apiKey={news_key}")
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                articles = r.json().get("articles", [])
                headlines = [a["title"] for a in articles if a.get("title")]
                # simple polarity: count positive/negative keywords
                pos_words = ["beat","surge","strong","record","growth","upgraded","outperform","rally","profit"]
                neg_words = ["miss","fall","weak","loss","downgrade","underperform","crash","cut","decline"]
                pos = sum(1 for h in headlines for w in pos_words if w in h.lower())
                neg = sum(1 for h in headlines for w in neg_words if w in h.lower())
                total = pos + neg
                if total > 0:
                    news_sent = (pos/total)*100
        except Exception:
            pass

    parts = []
    if rec:   parts.append(100 - norm(rec, 1, 5))  # 1=Strong Buy → high score
    if eps_s: parts.append(norm(eps_s, -0.30, 0.50))
    parts.append(news_sent)

    score = float(np.mean(parts)) if parts else 50
    return max(min(score,100),0), {"recommendation":rec,"earnings_surprise":eps_s,
                                    "news_sentiment":news_sent,"headlines":headlines}

# ─── HARD RISK CAP ──────────────────────────────────────────────────────────────
def apply_hard_caps(signal, score, vol_30d, dist_52w_high, max_vol_cap, max_dd_cap):
    """If risk thresholds are breached, cap signal to Hold regardless of score."""
    capped = False
    reason = ""
    if vol_30d > max_vol_cap:
        capped = True
        reason = f"Vol {vol_30d:.0%} > cap {max_vol_cap:.0%}"
    if dist_52w_high < max_dd_cap:
        capped = True
        reason += (" | " if reason else "") + f"Drawdown {dist_52w_high:.0%} < cap {max_dd_cap:.0%}"
    if capped and signal in ["Strong Buy", "Buy"]:
        return "Hold", min(score, 44), f"⚠️ Capped: {reason}"
    return signal, score, ""

# ─── BACKTEST (simple monthly rebalance) ────────────────────────────────────────
def simple_backtest(charts, df_results):
    """
    Equal-weight portfolio of current Buy/Strong Buy stocks.
    Benchmark = SPY.
    Returns daily cumulative return series.
    """
    buy_tickers = df_results[df_results["Signal"].isin(["Strong Buy","Buy"])]["Ticker"].tolist()
    if not buy_tickers:
        return None, None

    returns = {}
    for t in buy_tickers:
        df = charts.get(t)
        if df is None: continue
        s = safe_series(df, "Close").pct_change().dropna()
        returns[t] = s

    if not returns:
        return None, None

    ret_df  = pd.DataFrame(returns).dropna()
    port    = ret_df.mean(axis=1)  # equal weight
    cum_port = (1 + port).cumprod() - 1

    # Benchmark SPY
    try:
        spy = yf.download("SPY", period="1y", auto_adjust=True, progress=False)
        spy_close = safe_series(spy, "Close")
        spy_ret   = spy_close.pct_change().dropna()
        spy_ret.index = spy_ret.index.tz_localize(None) if spy_ret.index.tzinfo else spy_ret.index
        port.index    = port.index.tz_localize(None) if port.index.tzinfo else port.index
        combined = pd.DataFrame({"Portfolio": port, "SPY": spy_ret}).dropna()
        cum_both = (1 + combined).cumprod() - 1
        return cum_port, cum_both
    except Exception:
        return cum_port, None

# ─── ML SCAFFOLD (XGBoost on factor scores) ───────────────────────────────────
def ml_score(mom_s, val_s, qual_s, risk_s, sent_s):
    """
    Placeholder ML score using a simple logistic blend.
    Replace with trained XGBoost model in next iteration.
    """
    features = np.array([mom_s, val_s, qual_s, risk_s, sent_s])
    weights  = np.array([0.30, 0.25, 0.25, 0.10, 0.10])
    raw = float(np.dot(features, weights))
    # sigmoid to keep in 0-100
    sig = 100 / (1 + np.exp(-(raw - 50) / 15))
    return round(sig, 1)

# ─── MAIN ANALYSIS ENGINE ──────────────────────────────────────────────────────
def analyse_ticker(ticker, period, w_mom, w_val, w_qual, w_risk,
                  news_key, max_vol_cap, max_dd_cap):
    try:
        tk    = yf.Ticker(ticker)
        df    = tk.history(period=period, auto_adjust=True)
        if df.empty or len(df)<60: return None, None
        close = safe_series(df, "Close")
        info  = tk.info or {}
        sector = info.get("sector", "default")

        ms, m_meta = momentum_score(close)
        vs, v_meta = value_score(info, sector)
        qs, q_meta = quality_score(info)
        rs, r_meta = risk_score_fn(close)
        ss, s_meta = sentiment_score_fn(info, ticker, news_key)

        total_w = w_mom+w_val+w_qual+w_risk or 100
        comp = (ms*w_mom + vs*w_val + qs*w_qual + rs*w_risk) / total_w

        # ML scaffold
        ml = ml_score(ms, vs, qs, rs, ss)
        # blend composite 80% rule-based + 20% ML
        comp = comp*0.80 + ml*0.20
        comp = round(min(max(comp,0),100), 1)

        sig = signal_from_score(comp)

        # Hard risk cap
        sig, comp, cap_reason = apply_hard_caps(
            sig, comp, r_meta["vol_30d"], r_meta["dist_52w_high"],
            max_vol_cap, max_dd_cap
        )

        return {
            "Ticker":        ticker,
            "Sector":        sector,
            "Last Price":    round(float(close.iloc[-1]),2),
            "Mom Score":     round(ms,1), "Value Score": round(vs,1),
            "Qual Score":    round(qs,1), "Risk Score":  round(rs,1),
            "Sentiment":     round(ss,1), "ML Score":    ml,
            "Composite":     comp,        "Signal":      sig,
            "Cap Reason":    cap_reason,
            "1M Mom":  m_meta["m1"],  "6M Mom":  m_meta["m6"],  "12M Mom": m_meta["m12"],
            "Above MA50":  "✅" if m_meta["above_ma50"]  else "❌",
            "Above MA200": "✅" if m_meta["above_ma200"] else "❌",
            "P/E":         v_meta.get("pe"),
            "P/S":         v_meta.get("ps"),
            "EV/EBITDA":   v_meta.get("ev_ebitda"),
            "FCF Yield":   v_meta.get("fcf_yield"),
            "Sector PE":   v_meta.get("sector_median_pe"),
            "ROE":         q_meta.get("roe"),
            "Margin":      q_meta.get("profit_margin"),
            "D/E":         q_meta.get("debt_equity"),
            "Curr Ratio":  q_meta.get("current_ratio"),
            "Accrual":     q_meta.get("accrual_ratio"),
            "30D Vol":     r_meta["vol_30d"],
            "Dist 52W Hi": r_meta["dist_52w_high"],
            "Max DD":      r_meta["max_dd"],
            "Analyst Rec": s_meta.get("recommendation"),
            "EPS Surprise":s_meta.get("earnings_surprise"),
            "News Sent":   s_meta.get("news_sentiment"),
            "Headlines":   s_meta.get("headlines", []),
        }, df
    except Exception as e:
        return None, None


# ──────────────────── MAIN UI ────────────────────────────────────────────────
st.title("📈 Stock Advisor Dashboard")

tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
if not tickers:
    st.info("Enter tickers in the sidebar.")
    st.stop()

if run_btn:
    rows, charts = [], {}
    prog = st.progress(0, text="Analysing...")
    for i, t in enumerate(tickers):
        prog.progress((i+1)/len(tickers), text=f"Analysing {t}...")
        row, df = analyse_ticker(t, period, w_mom, w_val, w_qual, w_risk,
                                  news_key, max_vol, max_dd)
        if row:
            rows.append(row)
            charts[t] = df
    prog.empty()
    if rows:
        st.session_state["results"] = pd.DataFrame(rows).sort_values("Composite",ascending=False).reset_index(drop=True)
        st.session_state["charts"]  = charts
    else:
        st.error("No data returned.")
        st.stop()

if st.session_state["results"] is None:
    st.info("Configure settings in the sidebar, then click **🔄 Run analysis**.")
    st.stop()

df_all  = st.session_state["results"]
charts  = st.session_state["charts"]
df_view = df_all[df_all["Composite"] >= min_score].copy()

# ─── MAIN PAGE TABS ────────────────────────────────────────────────────────────
tab_sig, tab_val, tab_mom, tab_risk, tab_watch, tab_port, tab_bt = st.tabs([
    "📊 Signals", "💰 Value & Quality", "⚡ Momentum",
    "⚠️ Risk & Sentiment", "⭐ Watchlist", "💼 Portfolio", "🕰️ Backtest"
])

# ─── TAB 1: SIGNALS ──────────────────────────────────────────────────────────────with tab_sig:
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🟢 Strong Buy", int((df_view["Signal"]=="Strong Buy").sum()))
    c2.metric("🟡 Buy",        int((df_view["Signal"]=="Buy").sum()))
    c3.metric("🔵 Hold",       int((df_view["Signal"]=="Hold").sum()))
    c4.metric("Avg Composite", f"{df_view['Composite'].mean():.1f}")
    c5.metric("Stocks",        len(df_view))
    st.markdown("---")

    left, right = st.columns([2,1])
    with left:
        disp = df_view[["Ticker","Sector","Last Price","Mom Score","Value Score",
                        "Qual Score","Risk Score","Sentiment","ML Score","Composite","Signal","Cap Reason"]].copy()
        disp["Signal"] = disp["Signal"].apply(lambda s: f"{ICON.get(s,'')} {s}")
        st.dataframe(disp.style.format({"Last Price":"{:.2f}",
            "Mom Score":"{:.1f}","Value Score":"{:.1f}","Qual Score":"{:.1f}",
            "Risk Score":"{:.1f}","Sentiment":"{:.1f}","ML Score":"{:.1f}","Composite":"{:.1f}"}),
            use_container_width=True, height=420)

    with right:
        st.subheader("🔍 Deep Dive")
        sel = st.selectbox("Select ticker", df_view["Ticker"].tolist(), key="sel_main")
        row = df_view[df_view["Ticker"]==sel].iloc[0]
        icon = ICON.get(row["Signal"],"")
        st.markdown(f"### {sel} {icon} {row['Signal']}")
        if row["Cap Reason"]:
            st.warning(row["Cap Reason"])
        st.markdown(f"""
| Factor | Score |
|---|---|
| 🚀 Momentum | {row['Mom Score']:.1f} |
| 💰 Value | {row['Value Score']:.1f} |
| ⭐ Quality | {row['Qual Score']:.1f} |
| 🛡️ Risk | {row['Risk Score']:.1f} |
| 📰 Sentiment | {row['Sentiment']:.1f} |
| 🤖 ML | {row['ML Score']:.1f} |
| **Composite** | **{row['Composite']:.1f}** |
""")
        df_price = charts.get(sel)
        if df_price is not None:
            ps = safe_series(df_price,"Close")
            st.line_chart(ps.tail(252), height=200, use_container_width=True)

        # News headlines
        headlines = row.get("Headlines", [])
        if headlines:
            st.markdown("**Latest news**")
            for h in headlines[:5]:
                st.caption(f"• {h}")

        # Watchlist toggle
        wl = st.session_state["watchlist"]
        if sel in wl:
            if st.button(f"⭐ Remove {sel} from watchlist"):
                wl.discard(sel)
        else:
            if st.button(f"☆ Add {sel} to watchlist"):
                wl.add(sel)

# ─── TAB 2: VALUE & QUALITY ──────────────────────────────────────────────────────
with tab_val:
    st.caption("P/E is scored relative to sector median. See 'Sector PE' column.")
    cols2 = ["Ticker","Sector","Value Score","P/E","Sector PE","P/S","EV/EBITDA","FCF Yield",
             "Qual Score","ROE","Margin","D/E","Curr Ratio","Accrual"]
    fmt2  = {"Value Score":"{:.1f}","P/E":"{:.1f}","Sector PE":"{:.0f}","P/S":"{:.1f}",
             "EV/EBITDA":"{:.1f}","FCF Yield":"{:.1%}","Qual Score":"{:.1f}",
             "ROE":"{:.1%}","Margin":"{:.1%}","D/E":"{:.1f}","Curr Ratio":"{:.2f}","Accrual":"{:.2f}"}
    av2 = [c for c in cols2 if c in df_view.columns]
    st.dataframe(df_view[av2].style.format({k:v for k,v in fmt2.items() if k in av2}),
                 use_container_width=True, height=420)

# ─── TAB 3: MOMENTUM ────────────────────────────────────────────────────────────
with tab_mom:
    cols3 = ["Ticker","Mom Score","1M Mom","6M Mom","12M Mom","Above MA50","Above MA200"]
    fmt3  = {"Mom Score":"{:.1f}","1M Mom":"{:.1%}","6M Mom":"{:.1%}","12M Mom":"{:.1%}"}
    av3 = [c for c in cols3 if c in df_view.columns]
    st.dataframe(df_view[av3].style.format({k:v for k,v in fmt3.items() if k in av3}),
                 use_container_width=True, height=420)

# ─── TAB 4: RISK & SENTIMENT ──────────────────────────────────────────────────────
with tab_risk:
    cols4 = ["Ticker","Risk Score","30D Vol","Dist 52W Hi","Max DD",
             "Sentiment","News Sent","Analyst Rec","EPS Surprise","Cap Reason"]
    fmt4  = {"Risk Score":"{:.1f}","30D Vol":"{:.1%}","Dist 52W Hi":"{:.1%}",
             "Max DD":"{:.1%}","Sentiment":"{:.1f}","News Sent":"{:.1f}",
             "Analyst Rec":"{:.2f}","EPS Surprise":"{:.1%}"}
    av4 = [c for c in cols4 if c in df_view.columns]
    st.dataframe(df_view[av4].style.format({k:v for k,v in fmt4.items() if k in av4}),
                 use_container_width=True, height=420)

# ─── TAB 5: WATCHLIST ─────────────────────────────────────────────────────────────
with tab_watch:
    st.subheader("⭐ Your Watchlist")
    wl = st.session_state["watchlist"]

    # Add manually
    new_watch = st.text_input("Add ticker to watchlist", key="wl_add")
    if st.button("Add", key="wl_add_btn") and new_watch.strip():
        wl.add(new_watch.strip().upper())

    if not wl:
        st.info("No stocks in your watchlist yet. Add them from the Signals tab or above.")
    else:
        wl_df = df_all[df_all["Ticker"].isin(wl)].copy()
        if wl_df.empty:
            st.warning("Watchlist tickers not in current analysis. Run analysis with these tickers.")
            for t in sorted(wl):
                col_a, col_b = st.columns([3,1])
                col_a.write(t)
                if col_b.button("Remove", key=f"rm_{t}"):
                    wl.discard(t)
        else:
            wl_df["Signal"] = wl_df["Signal"].apply(lambda s: f"{ICON.get(s,'')} {s}")
            st.dataframe(wl_df[["Ticker","Last Price","Composite","Signal"]].style.format(
                {"Last Price":"{:.2f}","Composite":"{:.1f}"}),
                use_container_width=True)
            for t in sorted(wl):
                if st.button(f"Remove {t}", key=f"rmwl_{t}"):
                    wl.discard(t)

# ─── TAB 6: PORTFOLIO OVERLAY ──────────────────────────────────────────────────────
with tab_port:
    st.subheader("💼 Portfolio Overlay")
    st.caption("Enter your holdings to see risk concentration and signal alignment.")

    owned = st.session_state["owned"]

    # Add holding
    pa, pb, pc, pd_ = st.columns(4)
    pt = pa.text_input("Ticker", key="pt")
    pq = pb.number_input("Quantity", min_value=0.0, step=1.0, key="pq")
    pp = pc.number_input("Avg buy price", min_value=0.0, step=0.01, key="pp")
    if pd_.button("Add holding", key="padd") and pt.strip():
        owned[pt.strip().upper()] = {"qty": pq, "avg_price": pp}

    if not owned:
        st.info("No holdings added yet.")
    else:
        port_rows = []
        for t, h in owned.items():
            row_sig = df_all[df_all["Ticker"]==t]
            if not row_sig.empty:
                r = row_sig.iloc[0]
                current_price = r["Last Price"]
                value   = h["qty"] * current_price
                cost    = h["qty"] * h["avg_price"]
                pnl     = value - cost
                pnl_pct = (pnl/cost) if cost>0 else 0
                port_rows.append({
                    "Ticker":    t,
                    "Qty":       h["qty"],
                    "Avg Price": h["avg_price"],
                    "Curr Price":current_price,
                    "Value ($)":  round(value,2),
                    "P&L ($)":    round(pnl,2),
                    "P&L %":      pnl_pct,
                    "Signal":    f"{ICON.get(r['Signal'],'')} {r['Signal']}",
                    "Composite": r["Composite"],
                    "Risk Score":r["Risk Score"],
                    "30D Vol":   r["30D Vol"],
                })
            else:
                port_rows.append({"Ticker":t,"Qty":h["qty"],
                    "Avg Price":h["avg_price"],"Curr Price":"N/A",
                    "Value ($)":"N/A","P&L ($)":"N/A","P&L %":"N/A",
                    "Signal":"Run analysis","Composite":"N/A",
                    "Risk Score":"N/A","30D Vol":"N/A"})

        port_df = pd.DataFrame(port_rows)
        num_cols = {"Avg Price":"{:.2f}","Curr Price":"{:.2f}",
                    "Value ($)":"{:.2f}","P&L ($)":"{:.2f}",
                    "P&L %":"{:.1%}","Composite":"{:.1f}","Risk Score":"{:.1f}","30D Vol":"{:.1%}"}
        fmt_avail = {k:v for k,v in num_cols.items() if k in port_df.columns
                     and port_df[k].apply(lambda x: isinstance(x,(int,float))).all()}
        st.dataframe(port_df.style.format(fmt_avail), use_container_width=True)

        # Concentration warnings
        numeric_port = port_df[port_df["Value ($)"].apply(lambda x: isinstance(x,(int,float)))]
        if not numeric_port.empty:
            total_val = numeric_port["Value ($)"].sum()
            numeric_port = numeric_port.copy()
            numeric_port["Weight"] = numeric_port["Value ($)"] / total_val
            st.markdown("**Concentration check**")
            heavy = numeric_port[numeric_port["Weight"]>0.15]
            if not heavy.empty:
                for _, r in heavy.iterrows():
                    st.warning(f"⚠️ {r['Ticker']} is {r['Weight']:.0%} of portfolio (>15%)")
            high_risk = numeric_port[numeric_port["30D Vol"].apply(lambda x: isinstance(x,float) and x>0.5)]
            if not high_risk.empty:
                st.warning(f"⚠️ High-volatility holdings: {', '.join(high_risk['Ticker'].tolist())}")

        # Remove holding
        to_rm = st.selectbox("Remove a holding", [""] + list(owned.keys()), key="prm")
        if st.button("Remove", key="prm_btn") and to_rm:
            owned.pop(to_rm, None)

# ─── TAB 7: BACKTEST ──────────────────────────────────────────────────────────────with tab_bt:
    st.subheader("🕰️ Simple Backtest")
    st.caption(
        "Equal-weight portfolio of current Buy/Strong Buy signals vs SPY benchmark. "
        "Based on 1-year historical returns."
    )
    if st.button("▶️ Run backtest"):
        with st.spinner("Running backtest..."):
            cum_port, cum_both = simple_backtest(charts, df_all)
        if cum_both is not None:
            st.line_chart(cum_both, height=350, use_container_width=True)
            final = cum_both.iloc[-1]
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Portfolio total return", f"{final['Portfolio']:.1%}")
            bc2.metric("SPY total return",       f"{final['SPY']:.1%}")
            bc3.metric("Alpha", f"{final['Portfolio']-final['SPY']:.1%}")
        elif cum_port is not None:
            st.line_chart(cum_port, height=300, use_container_width=True)
        else:
            st.warning("Not enough Buy/Strong Buy signals to run a backtest.")
