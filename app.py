import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import pickle
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0e1117; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
[data-testid="metric-container"] {
    background: #1c2333;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 12px 16px;
}
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: #1c2333;
    border-radius: 8px 8px 0 0;
    padding: 6px 16px;
    font-size: 13px;
}
.stTabs [aria-selected="true"] { background: #2d3748 !important; }
.card {
    background: #1c2333;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
[data-testid="stSidebar"] { background: #141820; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
for key, default in [
    ("watchlist",    set()),
    ("owned",        {}),
    ("results",      None),
    ("charts",       {}),
    ("prev_signals", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── ML MODEL ─────────────────────────────────────────────────────────────────
_ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")

@st.cache_resource
def load_ml_model():
    try:
        with open(os.path.join(_ARTIFACTS, "model.pkl"),  "rb") as f: model  = pickle.load(f)
        with open(os.path.join(_ARTIFACTS, "scaler.pkl"), "rb") as f: scaler = pickle.load(f)
        with open(os.path.join(_ARTIFACTS, "meta.pkl"),   "rb") as f: meta   = pickle.load(f)
        return model, scaler, meta
    except FileNotFoundError:
        return None, None, None

ML_MODEL, ML_SCALER, ML_META = load_ml_model()

def get_ml_prob(m1, m3, m6, m12, ama50, ama200, vol, mdd, d52):
    if ML_MODEL is None: return 0.5
    feat = np.array([[m1, m3, m6, m12, float(ama50), float(ama200), vol, mdd, d52]])
    return float(ML_MODEL.predict_proba(ML_SCALER.transform(feat))[0, 1])

# ─── HELPERS ──────────────────────────────────────────────────────────────────
SIGNAL_ICON  = {"Strong Buy": "🟢", "Buy": "🟡", "Hold": "🔵", "Reduce": "🟠", "Sell": "🔴"}
SIGNAL_COLOR = {
    "Strong Buy": "#00c853", "Buy": "#76ff03",
    "Hold": "#40c4ff", "Reduce": "#ff9100", "Sell": "#ff1744"
}

def sig_badge(s):
    c = SIGNAL_COLOR.get(s, "#aaa")
    icon = SIGNAL_ICON.get(s, "")
    return (
        f'<span style="background:{c}22;color:{c};border:1px solid {c}55;'
        f'border-radius:6px;padding:2px 8px;font-size:12px;font-weight:700">'
        f'{icon} {s}</span>'
    )

def signal_from_score(s):
    if s >= 75: return "Strong Buy"
    if s >= 60: return "Buy"
    if s >= 45: return "Hold"
    if s >= 30: return "Reduce"
    return "Sell"

def safe_series(df, col):
    s = df[col].squeeze()
    if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
    return s.dropna()

def norm(val, lo, hi):
    if val is None or (hi - lo) == 0: return 50
    return float(min(max((val - lo) / (hi - lo) * 100, 0), 100))

SECTOR_MEDIAN_PE = {
    "Technology": 28, "Consumer Cyclical": 22, "Healthcare": 20,
    "Financial Services": 14, "Communication Services": 18,
    "Industrials": 20, "Consumer Defensive": 22, "Energy": 12,
    "Basic Materials": 15, "Real Estate": 35, "Utilities": 18, "default": 20
}

# ─── FACTOR ENGINES ───────────────────────────────────────────────────────────
def momentum_score(close):
    last  = float(close.iloc[-1])
    m1    = float(close.iloc[-1]/close.iloc[-21]  - 1) if len(close) >= 21  else 0.0
    m3    = float(close.iloc[-1]/close.iloc[-63]  - 1) if len(close) >= 63  else 0.0
    m6    = float(close.iloc[-1]/close.iloc[-126] - 1) if len(close) >= 126 else 0.0
    m12   = float(close.iloc[-1]/close.iloc[0]    - 1)
    ma50  = float(close.tail(50).mean())
    ma200 = float(close.tail(200).mean()) if len(close) >= 200 else float(close.mean())
    ama50, ama200 = last > ma50, last > ma200
    trend = (10 if ama50 else 0) + (10 if ama200 else 0)
    base  = norm(m1 * 0.20 + m6 * 0.40 + m12 * 0.40, -0.50, 0.80)
    return min(base + trend, 100), {
        "m1": m1, "m3": m3, "m6": m6, "m12": m12,
        "above_ma50": ama50, "above_ma200": ama200,
        "ma50": ma50, "ma200": ma200
    }

def value_score(info, sector="default"):
    pe   = info.get("trailingPE") or info.get("forwardPE")
    ps   = info.get("priceToSalesTrailing12Months")
    ev_e = info.get("enterpriseToEbitda")
    mkt  = info.get("marketCap")
    fcf  = info.get("freeCashflow")
    fcfy = (fcf / mkt) if (fcf and mkt and mkt > 0) else None
    spe  = SECTOR_MEDIAN_PE.get(sector, SECTOR_MEDIAN_PE["default"])
    pts  = []
    if pe   and pe   > 0: pts.append(max(min((2 - pe / spe) * 50, 100), 0))
    if ps   and ps   > 0: pts.append(100 - norm(ps,   0.5, 20))
    if ev_e and ev_e > 0: pts.append(100 - norm(ev_e, 3,   40))
    if fcfy:              pts.append(norm(fcfy, -0.05, 0.15))
    score = float(np.mean(pts)) if pts else 50
    return max(min(score, 100), 0), {
        "pe": pe, "ps": ps, "ev_ebitda": ev_e,
        "fcf_yield": fcfy, "sector": sector, "sector_median_pe": spe
    }

def quality_score(info):
    roe    = info.get("returnOnEquity")
    margin = info.get("profitMargins")
    de     = info.get("debtToEquity")
    cr     = info.get("currentRatio")
    ocf    = info.get("operatingCashflow")
    ni     = info.get("netIncomeToCommon")
    accrual = (ocf / abs(ni)) if (ocf and ni and ni != 0) else None
    pts = []
    if roe     is not None: pts.append(norm(roe,     -0.10, 0.40))
    if margin  is not None: pts.append(norm(margin,  -0.05, 0.35))
    if de      is not None: pts.append(100 - norm(de, 0, 200))
    if cr      is not None: pts.append(norm(cr, 0.5, 3.0))
    if accrual is not None: pts.append(norm(accrual,  0.5, 2.5))
    score = float(np.mean(pts)) if pts else 50
    return max(min(score, 100), 0), {
        "roe": roe, "profit_margin": margin,
        "debt_equity": de, "current_ratio": cr, "accrual_ratio": accrual
    }

def risk_score_fn(close):
    ret    = close.pct_change().dropna()
    vol_30 = float(ret.tail(30).std() * (252 ** 0.5))
    dist   = float(close.iloc[-1] / close.max() - 1)
    max_dd = float(((close - close.cummax()) / close.cummax()).min())
    score  = (
        (100 - norm(vol_30, 0.10, 0.80)) * 0.4
        + norm(dist, -0.50, 0.0) * 0.3
        + (100 - norm(abs(max_dd), 0, 0.60)) * 0.3
    )
    return max(min(score, 100), 0), {
        "vol_30d": vol_30, "dist_52w_high": dist, "max_dd": max_dd
    }

def sentiment_score_fn(info, ticker, news_key):
    rec    = info.get("recommendationMean")
    eps_s  = info.get("earningsQuarterlyGrowth")
    news_sent, headlines = 50, []
    if news_key:
        try:
            since = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            url   = (
                f"https://newsapi.org/v2/everything?q={ticker}&language=en"
                f"&sortBy=publishedAt&from={since}&pageSize=10&apiKey={news_key}"
            )
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                articles  = r.json().get("articles", [])
                headlines = [a["title"] for a in articles if a.get("title")]
                pos_kw = ["beat","surge","strong","record","growth","upgraded","outperform","rally","profit"]
                neg_kw = ["miss","fall","weak","loss","downgrade","underperform","crash","cut","decline"]
                pos = sum(1 for h in headlines for w in pos_kw if w in h.lower())
                neg = sum(1 for h in headlines for w in neg_kw if w in h.lower())
                if (pos + neg) > 0:
                    news_sent = (pos / (pos + neg)) * 100
        except Exception:
            pass
    pts = []
    if rec:   pts.append(100 - norm(rec, 1, 5))
    if eps_s: pts.append(norm(eps_s, -0.30, 0.50))
    pts.append(news_sent)
    score = float(np.mean(pts)) if pts else 50
    return max(min(score, 100), 0), {
        "recommendation": rec, "earnings_surprise": eps_s,
        "news_sentiment": news_sent, "headlines": headlines
    }

def apply_hard_caps(signal, score, vol_30d, dist_52w_high, max_vol_cap, max_dd_cap):
    reason = ""
    if vol_30d > max_vol_cap:
        reason = f"Vol {vol_30d:.0%} > cap {max_vol_cap:.0%}"
    if dist_52w_high < max_dd_cap:
        reason += (" | " if reason else "") + f"DD {dist_52w_high:.0%} < cap {max_dd_cap:.0%}"
    if reason and signal in ["Strong Buy", "Buy"]:
        return "Hold", min(score, 44), f"⚠️ {reason}"
    return signal, score, ""

# ─── CORE ANALYSIS ────────────────────────────────────────────────────────────
def analyse_ticker(ticker, period, w_mom, w_val, w_qual, w_risk,
                   news_key, max_vol_cap, max_dd_cap):
    try:
        tk     = yf.Ticker(ticker)
        df     = tk.history(period=period, auto_adjust=True)
        if df.empty or len(df) < 60:
            return None, None
        close  = safe_series(df, "Close")
        info   = tk.info or {}
        sector = info.get("sector", "default")
        name   = info.get("shortName", ticker)

        ms, mm = momentum_score(close)
        vs, vm = value_score(info, sector)
        qs, qm = quality_score(info)
        rs, rm = risk_score_fn(close)
        ss, sm = sentiment_score_fn(info, ticker, news_key)

        ml_prob      = get_ml_prob(
            mm["m1"], mm["m3"], mm["m6"], mm["m12"],
            mm["above_ma50"], mm["above_ma200"],
            rm["vol_30d"], rm["max_dd"], rm["dist_52w_high"]
        )
        ml_score_100 = round(ml_prob * 100, 1)

        total_w    = w_mom + w_val + w_qual + w_risk or 100
        rules_comp = (ms * w_mom + vs * w_val + qs * w_qual + rs * w_risk) / total_w
        comp       = round(min(max(rules_comp * 0.70 + ml_score_100 * 0.30, 0), 100), 1)

        sig = signal_from_score(comp)
        sig, comp, cap = apply_hard_caps(
            sig, comp, rm["vol_30d"], rm["dist_52w_high"], max_vol_cap, max_dd_cap
        )
        return {
            "Ticker":      ticker,   "Name":       name,
            "Sector":      sector,   "Market Cap": info.get("marketCap", 0) or 0,
            "Last Price":  round(float(close.iloc[-1]), 2),
            "Mom Score":   round(ms, 1),  "Value Score": round(vs, 1),
            "Qual Score":  round(qs, 1),  "Risk Score":  round(rs, 1),
            "Sentiment":   round(ss, 1),  "ML Prob":     round(ml_prob, 3),
            "ML Score":    ml_score_100,  "Composite":   comp,
            "Signal":      sig,           "Cap Reason":  cap,
            "1M Mom":      mm["m1"],  "3M Mom":  mm["m3"],
            "6M Mom":      mm["m6"],  "12M Mom": mm["m12"],
            "Above MA50":  "✅" if mm["above_ma50"]  else "❌",
            "Above MA200": "✅" if mm["above_ma200"] else "❌",
            "P/E":         vm.get("pe"),        "P/S":       vm.get("ps"),
            "EV/EBITDA":   vm.get("ev_ebitda"), "FCF Yield": vm.get("fcf_yield"),
            "Sector PE":   vm.get("sector_median_pe"),
            "ROE":         qm.get("roe"),          "Margin":    qm.get("profit_margin"),
            "D/E":         qm.get("debt_equity"),  "Curr Ratio":qm.get("current_ratio"),
            "Accrual":     qm.get("accrual_ratio"),
            "30D Vol":     rm["vol_30d"],       "Dist 52W Hi": rm["dist_52w_high"],
            "Max DD":      rm["max_dd"],
            "Analyst Rec": sm.get("recommendation"), "EPS Surprise": sm.get("earnings_surprise"),
            "News Sent":   sm.get("news_sentiment"),  "Headlines":    sm.get("headlines", []),
            "_close":      close,
        }, df
    except Exception:
        return None, None

# ─── DISCOVERY UNIVERSE ───────────────────────────────────────────────────────
DISCOVERY_UNIVERSE = list(dict.fromkeys([
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AMD","AVGO","QCOM",
    "INTC","TXN","MU","AMAT","KLAC","LRCX","CRWD","PANW","SNOW","PLTR",
    "LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","ISRG","DXCM","MRNA",
    "JPM","BAC","GS","MS","V","MA","AXP","BLK","SCHW","COF",
    "COST","WMT","MCD","SBUX","NKE","TGT","HD","LOW","ABNB",
    "XOM","CVX","COP","SLB","EOG","MPC","VLO",
    "CAT","DE","HON","RTX","BA","GE","UPS","FDX","ETN","ROK",
    "NFLX","DIS","SPOT","SNAP","PINS","TTD",
    "CELH","HIMS","RXRX","IONQ","RKLB","JOBY","UPST","SQ","HOOD",
    "COIN","MSTR","SMCI","APP","DUOL","SOUN","BBAI","ENPH","FSLR",
    "ASML","TSM","BABA","JD","NVO","SAP","SE","GRAB","NU","MELI",
]))

DAY_TRADE_UNIVERSE = [
    "NVDA","TSLA","AMD","AAPL","META","AMZN","GOOGL","MSFT","SPY","QQQ",
    "COIN","MSTR","PLTR","SMCI","HOOD","SQ","SNAP","RKLB","IONQ","UPST",
    "APP","CRWD","PANW","MU","AVGO","NFLX","CELH","HIMS","SOUN","BBAI",
]

@st.cache_data(ttl=1800)
def run_discovery_scan(universe, scan_type="growth"):
    results = []
    for ticker in universe:
        try:
            tk    = yf.Ticker(ticker)
            df    = tk.history(period="6mo", auto_adjust=True)
            if df.empty or len(df) < 30: continue
            close = safe_series(df, "Close")
            info  = tk.info or {}
            vol   = safe_series(df, "Volume")
            price = float(close.iloc[-1])
            m1    = float(close.iloc[-1]/close.iloc[-21]  - 1) if len(close) >= 21 else 0
            m3    = float(close.iloc[-1]/close.iloc[-63]  - 1) if len(close) >= 63 else 0
            m6    = float(close.iloc[-1]/close.iloc[0]    - 1)
            ma50  = float(close.tail(50).mean())
            ma200 = float(close.tail(200).mean()) if len(close) >= 200 else float(close.mean())
            ret   = close.pct_change().dropna()
            vol30 = float(ret.tail(30).std() * (252 ** 0.5))
            dist  = float(close.iloc[-1] / close.max() - 1)
            avg_vol  = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
            rvol     = float(vol.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0
            hi   = safe_series(df, "High").tail(15)
            lo   = safe_series(df, "Low").tail(15)
            atr  = float((hi - lo).tail(14).mean())
            atr_pct = atr / price if price > 0 else 0
            name   = info.get("shortName", ticker)
            sector = info.get("sector", "Unknown")
            mktcap = info.get("marketCap", 0) or 0
            if scan_type == "day_trade":
                score = (
                    norm(rvol,    0.5, 5.0)  * 0.35
                    + norm(atr_pct, 0.01, 0.08) * 0.30
                    + norm(m1,     -0.15, 0.20) * 0.20
                    + norm(vol30,   0.20, 1.20) * 0.15
                )
                results.append({
                    "Ticker": ticker, "Name": name, "Sector": sector,
                    "Price": round(price, 2), "Day Score": round(score, 1),
                    "RVOL": round(rvol, 2), "ATR %": round(atr_pct * 100, 2),
                    "1M Mom": m1, "Vol 30D": round(vol30 * 100, 1),
                    "Above MA50": "✅" if price > ma50 else "❌",
                    "Market Cap": mktcap,
                })
            else:
                golden      = price > ma50 > ma200
                breakout_52 = dist > -0.05
                eps_g = info.get("earningsQuarterlyGrowth") or 0
                rev_g = info.get("revenueGrowth") or 0
                pe    = info.get("trailingPE") or info.get("forwardPE") or 0
                score = min(
                    norm(m3,    -0.20, 0.60) * 0.25
                    + norm(m6,  -0.30, 0.80) * 0.25
                    + norm(eps_g,-0.10, 0.50) * 0.20
                    + norm(rev_g,-0.05, 0.40) * 0.15
                    + (10 if golden else 0)
                    + (5  if breakout_52 else 0),
                    100
                )
                results.append({
                    "Ticker": ticker, "Name": name, "Sector": sector,
                    "Price": round(price, 2), "Growth Score": round(score, 1),
                    "3M Mom": m3, "6M Mom": m6,
                    "EPS Growth": eps_g, "Rev Growth": rev_g,
                    "Golden Cross": "✅" if golden else "❌",
                    "Near 52W Hi": "✅" if breakout_52 else "❌",
                    "P/E": pe if pe and pe > 0 else None,
                    "Market Cap": mktcap,
                })
        except Exception:
            continue
    if not results: return pd.DataFrame()
    score_col = "Day Score" if scan_type == "day_trade" else "Growth Score"
    return pd.DataFrame(results).sort_values(score_col, ascending=False).reset_index(drop=True)

# ─── ETF ENGINE ───────────────────────────────────────────────────────────────
ETF_UNIVERSE = {
    "SPY":  {"name": "S&P 500",        "category": "Broad",    "exp": 0.0945},
    "QQQ":  {"name": "Nasdaq 100",      "category": "Broad",    "exp": 0.20},
    "IWM":  {"name": "Russell 2000",    "category": "Broad",    "exp": 0.19},
    "VTI":  {"name": "Total US Market", "category": "Broad",    "exp": 0.03},
    "VT":   {"name": "Total World",     "category": "Broad",    "exp": 0.07},
    "XLK":  {"name": "Tech",            "category": "Sector",   "exp": 0.10},
    "XLF":  {"name": "Financials",      "category": "Sector",   "exp": 0.10},
    "XLE":  {"name": "Energy",          "category": "Sector",   "exp": 0.10},
    "XLV":  {"name": "Healthcare",      "category": "Sector",   "exp": 0.10},
    "XLI":  {"name": "Industrials",     "category": "Sector",   "exp": 0.10},
    "XLC":  {"name": "Comm Services",   "category": "Sector",   "exp": 0.10},
    "XLY":  {"name": "Consumer Discr.", "category": "Sector",   "exp": 0.10},
    "XLRE": {"name": "Real Estate",     "category": "Sector",   "exp": 0.10},
    "SOXX": {"name": "Semiconductors",  "category": "Thematic", "exp": 0.35},
    "ARKK": {"name": "ARK Innovation",  "category": "Thematic", "exp": 0.75},
    "ARKG": {"name": "ARK Genomic",     "category": "Thematic", "exp": 0.75},
    "AIQ":  {"name": "AI & Big Data",   "category": "Thematic", "exp": 0.68},
    "BOTZ": {"name": "Robotics & AI",   "category": "Thematic", "exp": 0.69},
    "CIBR": {"name": "Cybersecurity",   "category": "Thematic", "exp": 0.60},
    "ICLN": {"name": "Clean Energy",    "category": "Thematic", "exp": 0.41},
    "TLT":  {"name": "20Y Treasury",    "category": "Bonds",    "exp": 0.15},
    "HYG":  {"name": "High Yield Bond", "category": "Bonds",    "exp": 0.48},
    "AGG":  {"name": "US Agg Bond",     "category": "Bonds",    "exp": 0.03},
    "GLD":  {"name": "Gold",            "category": "Commodity","exp": 0.40},
    "SLV":  {"name": "Silver",          "category": "Commodity","exp": 0.50},
    "USO":  {"name": "Oil",             "category": "Commodity","exp": 0.81},
}

@st.cache_data(ttl=1800)
def run_etf_scan():
    rows = []
    for ticker, meta in ETF_UNIVERSE.items():
        try:
            tk    = yf.Ticker(ticker)
            df    = tk.history(period="1y", auto_adjust=True)
            if df.empty or len(df) < 30: continue
            close = safe_series(df, "Close")
            vol_s = safe_series(df, "Volume")
            price = float(close.iloc[-1])
            m1    = float(close.iloc[-1]/close.iloc[-21]  - 1) if len(close) >= 21  else 0
            m3    = float(close.iloc[-1]/close.iloc[-63]  - 1) if len(close) >= 63  else 0
            m6    = float(close.iloc[-1]/close.iloc[-126] - 1) if len(close) >= 126 else 0
            m12   = float(close.iloc[-1]/close.iloc[0]    - 1)
            ma50  = float(close.tail(50).mean())
            ma200 = float(close.tail(200).mean()) if len(close) >= 200 else float(close.mean())
            ret   = close.pct_change().dropna()
            vol30 = float(ret.tail(30).std() * (252 ** 0.5))
            dist  = float(close.iloc[-1] / close.max() - 1)
            max_dd = float(((close - close.cummax()) / close.cummax()).min())
            avg_vol = float(vol_s.tail(20).mean())
            rvol    = float(vol_s.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0
            mom_s   = norm(m1*0.20 + m3*0.30 + m6*0.30 + m12*0.20, -0.30, 0.60)
            risk_s  = (100 - norm(vol30, 0.05, 0.50)) * 0.5 + norm(dist, -0.40, 0.0) * 0.5
            exp_pen = max(0, 100 - meta["exp"] * 100)
            etf_score = round(mom_s * 0.55 + risk_s * 0.35 + exp_pen * 0.10, 1)
            trend = "✅" if price > ma50 > ma200 else ("➡️" if price > ma50 else "❌")
            rows.append({
                "Ticker": ticker, "Name": meta["name"], "Category": meta["category"],
                "Price": round(price, 2), "ETF Score": etf_score,
                "1M": m1, "3M": m3, "6M": m6, "12M": m12,
                "Vol 30D": round(vol30 * 100, 1), "Max DD": max_dd,
                "Dist 52W Hi": dist, "RVOL": round(rvol, 2),
                "Exp Ratio %": meta["exp"], "Trend": trend,
                "_close": close,
            })
        except Exception:
            continue
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ETF Score", ascending=False).reset_index(drop=True)

# ─── BACKTEST ─────────────────────────────────────────────────────────────────
def run_backtest(charts, df_results):
    buy_tickers = df_results[
        df_results["Signal"].isin(["Strong Buy", "Buy"])
    ]["Ticker"].tolist()
    if not buy_tickers: return None, None, None
    returns = {}
    for t in buy_tickers:
        df = charts.get(t)
        if df is None: continue
        s = safe_series(df, "Close").pct_change().dropna()
        returns[t] = s
    if not returns: return None, None, None
    port     = pd.DataFrame(returns).dropna().mean(axis=1)
    cum_port = (1 + port).cumprod() - 1
    cum_both = None
    try:
        spy     = yf.download("SPY", period="1y", auto_adjust=True, progress=False)
        spy_ret = safe_series(spy, "Close").pct_change().dropna()
        spy_ret.index = spy_ret.index.tz_localize(None) if spy_ret.index.tzinfo else spy_ret.index
        port.index    = port.index.tz_localize(None)    if port.index.tzinfo    else port.index
        combined  = pd.DataFrame({"Portfolio": port, "SPY": spy_ret}).dropna()
        cum_both  = (1 + combined).cumprod() - 1
    except Exception:
        pass
    ann_ret  = float((1 + port.mean()) ** 252 - 1)
    ann_vol  = float(port.std() * (252 ** 0.5))
    sharpe   = round(ann_ret / ann_vol, 2) if ann_vol > 0 else 0
    neg      = port[port < 0]
    sortino  = round(ann_ret / (neg.std() * (252 ** 0.5)), 2) if len(neg) > 0 else 0
    mdd      = float(((cum_port + 1 - (cum_port + 1).cummax()) / (cum_port + 1).cummax()).min())
    win_rate = round((port > 0).mean() * 100, 1)
    stats    = {
        "ann_ret": ann_ret, "ann_vol": ann_vol,
        "sharpe": sharpe, "sortino": sortino,
        "max_dd": mdd, "win_rate": win_rate
    }
    return cum_port, cum_both, stats

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Stock Intelligence")
    st.markdown("---")
    st.markdown("### 🎯 Tickers")
    tickers_str = st.text_area(
        "One per line or comma-separated",
        value="AAPL\nMSFT\nNVDA\nGOOGL\nASML\nTSLA\nAMZN\nMETA",
        height=130
    )
    period    = st.selectbox("Lookback period", ["6mo", "1y", "2y"], index=1)
    min_score = st.slider("Min composite score", 0, 100, 0, 5)

    with st.expander("⚖️ Factor weights", expanded=False):
        w_mom  = st.slider("Momentum",  0, 100, 30, 5)
        w_val  = st.slider("Value",     0, 100, 25, 5)
        w_qual = st.slider("Quality",   0, 100, 25, 5)
        w_risk = st.slider("Risk",      0, 100, 20, 5)
        tw = w_mom + w_val + w_qual + w_risk
        st.success(f"Total: {tw} ✓") if tw == 100 else st.warning(f"Total: {tw} (should be 100)")

    with st.expander("🚨 Risk caps", expanded=False):
        max_vol = st.slider("Max annualised 30D vol", 0.20, 1.50, 0.80, 0.05)
        max_dd  = st.slider("Max drawdown from 52W hi", -0.60, -0.05, -0.25, 0.05)

    with st.expander("📰 News & ML", expanded=False):
        news_key = st.text_input("NewsAPI key (optional)", type="password")
        if ML_MODEL is not None:
            st.success(f"🤖 ElasticNet active — AUC: {ML_META.get('test_auc', 0):.3f}")
        else:
            st.warning("⚠️ Run `python train_model.py`")

    st.markdown("---")
    run_btn = st.button("🔄 Run Analysis", use_container_width=True, type="primary")
    st.caption("Educational tool only — not investment advice.")

# ─── PARSE TICKERS ────────────────────────────────────────────────────────────
tickers = []
for t in tickers_str.replace(",", "\n").split("\n"):
    t = t.strip().upper()
    if t:
        tickers.append(t)

# ─── RUN ANALYSIS ─────────────────────────────────────────────────────────────
if run_btn:
    if st.session_state["results"] is not None:
        prev = st.session_state["results"]
        st.session_state["prev_signals"] = dict(zip(prev["Ticker"], prev["Signal"]))
    rows, charts_tmp = [], {}
    prog = st.progress(0, text="Analysing...")
    for i, t in enumerate(tickers):
        prog.progress((i + 1) / len(tickers), text=f"Analysing {t} ({i+1}/{len(tickers)})...")
        row, df = analyse_ticker(t, period, w_mom, w_val, w_qual, w_risk,
                                  news_key, max_vol, max_dd)
        if row:
            rows.append(row)
            charts_tmp[t] = df
    prog.empty()
    if rows:
        st.session_state["results"] = (
            pd.DataFrame(rows)
            .sort_values("Composite", ascending=False)
            .reset_index(drop=True)
        )
        st.session_state["charts"] = charts_tmp
    else:
        st.error("No data returned. Check your tickers.")
        st.stop()

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_home, tab_signals, tab_discover, tab_etf, tab_portfolio, tab_backtest, tab_ml = st.tabs([
    "🏠 Home", "📊 Signals", "🔭 Discover",
    "🌐 ETF Hub", "💼 Portfolio", "📈 Backtest", "🤖 ML Insights",
])

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    st.markdown("### 👋 Welcome to Stock Intelligence")
    st.caption(f"Last updated: {datetime.now().strftime('%d %b %Y, %H:%M')}  •  Educational tool only")

    if st.session_state["results"] is None:
        st.info("👈 Add tickers in the sidebar and click **🔄 Run Analysis** to get started.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='card'><h4>📊 Multi-Factor Signals</h4>"
                        "<p>Momentum, Value, Quality, Risk & Sentiment blended with ElasticNet ML</p></div>",
                        unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='card'><h4>🔭 Stock Discovery</h4>"
                        "<p>Scan 100+ stocks for growth setups and day-trading opportunities</p></div>",
                        unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='card'><h4>🌐 ETF Hub</h4>"
                        "<p>Sector rotation, ETF momentum rankings, expense-ratio-adjusted scoring</p></div>",
                        unsafe_allow_html=True)
    else:
        df_all = st.session_state["results"]
        prev   = st.session_state["prev_signals"]
        sb = int((df_all["Signal"] == "Strong Buy").sum())
        b  = int((df_all["Signal"] == "Buy").sum())
        h  = int((df_all["Signal"] == "Hold").sum())
        r  = int((df_all["Signal"] == "Reduce").sum())
        s  = int((df_all["Signal"] == "Sell").sum())
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("🟢 Strong Buy", sb)
        c2.metric("🟡 Buy",        b)
        c3.metric("🔵 Hold",       h)
        c4.metric("🟠 Reduce",     r)
        c5.metric("🔴 Sell",       s)
        c6.metric("Avg Score",     f"{df_all['Composite'].mean():.1f}")
        st.markdown("---")

        # Signal changes
        changes = []
        order   = ["Sell", "Reduce", "Hold", "Buy", "Strong Buy"]
        for _, row in df_all.iterrows():
            old = prev.get(row["Ticker"])
            if old and old != row["Signal"]:
                arrow = "⬆️" if order.index(row["Signal"]) > order.index(old) else "⬇️"
                changes.append((row["Ticker"], old, row["Signal"], arrow))
        if changes:
            st.markdown("#### 🔔 Signal Changes")
            for t, old, new, arrow in changes:
                st.markdown(
                    f"**{t}** {arrow} {SIGNAL_ICON.get(old,'')} {old} → "
                    f"{SIGNAL_ICON.get(new,'')} {new}"
                )
            st.markdown("---")

        # Top picks
        st.markdown("#### 🏆 Top Picks")
        top = df_all[df_all["Signal"].isin(["Strong Buy", "Buy"])].head(6)
        if top.empty:
            st.info("No Buy/Strong Buy signals.")
        else:
            cols = st.columns(min(len(top), 3))
            for idx, (_, row) in enumerate(top.iterrows()):
                sig_c = SIGNAL_COLOR.get(row["Signal"], "#aaa")
                mom_val = row["1M Mom"]
                pct_str = f"{mom_val:+.1%}" if isinstance(mom_val, float) else "N/A"
                col_str = "#00c853" if isinstance(mom_val, float) and mom_val > 0 else "#ff1744"
                with cols[idx % 3]:
                    st.markdown(
                        f"<div style='background:#1c2333;border:1px solid {sig_c}44;"
                        f"border-left:4px solid {sig_c};border-radius:10px;"
                        f"padding:14px;margin-bottom:10px'>"
                        f"<div style='font-size:18px;font-weight:700;color:{sig_c}'>{row['Ticker']}</div>"
                        f"<div style='font-size:11px;color:#8892a4;margin-bottom:6px'>{row.get('Name','')}</div>"
                        f"<div style='display:flex;justify-content:space-between'>"
                        f"<span style='font-size:20px;font-weight:700'>${row['Last Price']:.2f}</span>"
                        f"<span style='font-size:13px;color:{sig_c}'>"
                        f"{SIGNAL_ICON.get(row['Signal'],'')} {row['Signal']}</span></div>"
                        f"<div style='display:flex;gap:12px;margin-top:8px;font-size:12px;color:#8892a4'>"
                        f"<span>Score: <b style='color:#eee'>{row['Composite']}</b></span>"
                        f"<span>1M: <b style='color:{col_str}'>{pct_str}</b></span>"
                        f"<span>Vol: <b style='color:#eee'>{row['30D Vol']:.0%}</b></span>"
                        f"</div></div>",
                        unsafe_allow_html=True
                    )

        # Watchlist
        wl = st.session_state["watchlist"]
        if wl:
            st.markdown("---")
            st.markdown("#### ⭐ Watchlist")
            wl_df = df_all[df_all["Ticker"].isin(wl)]
            if not wl_df.empty:
                for _, row in wl_df.iterrows():
                    st.markdown(
                        f"**{row['Ticker']}** `${row['Last Price']:.2f}` &nbsp;"
                        f"{sig_badge(row['Signal'])} &nbsp; Score: **{row['Composite']}**",
                        unsafe_allow_html=True
                    )

# ══════════════════════════════════════════════════════════════════════════════
# SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
with tab_signals:
    if st.session_state["results"] is None:
        st.info("Run analysis first.")
    else:
        df_all  = st.session_state["results"]
        charts  = st.session_state["charts"]
        df_view = df_all[df_all["Composite"] >= min_score].copy()

        left, right = st.columns([3, 2])

        with left:
            fc1, fc2 = st.columns(2)
            sig_filter = fc1.multiselect(
                "Signal",
                ["Strong Buy", "Buy", "Hold", "Reduce", "Sell"],
                default=["Strong Buy", "Buy", "Hold", "Reduce", "Sell"]
            )
            sec_filter = fc2.multiselect(
                "Sector",
                sorted(df_view["Sector"].unique().tolist()),
                default=[]
            )
            df_filt = df_view[df_view["Signal"].isin(sig_filter)]
            if sec_filter:
                df_filt = df_filt[df_filt["Sector"].isin(sec_filter)]

            csv_data = df_filt.drop(columns=["Headlines", "_close"], errors="ignore").to_csv(index=False)
            st.download_button("⬇️ Export CSV", csv_data, "signals.csv", "text/csv", key="dl_sig")

            disp = df_filt[[
                "Ticker", "Name", "Sector", "Last Price",
                "Mom Score", "Value Score", "Qual Score", "Risk Score",
                "Sentiment", "ML Score", "Composite", "Signal", "Cap Reason"
            ]].copy()
            disp["Signal"] = disp["Signal"].apply(lambda s: f"{SIGNAL_ICON.get(s, '')} {s}")
            st.dataframe(
                disp.style
                    .format({
                        "Last Price": "{:.2f}", "Mom Score": "{:.1f}",
                        "Value Score": "{:.1f}", "Qual Score": "{:.1f}",
                        "Risk Score": "{:.1f}", "Sentiment": "{:.1f}",
                        "ML Score": "{:.1f}", "Composite": "{:.1f}",
                    })
                    .background_gradient(subset=["Composite"], cmap="RdYlGn", vmin=0, vmax=100),
                use_container_width=True, height=480
            )

        with right:
            st.markdown("#### 🔍 Deep Dive")
            tlist = df_filt["Ticker"].tolist()
            if not tlist:
                st.info("No tickers match filters.")
            else:
                sel     = st.selectbox("Select ticker", tlist, key="sel_dd")
                row     = df_filt[df_filt["Ticker"] == sel].iloc[0]
                wl      = st.session_state["watchlist"]
                prev    = st.session_state["prev_signals"]
                sig_c   = SIGNAL_COLOR.get(row["Signal"], "#aaa")
                old_sig = prev.get(sel)

                st.markdown(
                    f"<div style='background:#1c2333;border:1px solid {sig_c}44;"
                    f"border-left:4px solid {sig_c};border-radius:10px;padding:16px'>"
                    f"<div style='font-size:22px;font-weight:700'>{sel} "
                    f"<span style='color:{sig_c}'>"
                    f"{SIGNAL_ICON.get(row['Signal'], '')} {row['Signal']}</span></div>"
                    f"<div style='color:#8892a4;font-size:12px'>"
                    f"{row.get('Name', '')} · {row['Sector']}</div></div>",
                    unsafe_allow_html=True
                )

                if row["Cap Reason"]:
                    st.warning(row["Cap Reason"])
                if old_sig and old_sig != row["Signal"]:
                    st.info(f"🔔 Signal changed (was {old_sig})")

                # Factor metrics
                fa, fb = st.columns(2)
                fa.metric("🚀 Momentum",  f"{row['Mom Score']:.1f}")
                fb.metric("💰 Value",     f"{row['Value Score']:.1f}")
                fc, fd = st.columns(2)
                fc.metric("⭐ Quality",   f"{row['Qual Score']:.1f}")
                fd.metric("🛡️ Risk",      f"{row['Risk Score']:.1f}")
                fe, ff = st.columns(2)
                fe.metric("📰 Sentiment", f"{row['Sentiment']:.1f}")
                ml_label = f"{row['ML Score']:.1f}" if ML_MODEL else f"{row['ML Score']:.1f} (untrained)"
                ff.metric("🤖 ML Score",  ml_label)
                st.metric("🏆 Composite", f"{row['Composite']:.1f}")

                # Price chart
                df_price = charts.get(sel)
                if df_price is not None:
                    ps = safe_series(df_price, "Close")
                    st.line_chart(ps.tail(252), height=200, use_container_width=True)

                # Headlines
                headlines = row.get("Headlines", [])
                if headlines:
                    st.markdown("**Latest news**")
                    for h in headlines[:5]:
                        st.caption(f"• {h}")

                # Watchlist toggle
                if sel in wl:
                    if st.button(f"⭐ Remove {sel} from watchlist"):
                        wl.discard(sel)
                else:
                    if st.button(f"☆ Add {sel} to watchlist"):
                        wl.add(sel)

                # Extra tables
                st.markdown("---")
                st.markdown("**Momentum detail**")
                st.dataframe(pd.DataFrame([{
                    "1M": f"{row['1M Mom']:.1%}", "3M": f"{row['3M Mom']:.1%}",
                    "6M": f"{row['6M Mom']:.1%}", "12M": f"{row['12M Mom']:.1%}",
                    "MA50": row["Above MA50"], "MA200": row["Above MA200"]
                }]), use_container_width=True)

                st.markdown("**Risk detail**")
                st.dataframe(pd.DataFrame([{
                    "30D Vol": f"{row['30D Vol']:.1%}",
                    "Dist 52W Hi": f"{row['Dist 52W Hi']:.1%}",
                    "Max DD": f"{row['Max DD']:.1%}",
                }]), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# DISCOVER
# ══════════════════════════════════════════════════════════════════════════════
with tab_discover:
    st.markdown("### 🔭 Stock Discovery")
    st.caption("Scan a broad universe for growth setups or day-trading candidates.")

    scan_mode = st.radio("Scan type", ["📈 Growth", "⚡ Day Trading"], horizontal=True)
    scan_type = "day_trade" if "Day" in scan_mode else "growth"
    universe  = DAY_TRADE_UNIVERSE if scan_type == "day_trade" else DISCOVERY_UNIVERSE

    top_n     = st.slider("Show top N results", 10, len(universe), 20)

    if st.button("🔍 Run Scan", key="run_disc"):
        with st.spinner(f"Scanning {len(universe)} stocks..."):
            disc_df = run_discovery_scan(tuple(universe), scan_type=scan_type)
        st.session_state["discovery_results"] = (disc_df, scan_type)

    disc_state = st.session_state.get("discovery_results")
    if disc_state is not None:
        disc_df, last_scan = disc_state
        if disc_df.empty:
            st.warning("No results returned.")
        else:
            df_show = disc_df.head(top_n).copy()
            score_col = "Day Score" if last_scan == "day_trade" else "Growth Score"

            if last_scan == "day_trade":
                fmt = {
                    "Price": "{:.2f}", "Day Score": "{:.1f}",
                    "RVOL": "{:.2f}", "ATR %": "{:.2f}",
                    "1M Mom": "{:.1%}", "Vol 30D": "{:.1f}",
                }
                cols_show = ["Ticker","Name","Sector","Price","Day Score",
                             "RVOL","ATR %","1M Mom","Vol 30D","Above MA50"]
            else:
                fmt = {
                    "Price": "{:.2f}", "Growth Score": "{:.1f}",
                    "3M Mom": "{:.1%}", "6M Mom": "{:.1%}",
                    "EPS Growth": "{:.1%}", "Rev Growth": "{:.1%}",
                }
                cols_show = ["Ticker","Name","Sector","Price","Growth Score",
                             "3M Mom","6M Mom","EPS Growth","Rev Growth",
                             "Golden Cross","Near 52W Hi","P/E"]

            avail = [c for c in cols_show if c in df_show.columns]
            avail_fmt = {k: v for k, v in fmt.items() if k in avail}
            st.dataframe(
                df_show[avail].style
                    .format(avail_fmt)
                    .background_gradient(subset=[score_col], cmap="YlGn"),
                use_container_width=True, height=500
            )

            csv_disc = disc_df.to_csv(index=False)
            st.download_button("⬇️ Export CSV", csv_disc, "discovery.csv", "text/csv", key="dl_disc")

            # Quick add to watchlist
            st.markdown("---")
            add_t = st.selectbox("Add to watchlist", [""] + df_show["Ticker"].tolist(), key="disc_add")
            if st.button("Add", key="disc_add_btn") and add_t:
                st.session_state["watchlist"].add(add_t)
                st.success(f"Added {add_t} to watchlist!")
    else:
        st.info("Click **🔍 Run Scan** to start scanning.")

# ══════════════════════════════════════════════════════════════════════════════
# ETF HUB
# ══════════════════════════════════════════════════════════════════════════════
with tab_etf:
    st.markdown("### 🌐 ETF Hub")
    st.caption("Momentum + risk + expense-ratio scoring across 26 ETFs.")

    if st.button("🔄 Load ETF Data", key="run_etf"):
        with st.spinner("Fetching ETF data..."):
            etf_df = run_etf_scan()
        st.session_state["etf_results"] = etf_df

    etf_df = st.session_state.get("etf_results")
    if etf_df is not None and not etf_df.empty:
        cat_filter = st.multiselect(
            "Category",
            sorted(etf_df["Category"].unique().tolist()),
            default=[]
        )
        etf_show = etf_df if not cat_filter else etf_df[etf_df["Category"].isin(cat_filter)]

        disp_cols = ["Ticker","Name","Category","Price","ETF Score",
                     "1M","3M","6M","12M","Vol 30D","Dist 52W Hi",
                     "Max DD","Exp Ratio %","Trend"]
        avail = [c for c in disp_cols if c in etf_show.columns]
        fmt_etf = {
            "Price": "{:.2f}", "ETF Score": "{:.1f}",
            "1M": "{:.1%}", "3M": "{:.1%}", "6M": "{:.1%}", "12M": "{:.1%}",
            "Vol 30D": "{:.1f}", "Dist 52W Hi": "{:.1%}",
            "Max DD": "{:.1%}", "Exp Ratio %": "{:.2f}",
        }
        avail_fmt = {k: v for k, v in fmt_etf.items() if k in avail}
        st.dataframe(
            etf_show[avail].style
                .format(avail_fmt)
                .background_gradient(subset=["ETF Score"], cmap="RdYlGn", vmin=0, vmax=100),
            use_container_width=True, height=520
        )

        # Mini chart for selected ETF
        st.markdown("---")
        sel_etf = st.selectbox("View ETF chart", etf_show["Ticker"].tolist(), key="etf_sel")
        etf_row = etf_show[etf_show["Ticker"] == sel_etf].iloc[0]
        if "_close" in etf_row and etf_row["_close"] is not None:
            st.line_chart(etf_row["_close"].tail(252), height=250, use_container_width=True)

        csv_etf = etf_show.drop(columns=["_close"], errors="ignore").to_csv(index=False)
        st.download_button("⬇️ Export CSV", csv_etf, "etf_scores.csv", "text/csv", key="dl_etf")
    else:
        st.info("Click **🔄 Load ETF Data** to fetch rankings.")

# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════
with tab_portfolio:
    st.subheader("💼 Portfolio Overlay")
    st.caption("Track holdings against your analysis signals.")

    owned = st.session_state["owned"]
    pa, pb, pc, pd_ = st.columns(4)
    pt = pa.text_input("Ticker", key="pt")
    pq = pb.number_input("Quantity", min_value=0.0, step=1.0, key="pq")
    pp = pc.number_input("Avg buy price", min_value=0.0, step=0.01, key="pp")
    if pd_.button("Add holding", key="padd") and pt.strip():
        owned[pt.strip().upper()] = {"qty": pq, "avg_price": pp}

    if not owned:
        st.info("No holdings added yet.")
    else:
        df_all = st.session_state["results"]
        port_rows = []
        for t, h in owned.items():
            if df_all is not None:
                row_sig = df_all[df_all["Ticker"] == t]
            else:
                row_sig = pd.DataFrame()
            if not row_sig.empty:
                r = row_sig.iloc[0]
                curr  = r["Last Price"]
                val   = h["qty"] * curr
                cost  = h["qty"] * h["avg_price"]
                pnl   = val - cost
                pnlp  = (pnl / cost) if cost > 0 else 0
                port_rows.append({
                    "Ticker": t, "Qty": h["qty"],
                    "Avg Price": h["avg_price"], "Curr Price": curr,
                    "Value ($)": round(val, 2), "P&L ($)": round(pnl, 2),
                    "P&L %": pnlp,
                    "Signal": f"{SIGNAL_ICON.get(r['Signal'],'')} {r['Signal']}",
                    "Composite": r["Composite"], "ML Score": r["ML Score"],
                    "Risk Score": r["Risk Score"], "30D Vol": r["30D Vol"],
                })
            else:
                port_rows.append({
                    "Ticker": t, "Qty": h["qty"],
                    "Avg Price": h["avg_price"], "Curr Price": "N/A",
                    "Value ($)": "N/A", "P&L ($)": "N/A", "P&L %": "N/A",
                    "Signal": "Run analysis",
                    "Composite": "N/A", "ML Score": "N/A",
                    "Risk Score": "N/A", "30D Vol": "N/A",
                })
        port_df = pd.DataFrame(port_rows)
        num_fmts = {
            "Avg Price": "{:.2f}", "Curr Price": "{:.2f}",
            "Value ($)": "{:.2f}", "P&L ($)": "{:.2f}",
            "P&L %": "{:.1%}", "Composite": "{:.1f}",
            "ML Score": "{:.1f}", "Risk Score": "{:.1f}", "30D Vol": "{:.1%}"
        }
        safe_fmts = {
            k: v for k, v in num_fmts.items()
            if k in port_df.columns
            and port_df[k].apply(lambda x: isinstance(x, (int, float))).all()
        }
        st.dataframe(port_df.style.format(safe_fmts), use_container_width=True)

        # Concentration warnings
        num_port = port_df[port_df["Value ($)"].apply(lambda x: isinstance(x, (int, float)))].copy()
        if not num_port.empty:
            total_val = num_port["Value ($)"].sum()
            num_port["Weight"] = num_port["Value ($)"] / total_val
            st.markdown("**Concentration check**")
            for _, r in num_port[num_port["Weight"] > 0.15].iterrows():
                st.warning(f"⚠️ {r['Ticker']} is {r['Weight']:.0%} of portfolio (>15%)")
            hi_vol = num_port[
                num_port["30D Vol"].apply(lambda x: isinstance(x, float) and x > 0.5)
            ]
            if not hi_vol.empty:
                st.warning(f"⚠️ High-vol holdings: {', '.join(hi_vol['Ticker'].tolist())}")

            # Allocation bar
            st.markdown("**Allocation**")
            alloc = num_port.set_index("Ticker")["Weight"]
            st.bar_chart(alloc)

        to_rm = st.selectbox("Remove a holding", [""] + list(owned.keys()), key="prm")
        if st.button("Remove", key="prm_btn") and to_rm:
            owned.pop(to_rm, None)

# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.subheader("📈 Backtest")
    st.caption(
        "Equal-weight portfolio of current Buy/Strong Buy signals vs SPY. "
        "Based on 1-year historical returns — past performance ≠ future results."
    )

    if st.session_state["results"] is None:
        st.info("Run analysis first.")
    else:
        if st.button("▶️ Run Backtest"):
            with st.spinner("Running backtest..."):
                cum_port, cum_both, stats = run_backtest(
                    st.session_state["charts"],
                    st.session_state["results"]
                )
            if stats:
                s1, s2, s3, s4, s5, s6 = st.columns(6)
                s1.metric("Ann. Return",  f"{stats['ann_ret']:.1%}")
                s2.metric("Ann. Vol",     f"{stats['ann_vol']:.1%}")
                s3.metric("Sharpe",       f"{stats['sharpe']:.2f}")
                s4.metric("Sortino",      f"{stats['sortino']:.2f}")
                s5.metric("Max DD",       f"{stats['max_dd']:.1%}")
                s6.metric("Win Rate",     f"{stats['win_rate']:.1f}%")
                st.markdown("---")

            if cum_both is not None:
                st.line_chart(cum_both, height=380, use_container_width=True)
                final = cum_both.iloc[-1]
                b1, b2, b3 = st.columns(3)
                b1.metric("Portfolio total", f"{final['Portfolio']:.1%}")
                b2.metric("SPY total",        f"{final['SPY']:.1%}")
                b3.metric("Alpha",            f"{final['Portfolio'] - final['SPY']:.1%}")
            elif cum_port is not None:
                st.line_chart(cum_port, height=300, use_container_width=True)
            else:
                st.warning("Not enough Buy/Strong Buy signals to run a backtest.")

# ══════════════════════════════════════════════════════════════════════════════
# ML INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_ml:
    st.subheader("🤖 ElasticNet ML Insights")

    if ML_MODEL is None:
        st.info(
            "No trained model found. Run the trainer first:\n\n"
            "```bash\npython train_model.py\n```\n\n"
            "This downloads 3 years of price history, trains an ElasticNet "
            "logistic model, and saves artifacts to ./artifacts/."
        )
    else:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Train AUC", f"{ML_META['train_auc']:.3f}")
        mc2.metric("Test AUC",  f"{ML_META['test_auc']:.3f}")
        mc3.metric("Best C",    f"{ML_META['best_C']:.4f}")
        mc4.metric("l1 ratio",  f"{ML_META['best_l1_ratio']:.2f}")
        st.caption(
            f"Trained: {ML_META.get('trained_at', '?')[:19]}  |  "
            f"Features: {', '.join(ML_META['feature_cols'])}"
        )
        st.markdown("---")

        st.markdown("#### Feature Coefficients")
        st.caption(
            "Positive = associated with top-30% next-month returns. "
            "Negative = associated with underperformance."
        )
        coef_df = pd.DataFrame(
            [{"Feature": k, "Coefficient": v} for k, v in ML_META["coef"].items()]
        ).sort_values("Coefficient", key=abs, ascending=False).reset_index(drop=True)
        st.dataframe(
            coef_df.style
                .format({"Coefficient": "{:+.4f}"})
                .bar(subset="Coefficient", color=["#d65f5f", "#5fba7d"]),
            use_container_width=True
        )
        st.markdown("---")

        if st.session_state["results"] is not None:
            df_view = st.session_state["results"]
            df_view = df_view[df_view["Composite"] >= min_score]
            st.markdown("#### ML Probability per Stock")
            st.caption("P(top-30% next month) from ElasticNet model.")
            ml_disp = df_view[["Ticker", "Sector", "ML Prob", "ML Score", "Signal"]].copy()
            ml_disp = ml_disp.sort_values("ML Prob", ascending=False).reset_index(drop=True)
            ml_disp["Signal"] = ml_disp["Signal"].apply(lambda s: f"{SIGNAL_ICON.get(s, '')} {s}")
            st.dataframe(
                ml_disp.style.format({"ML Prob": "{:.1%}", "ML Score": "{:.1f}"}),
                use_container_width=True
            )
