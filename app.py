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

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Cleaner font & spacing */
[data-testid="stAppViewContainer"] { background: #0e1117; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
/* Metric cards */
[data-testid="metric-container"] {
    background: #1c2333;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 12px 16px;
}
/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: #1c2333;
    border-radius: 8px 8px 0 0;
    padding: 6px 16px;
    font-size: 13px;
}
.stTabs [aria-selected="true"] { background: #2d3748 !important; }
/* Signal badge colours */
.sig-strong-buy { color: #00c853; font-weight: 700; }
.sig-buy        { color: #76ff03; font-weight: 700; }
.sig-hold       { color: #40c4ff; font-weight: 700; }
.sig-reduce     { color: #ff9100; font-weight: 700; }
.sig-sell       { color: #ff1744; font-weight: 700; }
/* Card container */
.card {
    background: #1c2333;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
/* Sidebar cleaner */
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
    ("discovery_results", None),
    ("etf_results",  None),
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
SIGNAL_COLOR = {"Strong Buy": "#00c853", "Buy": "#76ff03", "Hold": "#40c4ff",
                "Reduce": "#ff9100", "Sell": "#ff1744"}

def sig_badge(s):
    c = SIGNAL_COLOR.get(s, "#aaa")
    return f'<span style="background:{c}22;color:{c};border:1px solid {c}55;border-radius:6px;padding:2px 8px;font-size:12px;font-weight:700">{SIGNAL_ICON.get(s,"")} {s}</span>'

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

# ─── FACTOR ENGINES ──────────────────────────────────────────────────────────
def momentum_score(close):
    last = float(close.iloc[-1])
    m1   = float(close.iloc[-1]/close.iloc[-21]  - 1) if len(close)>=21  else 0.0
    m3   = float(close.iloc[-1]/close.iloc[-63]  - 1) if len(close)>=63  else 0.0
    m6   = float(close.iloc[-1]/close.iloc[-126] - 1) if len(close)>=126 else 0.0
    m12  = float(close.iloc[-1]/close.iloc[0]    - 1)
    ma50  = float(close.tail(50).mean())
    ma200 = float(close.tail(200).mean()) if len(close)>=200 else float(close.mean())
    ama50, ama200 = last > ma50, last > ma200
    trend = (10 if ama50 else 0) + (10 if ama200 else 0)
    base  = norm(m1*0.20 + m6*0.40 + m12*0.40, -0.50, 0.80)
    return min(base+trend, 100), {"m1":m1,"m3":m3,"m6":m6,"m12":m12,
                                   "above_ma50":ama50,"above_ma200":ama200,
                                   "ma50":ma50,"ma200":ma200}

def value_score(info, sector="default"):
    pe   = info.get("trailingPE") or info.get("forwardPE")
    ps   = info.get("priceToSalesTrailing12Months")
    ev_e = info.get("enterpriseToEbitda")
    mkt  = info.get("marketCap")
    fcf  = info.get("freeCashflow")
    fcfy = (fcf/mkt) if (fcf and mkt and mkt>0) else None
    spe  = SECTOR_MEDIAN_PE.get(sector, SECTOR_MEDIAN_PE["default"])
    pts  = []
    if pe and pe>0:   pts.append(max(min((2 - pe/spe)*50, 100), 0))
    if ps and ps>0:   pts.append(100-norm(ps, 0.5, 20))
    if ev_e and ev_e>0: pts.append(100-norm(ev_e, 3, 40))
    if fcfy:          pts.append(norm(fcfy, -0.05, 0.15))
    score = float(np.mean(pts)) if pts else 50
    return max(min(score,100),0), {"pe":pe,"ps":ps,"ev_ebitda":ev_e,"fcf_yield":fcfy,
                                    "sector":sector,"sector_median_pe":spe}

def quality_score(info):
    roe, margin = info.get("returnOnEquity"), info.get("profitMargins")
    de,  cr     = info.get("debtToEquity"),   info.get("currentRatio")
    ocf, ni     = info.get("operatingCashflow"), info.get("netIncomeToCommon")
    accrual = (ocf/abs(ni)) if (ocf and ni and ni!=0) else None
    pts = []
    if roe    is not None: pts.append(norm(roe,    -0.10, 0.40))
    if margin is not None: pts.append(norm(margin, -0.05, 0.35))
    if de     is not None: pts.append(100-norm(de, 0, 200))
    if cr     is not None: pts.append(norm(cr, 0.5, 3.0))
    if accrual is not None: pts.append(norm(accrual, 0.5, 2.5))
    score = float(np.mean(pts)) if pts else 50
    return max(min(score,100),0), {"roe":roe,"profit_margin":margin,
                                    "debt_equity":de,"current_ratio":cr,"accrual_ratio":accrual}

def risk_score_fn(close):
    ret     = close.pct_change().dropna()
    vol_30  = float(ret.tail(30).std() * (252**0.5))
    dist_hi = float(close.iloc[-1]/close.max() - 1)
    max_dd  = float(((close - close.cummax())/close.cummax()).min())
    score   = (100-norm(vol_30,0.10,0.80))*0.4 + norm(dist_hi,-0.50,0.0)*0.3 + (100-norm(abs(max_dd),0,0.60))*0.3
    return max(min(score,100),0), {"vol_30d":vol_30,"dist_52w_high":dist_hi,"max_dd":max_dd}

def sentiment_score_fn(info, ticker, news_key):
    rec, eps_s = info.get("recommendationMean"), info.get("earningsQuarterlyGrowth")
    news_sent, headlines = 50, []
    if news_key:
        try:
            url = (f"https://newsapi.org/v2/everything?q={ticker}&language=en"
                   f"&sortBy=publishedAt&from={(datetime.now()-timedelta(days=7)).strftime('%Y-%m-%d')}"
                   f"&pageSize=10&apiKey={news_key}")
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                articles  = r.json().get("articles", [])
                headlines = [a["title"] for a in articles if a.get("title")]
                pos = sum(1 for h in headlines for w in ["beat","surge","strong","record","growth","upgraded","outperform","rally","profit"] if w in h.lower())
                neg = sum(1 for h in headlines for w in ["miss","fall","weak","loss","downgrade","underperform","crash","cut","decline"] if w in h.lower())
                if (pos+neg) > 0: news_sent = (pos/(pos+neg))*100
        except Exception: pass
    pts = []
    if rec:   pts.append(100 - norm(rec, 1, 5))
    if eps_s: pts.append(norm(eps_s, -0.30, 0.50))
    pts.append(news_sent)
    return max(min(float(np.mean(pts)) if pts else 50, 100), 0), \
           {"recommendation":rec,"earnings_surprise":eps_s,
            "news_sentiment":news_sent,"headlines":headlines}

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
def analyse_ticker(ticker, period, w_mom, w_val, w_qual, w_risk, news_key, max_vol_cap, max_dd_cap):
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, auto_adjust=True)
        if df.empty or len(df) < 60: return None, None
        close  = safe_series(df, "Close")
        info   = tk.info or {}
        sector = info.get("sector", "default")
        name   = info.get("shortName", ticker)

        ms, mm = momentum_score(close)
        vs, vm = value_score(info, sector)
        qs, qm = quality_score(info)
        rs, rm = risk_score_fn(close)
        ss, sm = sentiment_score_fn(info, ticker, news_key)

        ml_prob      = get_ml_prob(mm["m1"],mm["m3"],mm["m6"],mm["m12"],
                                   mm["above_ma50"],mm["above_ma200"],
                                   rm["vol_30d"],rm["max_dd"],rm["dist_52w_high"])
        ml_score_100 = round(ml_prob * 100, 1)

        total_w    = w_mom + w_val + w_qual + w_risk or 100
        rules_comp = (ms*w_mom + vs*w_val + qs*w_qual + rs*w_risk) / total_w
        comp       = round(min(max(rules_comp*0.70 + ml_score_100*0.30, 0), 100), 1)

        sig = signal_from_score(comp)
        sig, comp, cap = apply_hard_caps(sig, comp, rm["vol_30d"], rm["dist_52w_high"],
                                         max_vol_cap, max_dd_cap)
        mktcap = info.get("marketCap", 0) or 0
        return {
            "Ticker":       ticker, "Name": name, "Sector": sector,
            "Market Cap":   mktcap,
            "Last Price":   round(float(close.iloc[-1]), 2),
            "Mom Score":    round(ms, 1), "Value Score": round(vs, 1),
            "Qual Score":   round(qs, 1), "Risk Score":  round(rs, 1),
            "Sentiment":    round(ss, 1), "ML Prob":     round(ml_prob, 3),
            "ML Score":     ml_score_100, "Composite":   comp,
            "Signal":       sig, "Cap Reason": cap,
            "1M Mom":       mm["m1"], "3M Mom": mm["m3"],
            "6M Mom":       mm["m6"], "12M Mom": mm["m12"],
            "Above MA50":   "✅" if mm["above_ma50"]  else "❌",
            "Above MA200":  "✅" if mm["above_ma200"] else "❌",
            "P/E":          vm.get("pe"),   "P/S":      vm.get("ps"),
            "EV/EBITDA":    vm.get("ev_ebitda"), "FCF Yield": vm.get("fcf_yield"),
            "Sector PE":    vm.get("sector_median_pe"),
            "ROE":          qm.get("roe"),  "Margin":   qm.get("profit_margin"),
            "D/E":          qm.get("debt_equity"), "Curr Ratio": qm.get("current_ratio"),
            "Accrual":      qm.get("accrual_ratio"),
            "30D Vol":      rm["vol_30d"], "Dist 52W Hi": rm["dist_52w_high"],
            "Max DD":       rm["max_dd"],
            "Analyst Rec":  sm.get("recommendation"), "EPS Surprise": sm.get("earnings_surprise"),
            "News Sent":    sm.get("news_sentiment"),  "Headlines":    sm.get("headlines", []),
            "_close":       close,
        }, df
    except Exception:
        return None, None

# ─── DISCOVERY ENGINE ─────────────────────────────────────────────────────────
# Universe: 120 liquid stocks across all sectors (small/mid/large cap)
DISCOVERY_UNIVERSE = [
    # Mega/Large Tech
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AMD","AVGO","QCOM",
    "INTC","TXN","MU","AMAT","KLAC","LRCX","CRWD","PANW","SNOW","PLTR",
    # Healthcare
    "LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","ISRG","DXCM","MRNA",
    # Financials
    "JPM","BAC","GS","MS","V","MA","AXP","BLK","SCHW","COF",
    # Consumer
    "AMZN","COST","WMT","MCD","SBUX","NKE","TGT","HD","LOW","ABNB",
    # Energy
    "XOM","CVX","COP","SLB","EOG","PXD","MPC","VLO",
    # Industrials
    "CAT","DE","HON","RTX","BA","GE","UPS","FDX","ETN","ROK",
    # Comm & Media
    "NFLX","DIS","SPOT","SNAP","PINS","TTD",
    # Small/Mid growth
    "CELH","HIMS","RXRX","IONQ","RKLB","JOBY","AEHR","UPST","SQ","HOOD",
    "COIN","MSTR","SMCI","APP","DUOL","SOUN","BBAI","ARRY","ENPH","FSLR",
    # International ADRs
    "ASML","TSM","BABA","JD","NVO","SAP","SE","GRAB","NU","MELI",
]
DISCOVERY_UNIVERSE = list(dict.fromkeys(DISCOVERY_UNIVERSE))  # deduplicate

# Day-trading candidates: high-vol liquid names
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

            price  = float(close.iloc[-1])
            m1     = float(close.iloc[-1]/close.iloc[-21]  - 1) if len(close)>=21  else 0
            m3     = float(close.iloc[-1]/close.iloc[-63]  - 1) if len(close)>=63  else 0
            m6     = float(close.iloc[-1]/close.iloc[0]    - 1)
            ma50   = float(close.tail(50).mean())
            ma200  = float(close.tail(200).mean()) if len(close)>=200 else float(close.mean())
            ret    = close.pct_change().dropna()
            vol30  = float(ret.tail(30).std() * (252**0.5))
            dist_hi= float(close.iloc[-1]/close.max() - 1)
            max_dd = float(((close - close.cummax())/close.cummax()).min())

            # Volume stats
            avg_vol   = float(vol.tail(20).mean()) if len(vol)>=20 else float(vol.mean())
            last_vol  = float(vol.iloc[-1])
            rvol      = last_vol / avg_vol if avg_vol > 0 else 1.0

            # ATR (14-day)
            hi = safe_series(df, "High").tail(15)
            lo = safe_series(df, "Low").tail(15)
            atr = float((hi - lo).tail(14).mean())
            atr_pct = atr / price if price > 0 else 0

            mktcap = info.get("marketCap", 0) or 0
            sector = info.get("sector", "Unknown")
            name   = info.get("shortName", ticker)

            if scan_type == "day_trade":
                # Day trade score: wants high RVOL, high ATR%, strong recent momentum
                score  = norm(rvol, 0.5, 5.0)*0.35 + norm(atr_pct, 0.01, 0.08)*0.30 \
                       + norm(m1, -0.15, 0.20)*0.20 + norm(vol30, 0.20, 1.20)*0.15
                results.append({
                    "Ticker": ticker, "Name": name, "Sector": sector,
                    "Price": round(price, 2), "Day Score": round(score, 1),
                    "RVOL": round(rvol, 2), "ATR %": round(atr_pct*100, 2),
                    "1M Mom": m1, "Vol 30D": round(vol30*100, 1),
                    "Above MA50": "✅" if price>ma50 else "❌",
                    "Market Cap": mktcap,
                })
            else:  # growth
                # Golden cross + breakout + earnings momentum
                golden = price > ma50 > ma200
                breakout_52 = dist_hi > -0.05  # within 5% of 52W high
                eps_g  = info.get("earningsQuarterlyGrowth") or 0
                rev_g  = info.get("revenueGrowth") or 0
                pe     = info.get("trailingPE") or info.get("forwardPE") or 0
                score  = norm(m3, -0.20, 0.60)*0.25 + norm(m6, -0.30, 0.80)*0.25 \
                       + norm(eps_g, -0.10, 0.50)*0.20 + norm(rev_g, -0.05, 0.40)*0.15 \
                       + (10 if golden else 0) + (5 if breakout_52 else 0)
                score  = min(score, 100)
                results.append({
                    "Ticker": ticker, "Name": name, "Sector": sector,
                    "Price": round(price, 2), "Growth Score": round(score, 1),
                    "3M Mom": m3, "6M Mom": m6,
                    "EPS Growth": eps_g, "Rev Growth": rev_g,
                    "Golden Cross": "✅" if golden else "❌",
                    "Near 52W Hi": "✅" if breakout_52 else "❌",
                    "P/E": pe if pe and pe > 0 else None,
                    "Market Cap": mktcap, "Sector": sector,
                })
        except Exception:
            continue
    if not results: return pd.DataFrame()
    df_out = pd.DataFrame(results)
    score_col = "Day Score" if scan_type == "day_trade" else "Growth Score"
    return df_out.sort_values(score_col, ascending=False).reset_index(drop=True)

# ─── ETF ENGINE ───────────────────────────────────────────────────────────────
ETF_UNIVERSE = {
    # Broad market
    "SPY":  {"name": "S&P 500",         "category": "Broad",    "exp": 0.0945},
    "QQQ":  {"name": "Nasdaq 100",       "category": "Broad",    "exp": 0.20},
    "IWM":  {"name": "Russell 2000",     "category": "Broad",    "exp": 0.19},
    "VTI":  {"name": "Total US Market",  "category": "Broad",    "exp": 0.03},
    "VT":   {"name": "Total World",      "category": "Broad",    "exp": 0.07},
    # Sector
    "XLK":  {"name": "Tech",             "category": "Sector",   "exp": 0.10},
    "XLF":  {"name": "Financials",       "category": "Sector",   "exp": 0.10},
    "XLE":  {"name": "Energy",           "category": "Sector",   "exp": 0.10},
    "XLV":  {"name": "Healthcare",       "category": "Sector",   "exp": 0.10},
    "XLI":  {"name": "Industrials",      "category": "Sector",   "exp": 0.10},
    "XLC":  {"name": "Comm Services",    "category": "Sector",   "exp": 0.10},
    "XLY":  {"name": "Consumer Discr.",  "category": "Sector",   "exp": 0.10},
    "XLRE": {"name": "Real Estate",      "category": "Sector",   "exp": 0.10},
    # Thematic
    "SOXX": {"name": "Semiconductors",   "category": "Thematic", "exp": 0.35},
    "ARKK": {"name": "ARK Innovation",   "category": "Thematic", "exp": 0.75},
    "ARKG": {"name": "ARK Genomic",      "category": "Thematic", "exp": 0.75},
    "AIQ":  {"name": "AI & Big Data",    "category": "Thematic", "exp": 0.68},
    "BOTZ": {"name": "Robotics & AI",    "category": "Thematic", "exp": 0.69},
    "CIBR": {"name": "Cybersecurity",    "category": "Thematic", "exp": 0.60},
    "ICLN": {"name": "Clean Energy",     "category": "Thematic", "exp": 0.41},
    # Bonds / Defensive
    "TLT":  {"name": "20Y Treasury",     "category": "Bonds",    "exp": 0.15},
    "HYG":  {"name": "High Yield Bond",  "category": "Bonds",    "exp": 0.48},
    "AGG":  {"name": "US Agg Bond",      "category": "Bonds",    "exp": 0.03},
    # Commodities
    "GLD":  {"name": "Gold",             "category": "Commodity","exp": 0.40},
    "SLV":  {"name": "Silver",           "category": "Commodity","exp": 0.50},
    "USO":  {"name": "Oil",              "category": "Commodity","exp": 0.81},
}

@st.cache_data(ttl=1800)
def run_etf_scan():
    rows = []
    tickers = list(ETF_UNIVERSE.keys())
    for ticker in tickers:
        meta = ETF_UNIVERSE[ticker]
        try:
            tk    = yf.Ticker(ticker)
            df    = tk.history(period="1y", auto_adjust=True)
            if df.empty or len(df) < 30: continue
            close = safe_series(df, "Close")
            vol_s = safe_series(df, "Volume")
            price = float(close.iloc[-1])
            m1    = float(close.iloc[-1]/close.iloc[-21]  - 1) if len(close)>=21  else 0
            m3    = float(close.iloc[-1]/close.iloc[-63]  - 1) if len(close)>=63  else 0
            m6    = float(close.iloc[-1]/close.iloc[-126] - 1) if len(close)>=126 else 0
            m12   = float(close.iloc[-1]/close.iloc[0]    - 1)
            ma50  = float(close.tail(50).mean())
            ma200 = float(close.tail(200).mean()) if len(close)>=200 else float(close.mean())
            ret   = close.pct_change().dropna()
            vol30 = float(ret.tail(30).std()*(252**0.5))
            dist  = float(close.iloc[-1]/close.max() - 1)
            max_dd= float(((close-close.cummax())/close.cummax()).min())
            avg_vol  = float(vol_s.tail(20).mean())
            rvol     = float(vol_s.iloc[-1]/avg_vol) if avg_vol>0 else 1.0
            # ETF momentum score (no value/quality — expense ratio penalty instead)
            mom_s   = norm(m1*0.20+m3*0.30+m6*0.30+m12*0.20, -0.30, 0.60)
            risk_s  = (100-norm(vol30,0.05,0.50))*0.5 + norm(dist,-0.40,0.0)*0.5
            exp_pen = max(0, 100 - meta["exp"]*100)  # lower expense = better
            etf_score = round(mom_s*0.55 + risk_s*0.35 + exp_pen*0.10, 1)
            rows.append({
                "Ticker": ticker, "Name": meta["name"], "Category": meta["category"],
                "Price": round(price, 2), "ETF Score": etf_score,
                "1M": m1, "3M": m3, "6M": m6, "12M": m12,
                "Vol 30D": round(vol30*100, 1), "Max DD": max_dd,
                "Dist 52W Hi": dist, "RVOL": round(rvol, 2),
                "Exp Ratio %": meta["exp"], "Trend": "✅" if price>ma50>ma200 else ("➡️" if price>ma50 else "❌"),
                "_close": close,
            })
        except Exception:
            continue
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ETF Score", ascending=False).reset_index(drop=True)

# ─── BACKTEST ─────────────────────────────────────────────────────────────────
def run_backtest(charts, df_results):
    buy_tickers = df_results[df_results["Signal"].isin(["Strong Buy","Buy"])]["Ticker"].tolist()
    if not buy_tickers: return None, None, None
    returns = {}
    for t in buy_tickers:
        df = charts.get(t)
        if df is None: continue
        s = safe_series(df, "Close").pct_change().dropna()
        returns[t] = s
    if not returns: return None, None, None
    port     = pd.DataFrame(returns).dropna().mean(axis=1)
    cum_port = (1+port).cumprod() - 1
    try:
        spy       = yf.download("SPY", period="1y", auto_adjust=True, progress=False)
        spy_ret   = safe_series(spy, "Close").pct_change().dropna()
        spy_ret.index = spy_ret.index.tz_localize(None) if spy_ret.index.tzinfo else spy_ret.index
        port.index    = port.index.tz_localize(None)    if port.index.tzinfo    else port.index
        combined  = pd.DataFrame({"Portfolio": port, "SPY": spy_ret}).dropna()
        cum_both  = (1+combined).cumprod() - 1
    except Exception:
        cum_both = None
    # Stats
    ann_ret  = float((1+port.mean())**252 - 1)
    ann_vol  = float(port.std() * (252**0.5))
    sharpe   = round(ann_ret / ann_vol, 2) if ann_vol > 0 else 0
    neg      = port[port < 0]
    sortino  = round(ann_ret / (neg.std()*(252**0.5)), 2) if len(neg)>0 else 0
    mdd      = float(((cum_port+1 - (cum_port+1).cummax())/(cum_port+1).cummax()).min())
    win_rate = round((port > 0).mean() * 100, 1)
    stats    = {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
                "sortino": sortino, "max_dd": mdd, "win_rate": win_rate}
    return cum_port, cum_both, stats

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Stock Intelligence")
    st.markdown("---")

    st.markdown("### 🎯 Watchlist Tickers")
    tickers_str = st.text_area("Tickers (one per line or comma-sep)",
                               value="AAPL\nMSFT\nNVDA\nGOOGL\nASML\nTSLA\nAMZN\nMETA",
                               height=130)
    period    = st.selectbox("Lookback period", ["6mo", "1y", "2y"], index=1)
    min_score = st.slider("Min composite score", 0, 100, 0, 5)

    with st.expander("⚖️ Factor weights", expanded=False):
        w_mom  = st.slider("Momentum",  0, 100, 30, 5)
        w_val  = st.slider("Value",     0, 100, 25, 5)
        w_qual = st.slider("Quality",   0, 100, 25, 5)
        w_risk = st.slider("Risk",      0, 100, 20, 5)
        total_w_ui = w_mom+w_val+w_qual+w_risk
        if total_w_ui != 100:
            st.warning(f"Total: {total_w_ui} (should be 100)")
        else:
            st.success(f"Total: {total_w_ui} ✓")

    with st.expander("🚨 Risk caps", expanded=False):
        max_vol = st.slider("Max annualised 30D vol", 0.20, 1.50, 0.80, 0.05)
        max_dd  = st.slider("Max drawdown from 52W hi", -0.60, -0.05, -0.25, 0.05)

    with st.expander("📰 News & ML", expanded=False):
        news_key = st.text_input("NewsAPI key (optional)", type="password")
        if ML_MODEL is not None:
            st.success(f"🤖 ElasticNet active\nAUC: {ML_META.get('test_auc',0):.3f}")
        else:
            st.warning("⚠️ Run `python train_model.py`")

    st.markdown("---")
    run_btn = st.button("🔄 Run Analysis", use_container_width=True, type="primary")
    st.caption("Educational tool only — not investment advice.")

# ─── PARSE TICKERS ────────────────────────────────────────────────────────────
tickers = []
for t in tickers_str.replace(",", "\n").split("\n"):
    t = t.strip().upper()
    if t: tickers.append(t)

# ─── RUN ANALYSIS ─────────────────────────────────────────────────────────────
if run_btn:
    # Save previous signals for change detection
    if st.session_state["results"] is not None:
        prev = st.session_state["results"]
        st.session_state["prev_signals"] = dict(zip(prev["Ticker"], prev["Signal"]))

    rows, charts = [], {}
    prog = st.progress(0, text="Analysing...")
    for i, t in enumerate(tickers):
        prog.progress((i+1)/len(tickers), text=f"Analysing {t} ({i+1}/{len(tickers)})...")
        row, df = analyse_ticker(t, period, w_mom, w_val, w_qual, w_risk, news_key, max_vol, max_dd)
        if row:
            rows.append(row)
            charts[t] = df
    prog.empty()
    if rows:
        df_res = pd.DataFrame(rows).sort_values("Composite", ascending=False).reset_index(drop=True)
        st.session_state["results"] = df_res
        st.session_state["charts"]  = charts
    else:
        st.error("No data returned. Check your tickers.")
        st.stop()

# ─── MAIN TABS ────────────────────────────────────────────────────────────────
tab_home, tab_signals, tab_discover, tab_etf, tab_portfolio, tab_backtest, tab_ml = st.tabs([
    "🏠 Home",
    "📊 Signals",
    "🔭 Discover",
    "🌐 ETF Hub",
    "💼 Portfolio",
    "📈 Backtest",
    "🤖 ML Insights",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HOME DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    st.markdown("### 👋 Welcome to Stock Intelligence")
    st.caption(f"Last updated: {datetime.now().strftime('%d %b %Y, %H:%M')}  •  Educational tool only")

    if st.session_state["results"] is None:
        st.info("👈  Add tickers in the sidebar and click **🔄 Run Analysis** to get started.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""<div class='card'>
            <h4>📊 Multi-Factor Signals</h4>
            <p>Momentum, Value, Quality, Risk & Sentiment scored and blended with ElasticNet ML</p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class='card'>
            <h4>🔭 Stock Discovery</h4>
            <p>Scan 120+ stocks for growth setups and day-trading opportunities automatically</p>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class='card'>
            <h4>🌐 ETF Hub</h4>
            <p>Sector rotation, ETF momentum rankings, and expense-ratio-adjusted scoring</p>
            </div>""", unsafe_allow_html=True)
    else:
        df_all = st.session_state["results"]
        prev   = st.session_state["prev_signals"]

        # ── Summary metrics
        sb = int((df_all["Signal"]=="Strong Buy").sum())
        b  = int((df_all["Signal"]=="Buy").sum())
        h  = int((df_all["Signal"]=="Hold").sum())
        r  = int((df_all["Signal"]=="Reduce").sum())
        s  = int((df_all["Signal"]=="Sell").sum())
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("🟢 Strong Buy", sb)
        m2.metric("🟡 Buy",        b)
        m3.metric("🔵 Hold",       h)
        m4.metric("🟠 Reduce",     r)
        m5.metric("🔴 Sell",       s)
        m6.metric("Avg Score",     f"{df_all['Composite'].mean():.1f}")
        st.markdown("---")

        # ── Signal changes
        changes = []
        for _, row in df_all.iterrows():
            old = prev.get(row["Ticker"])
            if old and old != row["Signal"]:
                changes.append((row["Ticker"], old, row["Signal"]))
        if changes:
            st.markdown("#### 🔔 Signal Changes Since Last Run")
            for t, old, new in changes:
                arrow = "⬆️" if ["Sell","Reduce","Hold","Buy","Strong Buy"].index(new) > ["Sell","Reduce","Hold","Buy","Strong Buy"].index(old) else "⬇️"
                st.markdown(f"**{t}** {arrow} {SIGNAL_ICON.get(old,'')} {old} → {SIGNAL_ICON.get(new,'')} {new}")
            st.markdown("---")

        # ── Top picks cards
        st.markdown("#### 🏆 Top Picks")
        top = df_all[df_all["Signal"].isin(["Strong Buy","Buy"])].head(6)
        if top.empty:
            st.info("No Buy or Strong Buy signals in current analysis.")
        else:
            cols = st.columns(min(len(top), 3))
            for idx, (_, row) in enumerate(top.iterrows()):
                with cols[idx % 3]:
                    sig_c = SIGNAL_COLOR.get(row["Signal"], "#aaa")
                    pct   = f"{row['1M Mom']:+.1%}" if isinstance(row['1M Mom'], float) else "N/A"
                    st.markdown(f"""
<div style='background:#1c2333;border:1px solid {sig_c}44;border-left:4px solid {sig_c};
border-radius:10px;padding:14px;margin-bottom:10px'>
<div style='font-size:18px;font-weight:700;color:{sig_c}'>{row['Ticker']}</div>
<div style='font-size:11px;color:#8892a4;margin-bottom:6px'>{row.get('Name','')}</div>
<div style='display:flex;justify-content:space-between'>
  <span style='font-size:20px;font-weight:700'>${row['Last Price']:.2f}</span>
  <span style='font-size:13px;color:{sig_c}'>{SIGNAL_ICON.get(row['Signal'],'')} {row['Signal']}</span>
</div>
<div style='display:flex;gap:12px;margin-top:8px;font-size:12px;color:#8892a4'>
  <span>Score: <b style='color:#eee'>{row['Composite']}</b></span>
  <span>1M: <b style='color:{"#00c853" if row["1M Mom"]>0 else "#ff1744"}'>{pct}</b></span>
  <span>Vol: <b style='color:#eee'>{row['30D Vol']:.0%}</b></span>
</div>
</div>""", unsafe_allow_html=True)

        # ── Watchlist quick view
        wl = st.session_state["watchlist"]
        if wl:
            st.markdown("---")
            st.markdown("#### ⭐ Watchlist")
            wl_df = df_all[df_all["Ticker"].isin(wl)]
            if not wl_df.empty:
                for _, row in wl_df.iterrows():
                    sig_c = SIGNAL_COLOR.get(row["Signal"], "#aaa")
                    st.markdown(f"**{row['Ticker']}** `${row['Last Price']:.2f}` &nbsp; {sig_badge(row['Signal'])} &nbsp; Score: **{row['Composite']}**", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SIGNALS
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
            # Filter controls
            fc1, fc2 = st.columns(2)
            sig_filter = fc1.multiselect("Filter by signal",
                ["Strong Buy","Buy","Hold","Reduce","Sell"],
                default=["Strong Buy","Buy","Hold","Reduce","Sell"])
            sec_filter = fc2.multiselect("Filter by sector",
                sorted(df_view["Sector"].unique().tolist()), default=[])
            df_filt = df_view[df_view["Signal"].isin(sig_filter)]
            if sec_filter:
                df_filt = df_filt[df_filt["Sector"].isin(sec_filter)]

            # CSV export
            csv = df_filt.drop(columns=["Headlines","_close"], errors="ignore").to_csv(index=False)
            st.download_button("⬇️ Export CSV", csv, "signals.csv", "text/csv", key="dl_sig")

            disp = df_filt[[
                "Ticker","Name","Sector","Last Price",
                "Mom Score","Value Score","Qual Score","Risk Score",
                "Sentiment","ML Score","Composite","Signal","Cap Reason"
            ]].copy()
            disp["Signal"] = disp["Signal"].apply(lambda s: f"{SIGNAL_ICON.get(s,'')} {s}")
            st.dataframe(
                disp.style.format({
                    "Last Price":"{:.2f}","Mom Score":"{:.1f}","Value Score":"{:.1f}",
                    "Qual Score":"{:.1f}","Risk Score":"{:.1f}","Sentiment":"{:.1f}",
                    "ML Score":"{:.1f}","Composite":"{:.1f}",
                }).background_gradient(subset=["Composite"], cmap="RdYlGn", vmin=0, vmax=100),
                use_container_width=True, height=480
            )

        with right:
            st.markdown("#### 🔍 Deep Dive")
            tlist = df_filt["Ticker"].tolist()
            if not tlist:
                st.info("No tickers match current filters.")
            else:
                sel  = st.selectbox("Select ticker", tlist, key="sel_dd")
                row  = df_filt[df_filt["Ticker"] == sel].iloc[0]
                wl   = st.session_state["watchlist"]
                prev = st.session_state["prev_signals"]

                sig_c = SIGNAL_COLOR.get(row["Signal"], "#aaa")
                old_sig = prev.get(sel)
                change_str = ""
                if old_sig and old_sig != row["Signal"]:
                    change_str = f" _(was {old_sig})_"

                st.markdown(f"""
<div style='background:#1c2333;border:1px solid {sig_c}44;border-left:4px solid {sig_c};
border-radius:10px;padding:16px'>
<div style='font-size:22px;font-weight:700'>{sel} <span style='color:{sig_c}'>{SIGNAL_ICON.get(row['Signal'],'')} {row['Signal']}</span></div>
<div style='color:#8892a4;font-size:12px'>{row.get('Name','')} · {row['Sector']}</div>
</div>""", unsafe_allow_html=True)

                if row["Cap Reason"]: st.warning(row["Cap Reason"])
                if change_str: st.info(f"🔔 Signal changed{change_str}")

                # Factor radar
                fa, fb = st.columns(2)
                fa.metric("🚀 Momentum",   f"{row['Mom Score