# 📈 Stock Advisor Dashboard

Full-featured stock analysis dashboard built with Streamlit + yfinance.

## Features

| Module | Details |
|---|---|
| 🚀 Momentum | 1M/6M/12M returns, MA50/MA200 trend |
| 💰 Value | P/E (sector-relative), P/S, EV/EBITDA, FCF Yield |
| ⭐ Quality | ROE, margin, D/E, current ratio, accrual ratio |
| 🛡️ Risk | 30D vol, 52W high distance, max drawdown |
| 📰 Sentiment | Analyst rec, EPS surprise, NewsAPI headlines |
| 🤖 ML scaffold | Logistic blend (XGBoost in next version) |
| 🚨 Hard risk caps | Vol/drawdown thresholds cap signal to Hold |
| ⭐ Watchlist | Track favourite stocks across sessions |
| 💼 Portfolio | Holdings tracker with P&L + concentration warnings |
| 🕰️ Backtest | Equal-weight Buy signals vs SPY benchmark |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## NewsAPI setup (optional)

1. Register free at https://newsapi.org
2. Paste your API key in the sidebar → News API field

## Deploy to Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Connect this GitHub repo
3. Main file: `app.py`
4. Click Deploy

## Roadmap
- [ ] Trained XGBoost model on historical factor scores
- [ ] Monthly auto-retraining scheduler
- [ ] Email/push alerts on signal changes
- [ ] Sector comparison charts
