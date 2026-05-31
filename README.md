# 📈 Stock Advisor Dashboard

A fully-featured Streamlit stock analysis dashboard scoring stocks across 5 factor modules:

| Module | Signals used |
|---|---|
| 🚀 Momentum | 1M / 6M / 12M returns, MA50 / MA200 |
| 💰 Value | P/E, P/S, EV/EBITDA, FCF Yield |
| ⭐ Quality | ROE, profit margin, debt/equity, current ratio |
| 🛡️ Risk | 30D volatility, distance from 52W high |
| 📰 Sentiment | Analyst recommendation mean, EPS surprise |

**Composite score** (0–100) is a user-adjustable weighted blend → Strong Buy / Buy / Hold / Reduce / Sell

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Connect this GitHub repo
3. Set main file to `app.py`
4. Click Deploy

## Roadmap

- [ ] ML ensemble layer (XGBoost on factor scores)
- [ ] Backtesting view
- [ ] Portfolio overlay & position tracker
- [ ] Email/push alerts when signal changes
