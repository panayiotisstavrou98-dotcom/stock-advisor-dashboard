# 📈 Stock Advisor Dashboard

A Streamlit dashboard that computes Buy/Hold/Sell signals for stocks using:
- **6-month momentum**
- **Distance from 52-week high**
- **30-day realized volatility**

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Fork this repo
2. Go to https://streamlit.io/cloud
3. Connect your GitHub and select this repo
4. Set main file: `app.py`
5. Click Deploy

## Roadmap
- [ ] Add value & quality factors (P/E, ROE, debt)
- [ ] Add news/earnings sentiment
- [ ] ML-based composite score (XGBoost)
- [ ] Portfolio overlay
- [ ] Backtesting view
