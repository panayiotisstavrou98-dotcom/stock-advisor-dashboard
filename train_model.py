"""
train_model.py — Build a regularised ElasticNet logistic model on historical
factor data and save artifacts to ./artifacts/

Run once before launching the dashboard:
    python train_model.py

What it does:
  1. Downloads 3 years of daily price history for a 50-stock universe.
  2. Computes 9 factor features per (date, ticker) snapshot.
  3. Labels each snapshot: 1 if next-month return >= top 30% of universe, else 0.
  4. Trains LogisticRegressionCV with elasticnet penalty (saga solver),
     cross-validating over regularisation strength C and l1_ratio.
  5. Saves model.pkl, scaler.pkl, meta.pkl to ./artifacts/.
"""
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# 50-stock training universe (S&P 500 large-caps across sectors)
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH",  "XOM",  "JNJ",  "WMT",   "PG",   "MA",   "HD",   "CVX",   "MRK", "ABBV",
    "PEP",  "KO",   "AVGO", "LLY",   "COST", "CSCO", "MCD",  "ACN",   "TMO", "ABT",
    "NKE",  "DIS",  "VZ",   "ADBE",  "CRM",  "INTC", "AMD",  "QCOM",  "TXN", "HON",
    "RTX",  "UPS",  "BA",   "CAT",   "GE",   "MMM",  "NEE",  "LOW",   "SBUX","AMAT",
]

TRAIN_YEARS  = 3     # years of history to download
FWD_DAYS     = 21    # forward window for target (~1 calendar month)
TOP_PCTILE   = 0.30  # top 30% of next-month returns = positive class

FEATURE_COLS = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m",
    "above_ma50", "above_ma200",
    "vol_30d", "max_drawdown", "dist_52w_high",
]


def download_prices(tickers, years):
    end   = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    print(f"Downloading {len(tickers)} tickers ({start.date()} to {end.date()}) ...")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    df = raw["Close"] if "Close" in raw.columns else raw
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df.dropna(axis=1, how="all", inplace=True)
    print(f"  Got {df.shape[1]} tickers, {df.shape[0]} trading days")
    return df


def compute_features_for_date(prices: pd.DataFrame, date_idx: int):
    """Compute factor features for all tickers at a single date index."""
    window = prices.iloc[max(0, date_idx - 252): date_idx + 1]
    if len(window) < 63:
        return None

    close = window.iloc[-1]

    # Momentum
    m1  = close / window.iloc[-22] - 1 if len(window) >= 22  else pd.Series(0.0, index=close.index)
    m3  = close / window.iloc[-63] - 1 if len(window) >= 63  else pd.Series(0.0, index=close.index)
    m6  = close / window.iloc[max(0, len(window) - 126)] - 1
    m12 = close / window.iloc[0] - 1

    # Trend flags
    ma50  = window.tail(50).mean()
    ma200 = window.tail(200).mean() if len(window) >= 200 else window.mean()
    above_ma50  = (close > ma50).astype(float)
    above_ma200 = (close > ma200).astype(float)

    # Risk
    daily_ret = window.pct_change().dropna()
    vol_30    = daily_ret.tail(30).std() * (252 ** 0.5)
    roll_max  = window.expanding().max()
    drawdown  = (window / roll_max - 1).min()
    dist_high = close / window.max() - 1

    return pd.DataFrame({
        "mom_1m":        m1,
        "mom_3m":        m3,
        "mom_6m":        m6,
        "mom_12m":       m12,
        "above_ma50":    above_ma50,
        "above_ma200":   above_ma200,
        "vol_30d":       vol_30,
        "max_drawdown":  drawdown,
        "dist_52w_high": dist_high,
    })


def build_dataset(prices: pd.DataFrame,
                  fwd_days: int = FWD_DAYS,
                  top_pctile: float = TOP_PCTILE,
                  sample_every: int = 5) -> pd.DataFrame:
    """Walk through time sampling every `sample_every` days."""
    records = []
    n = len(prices)
    for i in range(252, n - fwd_days, sample_every):
        feat_df = compute_features_for_date(prices, i)
        if feat_df is None or feat_df.isnull().all().all():
            continue
        fwd_ret   = prices.iloc[i + fwd_days] / prices.iloc[i] - 1
        threshold = fwd_ret.quantile(1 - top_pctile)
        label     = (fwd_ret >= threshold).astype(int)
        feat_df["target"] = label
        feat_df["date"]   = prices.index[i]
        records.append(feat_df)
    return pd.concat(records).reset_index().rename(columns={"index": "ticker"})


def train(df: pd.DataFrame):
    """Train ElasticNet logistic regression with time-based train/test split."""
    df = df.dropna(subset=FEATURE_COLS + ["target"])

    # Time-based split: first 80% of dates = train
    dates = sorted(df["date"].unique())
    split = dates[int(len(dates) * 0.80)]
    tr_df = df[df["date"] <  split]
    te_df = df[df["date"] >= split]

    X_tr, y_tr = tr_df[FEATURE_COLS].values, tr_df["target"].values
    X_te, y_te = te_df[FEATURE_COLS].values, te_df["target"].values
    print(f"  Train rows: {len(X_tr)}  |  Test rows: {len(X_te)}")

    # Standardise (fit on train only)
    scaler    = StandardScaler()
    X_tr_s    = scaler.fit_transform(X_tr)
    X_te_s    = scaler.transform(X_te)

    # ElasticNet logistic with cross-validated C and l1_ratio
    model = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 20),
        cv=5,
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
        scoring="roc_auc",
        max_iter=2000,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_tr_s, y_tr)

    tr_auc = roc_auc_score(y_tr, model.predict_proba(X_tr_s)[:, 1])
    te_auc = roc_auc_score(y_te, model.predict_proba(X_te_s)[:, 1])
    print(f"  Best C={model.C_[0]:.4f}  l1_ratio={model.l1_ratio_[0]:.2f}")
    print(f"  Train AUC={tr_auc:.3f}   Test AUC={te_auc:.3f}")

    meta = {
        "feature_cols":  FEATURE_COLS,
        "train_auc":     float(tr_auc),
        "test_auc":      float(te_auc),
        "best_C":        float(model.C_[0]),
        "best_l1_ratio": float(model.l1_ratio_[0]),
        "coef":          dict(zip(FEATURE_COLS, model.coef_[0].tolist())),
        "trained_at":    pd.Timestamp.now().isoformat(),
    }
    return scaler, model, meta


if __name__ == "__main__":
    print("=" * 60)
    print("Stock Intelligence — Regularised Model Trainer")
    print("=" * 60)

    prices  = download_prices(UNIVERSE, TRAIN_YEARS)

    print("Building feature/label dataset ...")
    dataset = build_dataset(prices)
    print(f"  Dataset rows: {len(dataset)}")

    print("Training ElasticNet logistic model ...")
    scaler, model, meta = train(dataset)

    # Persist artifacts
    for name, obj in [("model", model), ("scaler", scaler), ("meta", meta)]:
        path = os.path.join(ARTIFACTS_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Saved {path}")

    print("\nFeature importances (|coefficient|):")
    for feat, coef in sorted(meta["coef"].items(), key=lambda x: abs(x[1]), reverse=True):
        sign = "+" if coef >= 0 else "-"
        bar  = "█" * max(1, int(abs(coef) * 20))
        print(f"  {feat:<20} {sign}{abs(coef):.4f}  {bar}")

    print("\nDone. Launch dashboard:  streamlit run app.py")
