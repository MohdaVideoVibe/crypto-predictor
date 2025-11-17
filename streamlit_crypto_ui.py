import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

# Optional imports
try:
    from pycoingecko import CoinGeckoAPI
except Exception:
    CoinGeckoAPI = None

try:
    import ccxt
except Exception:
    ccxt = None

st.set_page_config(page_title="Crypto Predictor", layout="wide")

# ======= Helper indicator functions (no external 'ta' dependency) =======

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff(1)
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Use exponential moving average for smoothing (standard RSI)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral for initial values
    return rsi

def bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0):
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + (std * n_std)
    lower = sma - (std * n_std)
    return upper, lower, (upper - lower)

# ======= Data fetching =======

@st.cache_data
def fetch_coingecko(coin_id="bitcoin", vs_currency="usd", days=365):
    if CoinGeckoAPI is None:
        raise ImportError("pycoingecko not installed")
    cg = CoinGeckoAPI()
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)
    prices = data["prices"]
    vols = data.get("total_volumes", None)
    df = pd.DataFrame(prices, columns=["ts", "close"])
    df["volume"] = pd.DataFrame(vols)[1] if vols is not None else np.nan
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    df = df[["open", "high", "low", "close", "volume"]]
    return df

@st.cache_data
def fetch_ccxt_ohlcv(exchange_id="binance", symbol="BTC/USDT",
                     timeframe="1h", limit=1000, api_key=None, secret=None):
    if ccxt is None:
        raise ImportError("ccxt not installed")
    Exchange = getattr(ccxt, exchange_id)
    ex = Exchange({"enableRateLimit": True})
    if api_key and secret:
        ex.apiKey = api_key
        ex.secret = secret
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df

# ======= Indicators (pandas implementations) =======

@st.cache_data
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().astype(float)
    d["sma_7"] = d["close"].rolling(7).mean()
    d["sma_21"] = d["close"].rolling(21).mean()
    d["ema_12"] = d["close"].ewm(span=12, adjust=False).mean()
    d["ema_26"] = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = d["ema_12"] - d["ema_26"]
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["rsi_14"] = compute_rsi(d["close"], period=14)
    d["bb_h"], d["bb_l"], d["bb_width"] = bollinger_bands(d["close"], window=20, n_std=2.0)
    d["returns"] = d["close"].pct_change()
    d["vol_10"] = d["returns"].rolling(10).std()
    d["momentum_7"] = d["close"].pct_change(7)
    # drop rows where indicators are NaN
    return d.dropna()

# ======= Dataset creation =======

@st.cache_data
def create_tabular_dataset(df: pd.DataFrame):
    d = df.copy()
    lags = [1, 2, 3, 5, 7, 14]
    for lag in lags:
        d[f"close_lag_{lag}"] = d["close"].shift(lag)
        d[f"return_lag_{lag}"] = d["returns"].shift(lag)
    d = d.dropna()
    target = d["close"].pct_change(periods=1).shift(-1)
    d = d.dropna()
    X = d.copy()
    y = target.loc[X.index]
    return X, y

# ======= Backtest =======

def backtest(dates, close, preds, threshold=0.0):
    df = pd.DataFrame({"close": close}, index=dates)
    df["pred"] = preds
    df["pred_shift"] = df["pred"].shift(1)
    df["position"] = (df["pred_shift"] > threshold).astype(int)
    df["market_ret"] = df["close"].pct_change()
    df["strategy_ret"] = df["position"] * df["market_ret"]
    df["strategy_ret"].fillna(0, inplace=True)
    df["cum_strategy"] = (1 + df["strategy_ret"]).cumprod()
    df["cum_buyhold"] = (1 + df["market_ret"]).cumprod()
    return df

# ======= Streamlit UI =======

st.title("Crypto Predictor — Intraday + Daily")

col1, col2 = st.columns([1, 2])

with col1:
    data_source = st.radio("Data Source", ["CoinGecko (daily)", "CCXT (exchange intraday)"])
    if data_source == "CoinGecko (daily)":
        coin = st.selectbox("Coin ID", ["bitcoin", "ethereum", "binancecoin", "ripple", "cardano"])
        days = st.slider("Days of History", min_value=90, max_value=1200, value=400, step=10)
    else:
        exchange_id = st.selectbox("Exchange", ["binance", "kraken", "coinbasepro"])
        symbol = st.text_input("Symbol (example: BTC/USDT)", value="BTC/USDT")
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
        limit = st.slider("Bars to fetch", min_value=100, max_value=2000, value=1000, step=50)
        st.markdown("**API keys (optional)**: leave blank for public market data.")
        api_key = st.text_input("API Key (optional)", type="password")
        api_secret = st.text_input("API Secret (optional)", type="password")

    load_models = st.checkbox("Load existing XGBoost model", value=True)
    retrain = st.checkbox("Quick retrain XGBoost (light)", value=False)
    run_btn = st.button("Fetch data & analyze")

with col2:
    st.info("Supports CoinGecko daily + CCXT intraday. No TensorFlow included (XGBoost only).")

if run_btn:
    try:
        if data_source == "CoinGecko (daily)":
            if CoinGeckoAPI is None:
                st.error("pycoingecko not installed. Install with `pip install pycoingecko`")
                st.stop()
            df = fetch_coingecko(coin, "usd", days)
        else:
            if ccxt is None:
                st.error("ccxt not installed. Install with `pip install ccxt`")
                st.stop()
            df = fetch_ccxt_ohlcv(exchange_id=exchange_id, symbol=symbol, timeframe=timeframe, limit=limit, api_key=api_key or None, secret=api_secret or None)

        st.success(f"Fetched {len(df)} rows")
        df_ind = add_indicators(df)
        st.write(df_ind.tail())

        X, y = create_tabular_dataset(df_ind)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        default_feats = ["close", "volume", "sma_7", "sma_21", "rsi_14", "macd", "vol_10"]
        feature_cols = [c for c in default_feats if c in numeric_cols]
        X_sel = X[feature_cols]

        xgb_model = None
        if load_models and os.path.exists("models/xgb_model.joblib"):
            try:
                xgb_model = joblib.load("models/xgb_model.joblib")
                st.success("Loaded existing XGBoost model")
            except Exception:
                xgb_model = None

        if xgb_model is None or retrain:
            import xgboost as xgb
            st.info("Training lightweight XGBoost model (may take a moment)...")
            split = int(0.8 * len(X_sel))
            Xtr, Xte = X_sel.iloc[:split], X_sel.iloc[split:]
            ytr, yte = y.iloc[:split], y.iloc[split:]
            dtrain = xgb.DMatrix(Xtr, label=ytr)
            dtest = xgb.DMatrix(Xte, label=yte)
            params = {"objective": "reg:squarederror", "eval_metric": "rmse", "learning_rate": 0.05, "max_depth": 4}
            model = xgb.train(params, dtrain, 200, evals=[(dtest, "valid")], early_stopping_rounds=20)
            xgb_model = model
            os.makedirs("models", exist_ok=True)
            joblib.dump(xgb_model, "models/xgb_model.joblib")
            st.success("Model trained and saved")

        # Predict and backtest
        import xgboost as xgb
        preds = xgb_model.predict(xgb.DMatrix(X_sel))
        st.write("Predictions (last 5):")
        st.write(pd.Series(preds, index=X_sel.index).tail(5))

        bt = backtest(X_sel.index, X_sel["close"], preds)
        fig, ax = plt.subplots()
        ax.plot(bt["cum_strategy"], label="Strategy")
        ax.plot(bt["cum_buyhold"], label="Buy & Hold")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")

# ======= Deployment info =======

st.markdown("---")
st.header("Deployment & Public URL")

st.markdown("""
To get a **permanent public web link**, deploy this app to one of:

- **Streamlit Cloud (recommended)**
- Render
- Railway
- Your own VPS (Docker)

Files included:
- `requirements.txt`
- `Dockerfile`
- `README.md`
""")

st.caption("Not financial advice.")
