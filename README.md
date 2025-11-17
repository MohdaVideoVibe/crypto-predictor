# Crypto Predictor (Streamlit)

This repository contains a Streamlit web UI for crypto analysis and prediction. It supports CoinGecko (daily) and CCXT exchanges for intraday OHLCV.

## Quick start (local)

1. Create virtualenv and install deps:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Run the app:

   ```bash
   streamlit run streamlit_crypto_ui.py
   ```

3. Open http://localhost:8501

## Deploy to Streamlit Community Cloud (recommended)

1. Create a GitHub repo and push all files (including `streamlit_crypto_ui.py`, `requirements.txt`, and `README.md`).
2. Sign into https://streamlit.io/cloud and click **New app**.
3. Connect your GitHub repo, select branch and the file `streamlit_crypto_ui.py`, click **Deploy**.
4. In the Streamlit Cloud app settings add secrets via **Manage app secrets** (paste API keys). They appear in the app as `st.secrets`.

Streamlit Cloud will provide a stable URL like `https://share.streamlit.io/<user>/<repo>/<branch>/`.

## Deploy with Docker (Render / VPS / DigitalOcean)

1. Build the docker image locally:

   ```bash
   docker build -t crypto-predictor:latest .
   docker run -p 8501:8501 crypto-predictor:latest
   ```

2. Or push image to a registry and create a service on Render/Fly.io/your server.

## Security
- Do not commit API keys to the repo. Use Streamlit Secrets or environment variables. For Docker deployments, pass secrets using the host or the cloud provider's secret manager.

## Notes
- This is an experimental tool for learning. Not financial advice.
