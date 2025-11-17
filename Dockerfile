# Dockerfile — small, production-friendly
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true
ENV PORT=8501

CMD ["streamlit", "run", "streamlit_crypto_ui.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
