# Use official Python runtime
FROM python:3.11-slim

# Prevent Python from writing .pyc files and using buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (needed for numpy/pandas/matplotlib)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libatlas-base-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Streamlit configuration (avoid asking for email or opening browser)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Start Streamlit app on Render ($PORT is set automatically by Render)
CMD ["streamlit", "run", "streamlit_crypto_ui.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
