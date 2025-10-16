# ============================================
# AI Skin Analyzer Backend - Railway Build
# ============================================

FROM python:3.10-slim

# ---------------------
# 1) Install system dependencies
# ---------------------
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------
# 2) Set work directory
# ---------------------
WORKDIR /app

# ---------------------
# 3) Copy dependency list
# ---------------------
COPY requirements.txt .

# ---------------------
# 4) Upgrade pip + install Python packages
# ---------------------
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------
# 5) Copy application source code
# ---------------------
COPY . .

# ---------------------
# 6) Environment Variables
# ---------------------
ENV STORAGE_DIR=/data/uploads \
    INSIGHTFACE_PROVIDER=CPUExecutionProvider \
    DETECT_SIZE=640 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# ---------------------
# 7) Create directories for persistent storage
# ---------------------
RUN mkdir -p /data/uploads

# ---------------------
# 8) Expose API port and run server
# ---------------------
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

