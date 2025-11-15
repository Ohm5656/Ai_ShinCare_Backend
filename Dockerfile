# ============================================
# GlowbieBell – AI Skin Analyzer Backend
# Correct Railway Build (NO ZIP, DIRECT MODEL COPY)
# ============================================

FROM python:3.10-slim

# ---------------------
# 1) System dependencies
# ---------------------
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    g++ \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---------------------
# 2) Work directory
# ---------------------
WORKDIR /app

# ---------------------
# 3) Copy dependency list
# ---------------------
COPY requirements.txt .

# ---------------------
# 4) Install Python packages
# ---------------------
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------
# 5) Copy ALL source code (including models/)
# ---------------------
COPY . .

# ---------------------
# 6) Ensure models directory exists
# ---------------------
RUN mkdir -p /app/models

# ---------------------
# 7) Environment Variables
# ---------------------
ENV STORAGE_DIR=/data/uploads \
    MODEL_DIR=/app/models \
    INSIGHTFACE_PROVIDER=CPUExecutionProvider \
    DETECT_SIZE=640 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# ---------------------
# 8) Create upload directory (for railway volume)
# ---------------------
RUN mkdir -p /data/uploads

# ---------------------
# 9) Expose + Run
# ---------------------
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
