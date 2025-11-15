# ============================================
# GlowbieBell – AI Skin Analyzer Backend
# Railway Build with Model ZIP Extraction
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
    unzip \
    build-essential \
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
# 5) Copy application source code
# ---------------------
COPY . .

# ---------------------
# 6) Extract model ZIP files from /models_zip
# ---------------------

RUN mkdir -p /app/models

RUN for f in /app/models/*.zip; do \
        echo "Extracting $f ..."; \
        unzip -o "$f" -d /app/models/; \
    done

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
# 8) Create upload directory
# ---------------------
RUN mkdir -p /data/uploads

# ---------------------
# 9) Expose and Run
# ---------------------
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
