# =========================================================
# 1) Base image
# =========================================================
FROM gcr.io/distroless/python3-debian12


# =========================================================
# 2) Working directory
# =========================================================
WORKDIR /app

# =========================================================
# 3) Copy dependencies first
# =========================================================
COPY requirements.txt .

# =========================================================
# 4) Install system libs for OpenCV + InsightFace
# =========================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    tzdata \
 && ln -snf /usr/share/zoneinfo/Asia/Bangkok /etc/localtime && echo Asia/Bangkok > /etc/timezone \
 && rm -rf /var/lib/apt/lists/*

# =========================================================
# 5) Install Python dependencies
# =========================================================
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# =========================================================
# 6) Copy all source code
# =========================================================
COPY . .

# =========================================================
# 7) Environment Variables
# =========================================================
ENV STORAGE_DIR=/data/uploads \
    INSIGHTFACE_PROVIDER=CPUExecutionProvider \
    DETECT_SIZE=640 \
    PYTHONUNBUFFERED=1

# =========================================================
# 8) Create directories for persistent storage
# =========================================================
RUN mkdir -p /data/uploads

# =========================================================
# 9) Expose and run
# =========================================================
EXPOSE 8000
CMD ["bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
