FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# === ติดตั้ง tools สำหรับ build insightface + runtime libs ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ make \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 tzdata \
    && ln -snf /usr/share/zoneinfo/Asia/Bangkok /etc/localtime && echo Asia/Bangkok > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --force-reinstall -r requirements.txt

COPY . .

ENV STORAGE_DIR=/data/uploads \
    INSIGHTFACE_PROVIDER=CPUExecutionProvider \
    DETECT_SIZE=640 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

RUN mkdir -p /data/uploads

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
