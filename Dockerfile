FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    tzdata \
 && ln -snf /usr/share/zoneinfo/Asia/Bangkok /etc/localtime && echo Asia/Bangkok > /etc/timezone \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install --no-cache-dir insightface==0.7.3 && pip install --prefer-binary --no-cache-dir -r requirements.txt


COPY . .

ENV STORAGE_DIR=/data/uploads \
    INSIGHTFACE_PROVIDER=CPUExecutionProvider \
    DETECT_SIZE=640 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /data/uploads

EXPOSE 8000
CMD ["bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

