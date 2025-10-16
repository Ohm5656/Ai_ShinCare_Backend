# AI Skin Analyzer Backend (FastAPI)

## Run (Dev)
1) Python 3.10+
2) `pip install -r requirements.txt`
3) คัดลอก `.env.example` เป็น `.env` แล้วแก้ `DATABASE_URL` (dev ใช้ SQLite ได้เลย)
4) สร้างตาราง (วิธีง่ายสุด):
   ```python
   # สร้างไฟล์ scripts/create_db.py แล้วรันครั้งเดียว
   from app.db.session import engine
   from app.db.base import Base
   from app.db import models
   Base.metadata.create_all(bind=engine)
