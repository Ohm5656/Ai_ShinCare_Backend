from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ======================================
# สร้าง FastAPI app
# ======================================
app = FastAPI(
    title="AI Skin Analyzer Backend",
    description="API สำหรับวิเคราะห์สภาพผิวและจัดเก็บผลการสแกน",
    version="1.0.0"
)

# ======================================
# ตั้งค่า CORS (อนุญาตให้ frontend เรียก)
# ======================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ตอนทดสอบอนุญาตทุก origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# ตัวอย่าง endpoint ทดสอบระบบ
# ======================================
@app.get("/")
def root():
    return {"message": "AI Skin Analyzer Backend is running!"}

# ======================================
# โหลด routers
# ======================================
try:
    from app.routers import auth, users, scans, analyze, chat

    # ✅ รวม router เข้ากับ app หลัก
    app.include_router(auth.router, prefix="/auth", tags=["Auth"])
    app.include_router(users.router, prefix="/users", tags=["Users"])
    app.include_router(scans.router, prefix="/scans", tags=["Scans"])
    app.include_router(analyze.router, prefix="/analyze", tags=["Analyze"])
    app.include_router(chat.router, prefix="/chat", tags=["Chat"])

    print("✅ Routers loaded successfully.")
except Exception as e:
    print("⚠️ Warning: Some routers not loaded yet ->", e)
