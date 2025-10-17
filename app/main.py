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
# ตั้งค่า CORS
# ======================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# Root ทดสอบระบบ
# ======================================
@app.get("/")
def root():
    return {"message": "AI Skin Analyzer Backend is running!"}

# ======================================
# โหลด router auth เท่านั้นก่อน
# ======================================
try:
    from app.routers import auth
    app.include_router(auth.router, prefix="/auth", tags=["Auth"])
    print("✅ Router 'auth' loaded successfully.")
except Exception as e:
    print("⚠️ Warning: Auth router not loaded ->", e)
