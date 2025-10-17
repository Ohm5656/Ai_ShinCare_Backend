from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import analyze

app = FastAPI(
    title="AI Skin Analyzer Backend",
    description="API สำหรับวิเคราะห์สภาพผิวและจัดเก็บผลการสแกน",
    version="1.0.0",
)

# CORS: ใส่โดเมน frontend ของโอห์มด้วย
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # โปรดเปลี่ยนเป็นโดเมนจริงตอนโปรดักชัน
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# รวม router
app.include_router(analyze.router, prefix="/analyze", tags=["Analyze"])

@app.get("/")
def root():
    return {"message": "AI Skin Analyzer Backend is running!"}

@app.get("/health")
def health():
    return {"ok": True}
