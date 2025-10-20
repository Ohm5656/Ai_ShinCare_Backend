from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import analyze, scans

app = FastAPI(
    title="AI Skin Analyzer Backend",
    description="API สำหรับตรวจมุมใบหน้าและวิเคราะห์สภาพผิว",
    version="1.0.0",
)

# CORS (เปิดให้ frontend เข้าถึง)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# รวม routers
app.include_router(analyze.router, prefix="/analyze", tags=["Analyze"])
app.include_router(scans.router, prefix="/scan", tags=["Scan"])

@app.get("/")
def root():
    return {"message": "AI Skin Analyzer Backend is running!"}

@app.get("/health")
def health():
    return {"ok": True}
