# ==================================================================================================
# main.py — GlowbieBell + Dr.SkinAI Backend (NO DATABASE VERSION)
# --------------------------------------------------------------------------------------------------
# ❗ เวอร์ชันนี้ปิดระบบ Database ชั่วคราว เพื่อให้สแกนผ่านได้ก่อน (ไม่ error 500)
# ❗ ฟังก์ชัน save_scan จะถูก "ปิดการทำงาน" แต่ตัว Router ยังอยู่เหมือนเดิม
# ==================================================================================================

from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests, os, cv2
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# 🔹 Analyzer imports (พี่ใช้ Pure CV รุ่นใหม่แทนได้ในภายหลัง)
from analyzers.wrinkles_ffhq import score_wrinkles_multi
from analyzers.sagging_facemesh import score_sagging
from analyzers.pigmentation_vit import score_pigmentation_multi
from analyzers.acne_vit import score_acne_multi
from analyzers.redness_vit_or_hemo import score_redness_multi
from analyzers.texture_unet_or_opencv import score_texture
from analyzers.tone_lab import score_tone_multiview

from skin_fusion_model import SkinFusion, Profile
import base64
import io
import json

# ===================================================================================
# 🔹 โหลดค่า API Key
# ===================================================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

print("✅ DEBUG: Loaded OPENAI_API_KEY =", OPENAI_API_KEY[:10] + "..." if OPENAI_API_KEY else "❌ None")

# ===================================================================================
# 🔹 DEBUG ดูไฟล์โมเดล
# ===================================================================================
print("===== DEBUG: LIST MODELS FOLDER =====")
for root, dirs, files in os.walk("models"):
    print(root, files)
print("======================================")

# ===================================================================================
# 🔹 FastAPI + Routers
# ===================================================================================
app = FastAPI(title="GlowbieBell Backend", version="2.0.0")

# ❗ Router เดิม KEEP ไว้ (แต่ save_scan จะไม่ทำงานจริง)
from routers.history_router import router as history_router
from routers.scan_router import router as scan_router

app.include_router(history_router)
app.include_router(scan_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aishincarefrontend-production.up.railway.app",
        "http://localhost:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================================================================================
# 🔹 หน้าแรก
# ===================================================================================
@app.get("/")
async def root():
    return {"message": "🩺 GlowbieBell x DrSkinAI backend running OK (NO-DB mode)."}

# ===================================================================================
# 🔹 Chatbot
# ===================================================================================
class PromptRequest(BaseModel):
    prompt: str

@app.post("/ask-ai")
async def ask_ai(request: PromptRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")

    system_prompt = """
        คุณคือ Dr.SkinAI ผู้เชี่ยวชาญด้านผิวหนัง...
        (ข้อความ system prompt ของพี่ ผมไม่แก้)
    """

    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 500,
    }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=60)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return {"answer": response.json()["choices"][0]["message"]["content"]}

# ===================================================================================
# 🔹 Model จาก FE
# ===================================================================================
class FaceAnalyzePayload(BaseModel):
    front: str
    left: str
    right: str
    sex: str
    age_range: str
    skin_type: str
    sensitive: bool = False
    concerns: str = ""

# ===================================================================================
# 🔹 decode base64
# ===================================================================================
def decode_base64_to_image(b64_str: str) -> Image.Image:
    if b64_str.startswith("data:image"):
        b64_str = b64_str.split(",", 1)[1]

    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img

# ===================================================================================
# 🔹 Endpoint วิเคราะห์ผิว
# ===================================================================================
@app.post("/analyze-face-full")
async def analyze_face_full(payload: FaceAnalyzePayload):
    try:
        # ---------------- 1) decode ----------------
        imgF = decode_base64_to_image(payload.front)
        imgL = decode_base64_to_image(payload.left)
        imgR = decode_base64_to_image(payload.right)

        # ---------------- 2) วิเคราะห์ 7 มิติ ----------------
        scores = {
            "wrinkles": score_wrinkles_multi(imgF, imgL, imgR),
            "sagging": score_sagging(imgF, imgL, imgR),
            "pigmentation": score_pigmentation_multi(imgF, imgL, imgR),
            "acne": score_acne_multi(imgF, imgL, imgR),
            "redness": score_redness_multi(imgF, imgL, imgR),
            "texture": score_texture(imgF),
            "tone": score_tone_multiview(imgF, imgL, imgR),
        }

        # ---------------- 3) รวมคะแนน ----------------
        fusion = SkinFusion()
        prof = Profile(age=payload.age_range, sex=payload.sex, skin_type=payload.skin_type)
        result = fusion.predict(scores, prof)

        # ---------------- 4) เตรียม prompt ----------------
        long_prompt = f"""
        (ข้อความ prompt ยาวของพี่ ผมไม่แก้)
        คะแนนรวม: {result['overall_score']}/100
        ...
        """

        short_prompt = f"""
        วิเคราะห์ผลคะแนนด้านผิว -> ขอ JSON สั้นๆ
        คะแนนรวม: {result['overall_score']}/100
        ...
        """

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

        # ---------------- 5) OpenAI LONG ----------------
        long_req = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "คุณคือ DrSkinAI ผู้เชี่ยวชาญผิวสำหรับคนเอเชีย"},
                {"role": "user", "content": long_prompt},
            ],
            "temperature": 0.8,
            "max_tokens": 700
        }
        long_resp = requests.post(OPENAI_API_URL, headers=headers, json=long_req, timeout=60)
        ai_advice_long = long_resp.json()["choices"][0]["message"]["content"]

        # ---------------- 6) OpenAI SHORT ----------------
        short_req = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "คุณคือ DrSkinAI ผู้เชี่ยวชาญผิวสำหรับคนเอเชีย"},
                {"role": "user", "content": short_prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 300
        }
        short_resp = requests.post(OPENAI_API_URL, headers=headers, json=short_req, timeout=60)

        try:
            short_json = short_resp.json()["choices"][0]["message"]["content"]
            short = json.loads(short_json)
        except Exception:
            short = {"highlights_short": [], "improvements_short": []}

        # ===================================================================================
        # ❗ 7) ปิดระบบบันทึกประวัติ (NO DATABASE MODE)
        # ===================================================================================
        # from routers.scan_router import save_scan
        # from database import SessionLocal
        # db = SessionLocal()
        # try:
        #     save_scan({...}, db)
        # finally:
        #     db.close()

        # (👆 ปิดการบันทึก, แต่ไม่ลบ code เดิม)

        # ===================================================================================
        # 8) RETURN RESPONSE
        # ===================================================================================
        return {
            "overall_score": result["overall_score"],
            "dimension_scores": result["dimension_scores"],
            "weighted_contrib": result["weighted_contrib"],
            "mode": result["mode"],

            "highlights_short": short.get("highlights_short", []),
            "improvements_short": short.get("improvements_short", []),
            "ai_advice": ai_advice_long,

            "profile": {
                "sex": payload.sex,
                "age_range": payload.age_range,
                "skin_type": payload.skin_type,
                "sensitive": bool(payload.sensitive),
                "concerns": payload.concerns,
            },

            "top_issue": max(result["dimension_scores"], key=result["dimension_scores"].get),
            "improvement": 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Internal error: {e}")


# ===================================================================================
# 🔹 Local run
# ===================================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
