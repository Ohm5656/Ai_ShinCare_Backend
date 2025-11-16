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
        คุณคือ Dr.SkinAI ผู้เชี่ยวชาญด้านผิวหนังและสกินแคร์สำหรับคนเอเชีย
        ตอบเป็นภาษาไทย ชัดเจน เป็นกันเอง และให้คำแนะนำแบบปรับตามสภาพผิว
        หลีกเลี่ยงการวินิจฉัยโรคแบบแทนแพทย์ ให้เน้นการดูแลผิวทั่วไปและการไปพบแพทย์เมื่อจำเป็น
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

        sex = payload.sex
        age_range = payload.age_range
        skin_type = payload.skin_type
        sensitive = payload.sensitive
        concerns = payload.concerns

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
        prof = Profile(age=age_range, sex=sex, skin_type=skin_type)
        result = fusion.predict(scores, prof)

        # ---------------- 4) เตรียม prompt (ยาว) ----------------
        long_prompt = f"""
        คุณคือ Dr.SkinAI ผู้เชี่ยวชาญด้านผิวหนังสำหรับคนเอเชีย
        ช่วยสรุปและให้คำแนะนำการดูแลผิวจากผลวิเคราะห์ต่อไปนี้เป็นภาษาไทย

        คะแนนรวมภาพรวมผิว: {result['overall_score']}/100

        คะแนนรายมิติ (ยิ่งสูงคือยิ่งดี):
        - ริ้วรอย (wrinkles): {result['dimension_scores']['wrinkles']}
        - ความหย่อนคล้อย (sagging): {result['dimension_scores']['sagging']}
        - ฝ้า/กระ/จุดด่างดำ (pigmentation): {result['dimension_scores']['pigmentation']}
        - สิว/รอยสิว (acne): {result['dimension_scores']['acne']}
        - ผิวแดง/ระคายเคือง (redness): {result['dimension_scores']['redness']}
        - ความเรียบเนียน/รูขุมขน (texture): {result['dimension_scores']['texture']}
        - ความสม่ำเสมอโทนสีผิว (tone): {result['dimension_scores']['tone']}

        โปรไฟล์ผู้ใช้:
        - เพศ: {sex}
        - ช่วงอายุ: {age_range}
        - ประเภทผิว: {skin_type}
        - ผิวแพ้ง่าย: {"ใช่" if sensitive else "ไม่ใช่"}
        - ความกังวลหลัก: {concerns or "ไม่ระบุ"}

        ให้คุณ:
        1) อธิบายภาพรวมสภาพผิวสั้น ๆ
        2) สรุปจุดเด่นของผิว
        3) สรุปสิ่งที่ควรระวังหรือจุดที่ควรปรับปรุง
        4) แนะนำรูทีนดูแลผิวแบบเข้าใจง่าย โดยแบ่งเป็น:
           - Cleanser
           - Treatment / Serum
           - Moisturizer
           - Sunscreen
        5) ปิดท้ายด้วยประโยคสั้น ๆ ว่า
           "ถ้ามีข้อสงสัยเพิ่มเติม สามารถถาม Dr.SkinAI ต่อได้เลยครับ/ค่ะ"
        """

        # ---------------- 4.2) Prompt สั้น (JSON + summary) ----------------
        short_prompt = f"""
        คุณคือ Dr.SkinAI ผู้เชี่ยวชาญด้านผิวหนัง

        จากคะแนนผิวด้านล่างนี้ ช่วยสรุปผลแบบกระชับ และต้องตอบเป็น JSON เท่านั้น  
        ไม่มีข้อความอื่นก่อนหรือหลัง JSON ห้ามมีคำอธิบายเพิ่ม

        โครงสร้าง JSON ที่ต้องส่งกลับ:
        {{
        "summary": "<สรุปผิว 1 ประโยค>",
        "highlights_short": ["<จุดเด่น1>", "<จุดเด่น2>"],
        "improvements_short": ["<ข้อควรปรับปรุง1>", "<ข้อควรปรับปรุง2>"]
        }}

        เงื่อนไขสำคัญ:
        - ทุกข้อความต้องเป็นภาษาไทย
        - summary ต้องเป็นประโยคเดียวเท่านั้น
        - highlights_short = จุดเด่น 1–3 ข้อ (ข้อความสั้น)
        - improvements_short = สิ่งที่ควรปรับปรุง 1–3 ข้อ (ข้อความสั้น)
        - ห้ามตอบในรูปแบบ Markdown
        - ห้ามตอบด้วยคำว่า ```json

        ข้อมูลที่ต้องใช้วิเคราะห์:
        คะแนนรวม: {result['overall_score']}/100
        คะแนนรายด้าน (ยิ่งสูงคือยิ่งดี): {result['dimension_scores']}
        โปรไฟล์: เพศ={sex}, อายุ={age_range}, ผิว={skin_type}, แพ้ง่าย={"ใช่" if sensitive else "ไม่ใช่"}, กังวล="{concerns or "ไม่ระบุ"}"

        ตัวอย่างรูปแบบการตอบ (อย่าใช้ข้อความนี้):
        {{
        "summary": "ผิวโดยรวมดีแต่มีรอยแดงเล็กน้อย",
        "highlights_short": ["ความชุ่มชื้นค่อนข้างดี"],
        "improvements_short": ["มีจุดด่างดำบางส่วน"]
        }}
        """

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

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
        if long_resp.status_code != 200:
            raise HTTPException(status_code=long_resp.status_code, detail=long_resp.text)

        ai_advice_long = long_resp.json()["choices"][0]["message"]["content"]


        # ---------------- 6) OpenAI SHORT ----------------
        short_req = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "คุณต้องตอบเป็น JSON เท่านั้น ห้ามมีคำอื่น"},
                {"role": "user", "content": short_prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 300
        }

        short_resp = requests.post(OPENAI_API_URL, headers=headers, json=short_req, timeout=60)
        if short_resp.status_code != 200:
            raise HTTPException(status_code=short_resp.status_code, detail=short_resp.text)

        # ---------------- Parse JSON แบบแข็งแรงที่สุด ----------------
        try:
            raw_text = short_resp.json()["choices"][0]["message"]["content"]

            # ลบ ```json และ ``` ถ้ามี (AI ชอบใส่เอง)
            cleaned = raw_text.strip()
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

            short = json.loads(cleaned)

        except Exception as e:
            print("❌ JSON parse fail:", e, " raw AI:", raw_text)
            short = {
                "summary": "",
                "highlights_short": [],
                "improvements_short": []
            }


        # ---------------- RETURN RESPONSE ----------------
        return {
            "overall_score": result["overall_score"],
            "dimension_scores": result["dimension_scores"],
            "weighted_contrib": result["weighted_contrib"],
            "mode": result["mode"],

            "summary": short.get("summary", ""),
            "highlights_short": short.get("highlights_short", []),
            "improvements_short": short.get("improvements_short", []),

            "ai_advice": ai_advice_long,

            "profile": {
                "sex": sex,
                "age_range": age_range,
                "skin_type": skin_type,
                "sensitive": bool(sensitive),
                "concerns": concerns,
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
