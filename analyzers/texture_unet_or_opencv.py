import os
import requests
import numpy as np
import cv2
from PIL import Image

# ===================================================================================
# MODEL CONFIG
# ===================================================================================

MODEL_PATH = "models/texture.h5"
MODEL_URL = "https://raw.githubusercontent.com/Himika-Mishra/FaceAnalysisApp/main/more_data(3).h5"


def ensure_model():
    """
    ถ้ายังไม่มีไฟล์โมเดล .h5 → ดาวน์โหลดจาก GitHub มาเก็บในโฟลเดอร์ models/
    - ใช้ได้ทั้ง local และ Railway (container start แล้วโหลด)
    """
    if os.path.exists(MODEL_PATH):
        return

    try:
        print("⬇️ Downloading Himika-Mishra texture model (more_data(3).h5)...")
        os.makedirs("models", exist_ok=True)
        r = requests.get(MODEL_URL, stream=True, timeout=120)
        total = 0
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
        print(f"✅ Model downloaded: {total/1e6:.2f} MB saved to {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Cannot download texture model: {e}")


# ===================================================================================
# PREPROCESSING PRO — ปรับแสง + crop หน้า ให้เหมาะกับการวิเคราะห์ texture
# ===================================================================================

def _illumination_fix(img_rgb: np.ndarray) -> np.ndarray:
    """
    ปรับแสงสำหรับ “ผิว” โดยเฉพาะ:
      1) Retinex SSR → ลดเงา/หน้าแสงไม่สม่ำเสมอ
      2) CLAHE (LAB) → ดึงรายละเอียด L* (texture ผิว)
      3) Sharpen → เน้นรูขุมขน / ขรุขระให้ชัดขึ้น
    """
    # --- 1) Retinex SSR ---
    img_f = img_rgb.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(img_f, (0, 0), 60)
    ssr = np.log(img_f) - np.log(blur + 1.0)
    ssr = ssr - ssr.min()
    ssr = (255.0 * ssr / (ssr.max() + 1e-6)).astype(np.uint8)

    # --- 2) CLAHE บน L-channel ---
    lab = cv2.cvtColor(ssr, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    img2 = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB)

    # --- 3) Sharpen (เน้นขอบละเอียด) ---
    blur2 = cv2.GaussianBlur(img2, (0, 0), 3)
    sharp = cv2.addWeighted(img2, 1.5, blur2, -0.5, 0)

    return sharp


def _soft_face_crop(img_rgb: np.ndarray) -> np.ndarray:
    """
    Soft crop: ตัดภาพให้เหลือเฉพาะใบหน้าช่วงกลางๆ
    - ไม่ใช่ FaceMesh → เร็วและ robust บนทุกภาพ
    - ลดผลรบกวนจากผม / เสื้อ / background
    """
    h, w, _ = img_rgb.shape
    # กำหนดเป็นสัดส่วน (ประมาณ: ตัดหัวนิดนึง, ตัดคาง/หูออกหน่อย)
    y1 = int(0.12 * h)
    y2 = int(0.88 * h)
    x1 = int(0.18 * w)
    x2 = int(0.82 * w)
    if y2 <= y1 or x2 <= x1:
        return img_rgb
    return img_rgb[y1:y2, x1:x2]


# ===================================================================================
# DEEP LEARNING BACKEND (U-Net++)
# ===================================================================================

def _dl_score(img_rgb: np.ndarray) -> float | None:
    """
    พยายามใช้โมเดล U-Net++ (.h5) วิเคราะห์:
      - input: 224x224 RGB (หลังปรับแสง+crop)
      - output: mask 0..1 (pixel ที่คิดว่าเป็นรูขุมขน/ผิวสาก)
      - risk = สัดส่วน pixel ที่ > 0.5
    ถ้าโมเดลมีปัญหา → คืน None เพื่อให้ fallback ทำงานต่อ
    """
    try:
        ensure_model()
        if not os.path.exists(MODEL_PATH):
            print("⚠️ Texture model file not found after ensure_model.")
            return None

        from tensorflow.keras.models import load_model

        model = load_model(MODEL_PATH, compile=False)

        # ----- PREPROCESS PRO -----
        img_fix = _illumination_fix(img_rgb)
        img_crop = _soft_face_crop(img_fix)

        img_resized = cv2.resize(img_crop, (224, 224), interpolation=cv2.INTER_AREA)
        x = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

        y_pred = model.predict(x, verbose=0)[0]  # [H,W,1] or [H,W]
        mask = y_pred
        if mask.ndim == 3:
            mask = mask[..., 0]

        pores_ratio = float(np.mean(mask > 0.5))
        return float(np.clip(pores_ratio, 0.0, 1.0))

    except Exception as e:
        print(f"⚠️ Texture DL model failed ({e}); fallback to OpenCV metric.")
        return None


# ===================================================================================
# FALLBACK: Pure-OpenCV Texture Metric (GLCM-like + Laplacian)
# ===================================================================================

def _fallback_texture(img_rgb: np.ndarray) -> float:
    """
    ใช้เฉพาะ OpenCV (ไม่มีโมเดล)
      - ปรับแสง + crop หน้า แบบเดียวกับ deep model
      - ใช้ Laplacian mean + variance หลัง Gaussian → ประมาณความสากของผิว
      - คืนค่า risk 0..1 (มาก = ผิวสาก / รูขุมขนชัด)
    """
    img_fix = _illumination_fix(img_rgb)
    img_crop = _soft_face_crop(img_fix)

    gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # 1) Laplacian roughness
    lap = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_32F, ksize=3)
    lap_mean = float(np.mean(np.abs(lap)))  # ยิ่งสูง = ผิวมี texture ชัด

    # 2) GLCM-like contrast (approx) ด้วย Laplacian หลัง blur
    gl = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gl2 = cv2.Laplacian((gl * 255).astype(np.uint8), cv2.CV_32F)
    contrast = float(np.var(gl2) / 5000.0)

    # 3) Fusion →
    #   - lap_mean ~ [5..25] บนภาพมือถือทั่วไป
    #   - contrast มาช่วยย้ำ texture
    risk = np.clip(0.65 * (lap_mean / 18.0) + 0.35 * contrast, 0.0, 1.0)
    return float(risk)


# ===================================================================================
# PUBLIC API — ใช้ใน main.py
# ===================================================================================

def score_texture(img_pil: Image.Image) -> float:
    """
    วิเคราะห์ “ผิวเรียบเนียน / รูขุมขน”
    return: risk ∈ [0,1]  (มาก = ผิวสาก, รูขุมขนกว้าง/ชัด)
    """
    img_rgb = np.array(img_pil.convert("RGB"))

    # พยายามใช้ Deep Model ก่อน
    val = _dl_score(img_rgb)
    if val is not None:
        return float(val)

    # ถ้าโมเดล load/predict ไม่ได้ → ใช้ OpenCV แทน
    return _fallback_texture(img_rgb)


# ===================================================================================
# CLI TEST
# ===================================================================================

if __name__ == "__main__":
    p = "sample_face.jpg"
    if os.path.isfile(p):
        img = Image.open(p)
        s = score_texture(img)
        print(f"🧪 Texture Risk = {s:.3f} ({s*100:.1f}%)")
    else:
        print("ℹ️ Put a sample image at sample_face.jpg to test.")
