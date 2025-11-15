# ===================================================================================
# redness_vit_local_multi.py — GlowbieBell Redness Analyzer (Hybrid + Multi-angle)
# -----------------------------------------------------------------------------------
# ⭐ ใช้ LOCAL ViT model: models/face_characteristics_vit
# ⭐ Hybrid: Hemoglobin Map (Hb) + ViT classifier
# ⭐ Robust ต่อแสง / กล้อง / ผิวเอเชีย
# ⭐ รองรับ 3 มุม: front / left / right
# ===================================================================================

import os
import numpy as np
import cv2
from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
)

# ===================================================================================
# CONFIG
# ===================================================================================

LOCAL_MODEL_DIR = os.path.join("models", "face_characteristics_vit")
red_processor = None
red_model = None


# ===================================================================================
# STEP 1: Normalize lighting
# ===================================================================================

def _normalize_lighting(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    L_eq = cv2.equalizeHist(L)
    img_norm = cv2.merge([L_eq, A, B])
    return cv2.cvtColor(img_norm, cv2.COLOR_LAB2RGB)


def _auto_face_crop(img):
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(keep_all=False)
        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            x1, y1, x2, y2 = [int(v) for v in boxes[0]]
            h, w = img.shape[:2]
            pad = int(0.12 * max(h, w))
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            return img[y1:y2, x1:x2]
    except Exception:
        pass
    return img


def _preprocess_redness_for_model(img_rgb: np.ndarray) -> np.ndarray:
    img = _normalize_lighting(img_rgb)
    img = _auto_face_crop(img)
    img = cv2.resize(img, (512, 512))
    return img


# ===================================================================================
# STEP 2: Hemoglobin Map (physics-based redness)
# ===================================================================================

def _hemo_map_risk(img_rgb: np.ndarray) -> float:
    """
    Hb = 0.299R - 0.172G - 0.131B → map ความแดงเชิง bio-optics
    """
    rgb = img_rgb.astype(np.float32) / 255.0
    Hb = 0.299 * rgb[..., 0] - 0.172 * rgb[..., 1] - 0.131 * rgb[..., 2]
    Hb_norm = (Hb - Hb.min()) / (Hb.max() - Hb.min() + 1e-6)
    Hb_norm = np.clip(Hb_norm, 0.0, 1.0)

    mean_val = float(np.mean(Hb_norm))
    q90_val = float(np.quantile(Hb_norm, 0.90))

    risk = np.clip((0.4 * mean_val + 0.6 * q90_val - 0.10) / 0.80, 0.0, 1.0)
    return float(risk)


# ===================================================================================
# STEP 3: Load LOCAL ViT model
# ===================================================================================

def _load_local_model() -> bool:
    global red_processor, red_model

    if red_model is not None:
        return True

    if not os.path.exists(LOCAL_MODEL_DIR):
        print("❌ Redness model folder not found:", LOCAL_MODEL_DIR)
        return False

    try:
        try:
            red_processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_DIR)
        except Exception:
            red_processor = AutoFeatureExtractor.from_pretrained(LOCAL_MODEL_DIR)

        red_model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_DIR)
        print(f"✅ Loaded LOCAL redness model from: {LOCAL_MODEL_DIR}")
        return True

    except Exception as e:
        print("❌ Cannot load LOCAL redness model →", e)
        red_model = None
        return False


# ===================================================================================
# STEP 4: ViT-based redness risk (ใช้ label ที่เกี่ยวกับสิว/แดง/อักเสบ)
# ===================================================================================

def _vit_redness_risk(img_for_model: np.ndarray) -> float | None:
    if not _load_local_model():
        return None

    try:
        inputs = red_processor(images=Image.fromarray(img_for_model), return_tensors="pt")
        logits = red_model(**inputs).logits
        probs = logits.softmax(dim=1)[0].detach().numpy()

        id2label = red_model.config.id2label

        redness_keywords = ["red", "flush", "rosacea", "irritation", "inflammation",
                            "acne", "pimple", "blemish"]

        redness_indices = []
        for k, v in id2label.items():
            v_low = v.lower()
            if any(kw in v_low for kw in redness_keywords):
                redness_indices.append(int(k))

        if not redness_indices:
            return None

        risk = float(np.clip(sum(probs[i] for i in redness_indices), 0.0, 1.0))
        return risk

    except Exception as e:
        print("⚠️ ViT redness inference failed →", e)
        return None


# ===================================================================================
# PUBLIC: SINGLE IMAGE
# ===================================================================================

def score_redness_single(img_pil: Image.Image) -> float:
    """
    วิเคราะห์ผิวแดงจากภาพเดี่ยว:
      - Normalize แสง
      - HbMap physics-based
      - ViT classifier จาก face_characteristics_vit
      - รวมผลแบบ adaptive
    """
    img_rgb = np.array(img_pil.convert("RGB"))

    # 1) Normalize lighting สำหรับ Hb และ ViT
    img_norm = _normalize_lighting(img_rgb)

    # 2) HbMap จากภาพเต็ม (เน้น vascular redness)
    hb_risk = _hemo_map_risk(img_norm)

    # 3) Preprocess เฉพาะส่วนหน้า ส่งเข้าโมเดล
    img_model = _preprocess_redness_for_model(img_rgb)
    vit_risk = _vit_redness_risk(img_model)

    # 4) รวมผลแบบ adaptive ตามความสว่าง
    brightness = np.mean(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)) / 255.0
    if brightness < 0.4:
        w_hb, w_vit = 0.7, 0.3
    elif brightness > 0.7:
        w_hb, w_vit = 0.4, 0.6
    else:
        w_hb, w_vit = 0.5, 0.5

    if vit_risk is not None:
        risk = w_hb * hb_risk + w_vit * vit_risk
    else:
        risk = hb_risk

    return float(np.clip(risk, 0.0, 1.0))


# ===================================================================================
# MULTI-ANGLE WRAPPER
# ===================================================================================

def score_redness_multi(
    front_img: Image.Image, left_img: Image.Image, right_img: Image.Image
) -> float:
    """
    รวมคะแนน redness จาก 3 มุม (front 50% + left 25% + right 25%)
    """
    r_front = score_redness_single(front_img)
    r_left = score_redness_single(left_img)
    r_right = score_redness_single(right_img)

    final = 0.5 * r_front + 0.25 * r_left + 0.25 * r_right
    return round(float(final), 4)


# ===================================================================================
# TEST
# ===================================================================================

if __name__ == "__main__":
    f = Image.open("front.jpg")
    l = Image.open("left.jpg")
    r = Image.open("right.jpg")
    val = score_redness_multi(f, l, r)
    print(f"🧪 Redness risk (0–1) = {val:.4f}")
