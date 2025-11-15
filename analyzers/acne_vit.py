import os
import numpy as np
import cv2
from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoFeatureExtractor,
    AutoModelForImageClassification
)

# ===================================================================================
# CONFIG
# ===================================================================================

LOCAL_MODEL_DIR = os.path.join("models", "acne_vit")
pipe_processor = None
pipe_model = None


# ===================================================================================
# PREPROCESS PRO – Trust Me, This Makes Huge Difference
# ===================================================================================

def white_balance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    L2 = cv2.equalizeHist(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB)


def skin_normalize(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)


def enhance_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB)


def sharpen_skin(img):
    blur = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)
    sharp = cv2.addWeighted(img, 1.7, blur, -0.7, 0)
    return sharp


def auto_face_crop(img):
    """ใช้ MTCNN crop เฉพาะหน้า — ถ้าไม่เจอให้ใช้ภาพเต็ม"""
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(keep_all=False)
        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            x1, y1, x2, y2 = [int(v) for v in boxes[0]]
            h, w = img.shape[:2]
            pad = int(0.1 * max(h, w))
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            return img[y1:y2, x1:x2]
    except:
        pass
    return img


def preprocess_acne(img_rgb):
    """ขั้นตอนรวมทั้งหมดก่อนส่งเข้าโมเดล"""
    img = white_balance(img_rgb)
    img = skin_normalize(img)
    img = enhance_clahe(img)
    img = sharpen_skin(img)
    img = auto_face_crop(img)
    img = cv2.resize(img, (512, 512))
    return img


# ===================================================================================
# LOAD LOCAL MODEL
# ===================================================================================

def load_local_model():
    global pipe_processor, pipe_model

    if pipe_model is not None:
        return True

    if not os.path.exists(LOCAL_MODEL_DIR):
        print("❌ Acne model folder not found:", LOCAL_MODEL_DIR)
        return False

    try:
        try:
            pipe_processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_DIR)
        except:
            pipe_processor = AutoFeatureExtractor.from_pretrained(LOCAL_MODEL_DIR)

        pipe_model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_DIR)

        print(f"✅ Loaded LOCAL acne model from: {LOCAL_MODEL_DIR}")
        return True

    except Exception as e:
        print("❌ Cannot load LOCAL acne model →", e)
        pipe_model = None
        return False


# ===================================================================================
# FALLBACK (OpenCV)
# ===================================================================================

def fallback_acne_risk(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((h < 15) | (h > 170)) & (s > 80) & (v > 60)
    mask = mask.astype(np.uint8) * 255
    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small = [c for c in cnt if 5 < cv2.contourArea(c) < 150]
    density = min(1.0, len(small) / 120.0)
    return float(density)


# ===================================================================================
# SINGLE IMAGE AI INFERENCE
# ===================================================================================

def score_acne_single(img_pil):
    img = np.array(img_pil.convert("RGB"))
    img = preprocess_acne(img)

    if not load_local_model():
        return fallback_acne_risk(img)

    try:
        inputs = pipe_processor(images=Image.fromarray(img), return_tensors="pt")
        logits = pipe_model(**inputs).logits
        probs = logits.softmax(dim=1)[0].detach().numpy()

        # หา index ของระดับสิว
        id2label = pipe_model.config.id2label
        idx_clear = None
        idx_mild = None
        idx_moderate = None
        idx_severe = None

        for k, v in id2label.items():
            v = v.lower()
            if "clear" in v:
                idx_clear = int(k)
            if "mild" in v:
                idx_mild = int(k)
            if "moderate" in v:
                idx_moderate = int(k)
            if "severe" in v:
                idx_severe = int(k)

        # ถ้า label ไม่ครบ fallback
        if None in [idx_clear, idx_mild, idx_moderate, idx_severe]:
            return fallback_acne_risk(img)

        risk = (
            0.0   * probs[idx_clear] +
            0.3   * probs[idx_mild] +
            0.6   * probs[idx_moderate] +
            1.0   * probs[idx_severe]
        )

        return float(np.clip(risk, 0.0, 1.0))

    except Exception as e:
        print("⚠️ Acne inference failed →", e)
        return fallback_acne_risk(img)


# ===================================================================================
# MULTI ANGLE SCORING (front + left + right)
# ===================================================================================

def score_acne_multi(front_img, left_img, right_img):
    rF = score_acne_single(front_img)
    rL = score_acne_single(left_img)
    rR = score_acne_single(right_img)

    final = (0.5 * rF) + (0.25 * rL) + (0.25 * rR)
    return round(float(final), 4)


# ===================================================================================
# TEST
# ===================================================================================

if __name__ == "__main__":
    front = Image.open("front.jpg")
    left = Image.open("left.jpg")
    right = Image.open("right.jpg")
    print("Acne score:", score_acne_multi(front, left, right))
