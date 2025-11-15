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
pig_processor = None
pig_model = None


# ===================================================================================
# PREPROCESS (โฟกัสเม็ดสี + จุดเข้ม)
# ===================================================================================

def _white_balance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    L2 = cv2.equalizeHist(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB)


def _skin_tone_normalize(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)


def _enhance_pigment_clahe(img):
    # เน้น contrast ของจุดเข้ม (L-channel)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB)


def _denoise_soft(img):
    # ลด noise แบบไม่ทำให้จุดฝ้ากระหายไป
    return cv2.bilateralFilter(img, d=7, sigmaColor=40, sigmaSpace=40)


def _auto_face_crop(img):
    """Crop เฉพาะใบหน้า ถ้าหาไม่ได้จะคืนภาพเดิม"""
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


def _preprocess_pigmentation(img_rgb: np.ndarray) -> np.ndarray:
    """ขั้นตอนเตรียมภาพก่อนเข้าโมเดลฝ้า/กระ"""
    img = _white_balance(img_rgb)
    img = _skin_tone_normalize(img)
    img = _enhance_pigment_clahe(img)
    img = _denoise_soft(img)
    img = _auto_face_crop(img)
    img = cv2.resize(img, (512, 512))
    return img


# ===================================================================================
# LOAD LOCAL MODEL
# ===================================================================================

def _load_local_model() -> bool:
    global pig_processor, pig_model

    if pig_model is not None:
        return True

    if not os.path.exists(LOCAL_MODEL_DIR):
        print("❌ Pigmentation model folder not found:", LOCAL_MODEL_DIR)
        return False

    try:
        try:
            pig_processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_DIR)
        except Exception:
            pig_processor = AutoFeatureExtractor.from_pretrained(LOCAL_MODEL_DIR)

        pig_model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_DIR)
        print(f"✅ Loaded LOCAL pigmentation model from: {LOCAL_MODEL_DIR}")
        return True

    except Exception as e:
        print("❌ Cannot load LOCAL pigmentation model →", e)
        pig_model = None
        return False


# ===================================================================================
# FALLBACK: LAB variance + จุดมืด
# ===================================================================================

def _fallback_pigmentation(img_rgb: np.ndarray) -> float:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    var_ab = float(np.var(A) + np.var(B)) / 5000.0

    L_norm = L.astype(np.float32) / 255.0
    mean_L, std_L = np.mean(L_norm), np.std(L_norm)
    dark_mask = L_norm < (mean_L - 0.25 * std_L)
    dark_ratio = float(np.mean(dark_mask))

    risk = np.clip(0.6 * dark_ratio + 0.4 * var_ab * 2.0, 0.0, 1.0)
    return float(risk)


# ===================================================================================
# SINGLE IMAGE RISK (LOCAL MODEL)
# ===================================================================================

def score_pigmentation_single(img_pil: Image.Image) -> float:
    """
    วิเคราะห์ฝ้า/กระ/จุดด่างดำจากภาพเดี่ยว → risk 0..1
    """
    img_rgb = np.array(img_pil.convert("RGB"))
    img_prep = _preprocess_pigmentation(img_rgb)

    if not _load_local_model():
        return _fallback_pigmentation(img_prep)

    try:
        inputs = pig_processor(images=Image.fromarray(img_prep), return_tensors="pt")
        logits = pig_model(**inputs).logits
        probs = logits.softmax(dim=1)[0].detach().numpy()

        id2label = pig_model.config.id2label

        pigment_indices = []
        for k, v in id2label.items():
            v_low = v.lower()
            if (
                "pigment" in v_low
                or "hyperpig" in v_low
                or "dark spot" in v_low
                or "dark spots" in v_low
                or "spot" in v_low
                or "blemish" in v_low
            ):
                pigment_indices.append(int(k))

        if not pigment_indices:
            # ถ้าไม่มี label ฝ้า/กระชัดเจน → fallback heuristic
            return _fallback_pigmentation(img_prep)

        risk = float(np.clip(sum(probs[i] for i in pigment_indices), 0.0, 1.0))
        return risk

    except Exception as e:
        print("⚠️ Pigmentation inference failed →", e)
        return _fallback_pigmentation(img_prep)


# ===================================================================================
# MULTI-ANGLE WRAPPER
# ===================================================================================

def score_pigmentation_multi(
    front_img: Image.Image, left_img: Image.Image, right_img: Image.Image
) -> float:
    """
    รวมคะแนนฝ้า/กระจาก 3 มุม (front 50% + left 25% + right 25%)
    """
    r_front = score_pigmentation_single(front_img)
    r_left = score_pigmentation_single(left_img)
    r_right = score_pigmentation_single(right_img)

    final_score = 0.5 * r_front + 0.25 * r_left + 0.25 * r_right
    return round(float(final_score), 4)


# ===================================================================================
# TEST
# ===================================================================================

if __name__ == "__main__":
    f = Image.open("front.jpg")
    l = Image.open("left.jpg")
    r = Image.open("right.jpg")
    val = score_pigmentation_multi(f, l, r)
    print(f"🧪 Pigmentation risk (0–1) = {val:.4f}")
