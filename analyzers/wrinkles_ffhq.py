import os
import numpy as np
import cv2
from PIL import Image

# HuggingFace local loader
from transformers import (
    AutoImageProcessor,
    AutoFeatureExtractor,
    AutoModelForImageClassification
)

# ===================================================================================
# CONFIG
# ===================================================================================
LOCAL_MODEL_DIR = os.path.join("models", "wrinkles_vit")
pipe_processor = None
pipe_model = None


# ===================================================================================
# PREPROCESSING PRO (เหมาะกับริ้วรอยที่สุด)
# ===================================================================================

def white_balance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    L2 = cv2.equalizeHist(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB)


def skin_tone_normalize(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)


def enhance_wrinkle_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB)


def denoise_and_sharpen(img):
    blur = cv2.bilateralFilter(img, d=7, sigmaColor=40, sigmaSpace=40)
    sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return sharp


def auto_face_crop(img):
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(keep_all=False)
        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            x1, y1, x2, y2 = [int(v) for v in boxes[0]]
            h, w = img.shape[:2]
            pad = int(0.1 * max(h, w))
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            return img[y1:y2, x1:x2]
    except:
        pass
    return img


def preprocess_wrinkle(img_rgb):
    img = white_balance(img_rgb)
    img = skin_tone_normalize(img)
    img = enhance_wrinkle_clahe(img)
    img = denoise_and_sharpen(img)
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
        print("❌ Wrinkle local model folder not found.")
        return False

    try:
        # ตัวใหม่ใช้ AutoImageProcessor
        try:
            pipe_processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_DIR)
        except:
            pipe_processor = AutoFeatureExtractor.from_pretrained(LOCAL_MODEL_DIR)

        pipe_model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_DIR)

        print(f"✅ Loaded LOCAL wrinkle model from: {LOCAL_MODEL_DIR}")
        return True

    except Exception as e:
        print(f"❌ Cannot load LOCAL wrinkle model → {e}")
        pipe_model = None
        return False


# ===================================================================================
# FALLBACK
# ===================================================================================

def fallback_wrinkle_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    edge_strength = float(np.mean(np.abs(lap)))
    risk = np.clip((edge_strength - 3.0) / 20.0, 0.0, 1.0)
    return float(risk)


# ===================================================================================
# SINGLE IMAGE INFERENCE
# ===================================================================================

def score_wrinkles_single(img_pil):
    img = np.array(img_pil.convert("RGB"))
    img = preprocess_wrinkle(img)

    if not load_local_model():
        return fallback_wrinkle_score(img)

    try:
        inputs = pipe_processor(images=Image.fromarray(img), return_tensors="pt")
        outputs = pipe_model(**inputs).logits
        probs = outputs.softmax(dim=1)[0].detach().numpy()

        id2label = pipe_model.config.id2label
        wrinkle_idx = None

        for k, v in id2label.items():
            if "wrinkle" in v.lower():
                wrinkle_idx = int(k)
                break

        if wrinkle_idx is None:
            wrinkle_idx = 1

        return float(np.clip(probs[wrinkle_idx], 0.0, 1.0))

    except Exception as e:
        print("⚠️ Wrinkle inference failed →", e)
        return fallback_wrinkle_score(img)


# ===================================================================================
# MULTI-ANGLE
# ===================================================================================

def score_wrinkles_multi(front_img, left_img, right_img):
    r_front = score_wrinkles_single(front_img)
    r_left  = score_wrinkles_single(left_img)
    r_right = score_wrinkles_single(right_img)

    final = 0.5*r_front + 0.25*r_left + 0.25*r_right
    return round(final, 4)


# ===================================================================================
# TEST
# ===================================================================================

if __name__ == "__main__":
    front = Image.open("front.jpg")
    left  = Image.open("left.jpg")
    right = Image.open("right.jpg")
    score = score_wrinkles_multi(front, left, right)
    print("Wrinkle Score:", score)
