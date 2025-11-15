import numpy as np
import cv2
from PIL import Image
import mediapipe as mp

mp_face = mp.solutions.face_mesh

# -------- FACEMESH POINTS USED --------
UNDER_EYE_L = 145
UNDER_EYE_R = 374
MOUTH_CORNER_L = 61
MOUTH_CORNER_R = 291
JAW_L = 172
JAW_R = 397
CHEEK_L = 234
CHEEK_R = 454
CHIN = 152
UNDER_CHIN = 200
FOREHEAD = 10
NASION = 6

# ==============================================================
# 🔥 PREPROCESSING BEFORE FACEMESH (เพิ่มความเสถียร)
# ==============================================================

def _illumination_fix(img):
    """ปรับแสงสำหรับ FaceMesh: SSR + CLAHE + sharpen"""
    # --- 1) Retinex SSR (ลดเงา / แก้หน้ามืด) ---
    img_f = img.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(img_f, (0,0), 60)
    ssr = np.log(img_f) - np.log(blur + 1.0)
    ssr = ssr - ssr.min()
    ssr = (255 * ssr / (ssr.max() + 1e-6)).astype(np.uint8)

    # --- 2) CLAHE (เพิ่ม contrast เน้นขอบหน้า) ---
    lab = cv2.cvtColor(ssr, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)
    img2 = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB)

    # --- 3) sharpen (ทำให้ landmark มองเห็นง่ายขึ้น) ---
    blur2 = cv2.GaussianBlur(img2, (0,0), 3)
    sharp = cv2.addWeighted(img2, 1.6, blur2, -0.6, 0)

    return sharp

def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def _landmarks(img_rgb):
    """เรียก FaceMesh โดยใช้ภาพที่ปรับแสงแล้ว"""
    img_norm = _illumination_fix(img_rgb)

    with mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as fm:
        res = fm.process(img_norm)

    if not res.multi_face_landmarks:
        return None

    h, w, _ = img_rgb.shape
    lm = res.multi_face_landmarks[0].landmark
    pts = [(p.x*w, p.y*h) for p in lm]
    return pts

# ==============================================================
# ⭐ MAIN SAGGING INDEX
# ==============================================================

def _sagging_index(img_rgb):
    pts = _landmarks(img_rgb)
    if pts is None:
        return None

    # 1) normalize โดย face height
    fore = pts[FOREHEAD]
    chin = pts[CHIN]
    face_h = _dist(fore, chin) + 1e-6

    # 2) mid-face sag
    sag_L = max(0, pts[MOUTH_CORNER_L][1] - pts[UNDER_EYE_L][1]) / face_h
    sag_R = max(0, pts[MOUTH_CORNER_R][1] - pts[UNDER_EYE_R][1]) / face_h
    mid_sag = (sag_L + sag_R) / 2

    # 3) lower face sag
    mouth_mid = (
        (pts[MOUTH_CORNER_L][0] + pts[MOUTH_CORNER_R][0]) / 2,
        (pts[MOUTH_CORNER_L][1] + pts[MOUTH_CORNER_R][1]) / 2,
    )
    lower_sag = _dist(mouth_mid, chin) / face_h

    # 4) jowl droop (แก้มย้อย)
    jowl_L = max(0, pts[JAW_L][1] - pts[CHEEK_L][1]) / face_h
    jowl_R = max(0, pts[JAW_R][1] - pts[CHEEK_R][1]) / face_h
    jowl = (jowl_L + jowl_R) / 2

    # 5) chin-throat sag
    throat = pts[UNDER_CHIN]
    chin_th = max(0, throat[1] - chin[1]) / face_h

    # ⭐ Fusion ตามงานวิจัย
    idx = (
        0.40 * mid_sag +
        0.25 * jowl +
        0.20 * lower_sag +
        0.15 * chin_th
    )

    return float(idx)

# ==============================================================
# PUBLIC API
# ==============================================================

def score_sagging(img_front: Image.Image, img_left=None, img_right=None) -> float:
    img = np.array(img_front.convert("RGB"))
    idx = _sagging_index(img)
    if idx is None:
        return 0.5  # neutral fallback

    # map 0.20–0.45 → 0..1
    risk = np.clip((idx - 0.20) / 0.25, 0, 1)
    return float(risk)
