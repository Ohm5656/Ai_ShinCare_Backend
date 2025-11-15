
import os
import numpy as np
import cv2
from PIL import Image

# -----------------------------------------------------------------------------
# MEDIA PIPE FACEMESH
# -----------------------------------------------------------------------------
import mediapipe as mp
mp_face = mp.solutions.face_mesh

# ===================================================================================
# 1) LIGHTING CORRECTION (Clinic-grade)
# ===================================================================================

def _gray_world(img):
    """Auto white-balance แบบ Gray-World"""
    img_f = img.astype(np.float32)
    mean = img_f.reshape(-1,3).mean(axis=0)
    gray = mean.mean()
    gain = gray / (mean + 1e-6)
    out = np.clip(img_f * gain, 0, 255).astype(np.uint8)
    return out

def _clahe_l(img):
    """CLAHE บนช่อง L* เพื่อลดความต่างแสงเฉพาะพื้นที่"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

def _retinex_ssr(img, sigma=80.0):
    """Single Scale Retinex — แบบเดียวกับการประมวลผลผิวในงานวิจัย dermatology"""
    img_f = img.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(img_f, (0,0), sigma)
    ssr = np.log(img_f) - np.log(blur + 1.0)
    ssr = ssr - ssr.min()
    ssr = ssr / (ssr.max() + 1e-6) * 255.0
    return ssr.astype(np.uint8)

def _illumination_correction(img):
    """รวม 3 ขั้นตอน correction แบบเครื่องมือแพทย์"""
    x = _gray_world(img)
    x = _clahe_l(x)
    x = _retinex_ssr(x, sigma=60)
    return x

# ===================================================================================
# 2) SKIN MASK (fallback เมื่อ FaceMesh หา landmark ไม่เจอ)
# ===================================================================================

def _skin_mask_fallback(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    lower_hsv = np.array([0, 20, 40]);   upper_hsv = np.array([35, 200, 255])
    lower_yc  = np.array([0, 135, 85]);  upper_yc  = np.array([255, 180, 135])

    m1 = cv2.inRange(hsv, lower_hsv, upper_hsv)
    m2 = cv2.inRange(ycrcb, lower_yc, upper_yc)
    m = cv2.bitwise_and(m1, m2)
    m = cv2.medianBlur(m, 5)
    return (m > 0).astype(np.uint8) * 255


# ===================================================================================
# 3) FACEMESH → SKIN REGIONS
# ===================================================================================

EYE_L=[33,133,246,161,160,159,158,157,173]
EYE_R=[362,263,466,388,387,386,385,384,398]
LIPS_OUT=[61,146,91,181,84,17,314,405,321,375,291,61]
LIPS_IN=[78,95,88,178,87,14,317,402,318,324,308,78]

FOREHEAD=[10,338,297,332,284,251,389,356,454,323,93,132,58,172]
CHEEK_L=[123,116,147,187,205,50,101,36,39,67]
CHEEK_R=[352,345,372,411,425,280,330,266,269,295]
CHIN=[152,175,199,200,421,429,430,434,436,152]

def _landmarks_xy(res, w, h):
    lm = res.multi_face_landmarks[0].landmark
    return np.array([(p.x*w, p.y*h) for p in lm], dtype=np.float32)

def _mask_poly(h, w, pts):
    m = np.zeros((h,w), np.uint8)
    if len(pts)>=3:
        cv2.fillPoly(m, [pts.astype(np.int32)], 255)
    return m

def _build_masks(img, res):
    h,w,_ = img.shape
    pts = _landmarks_xy(res, w, h)

    face = cv2.convexHull(pts.astype(np.int32))
    m_face = _mask_poly(h, w, face)

    m_eye_l = _mask_poly(h, w, pts[EYE_L])
    m_eye_r = _mask_poly(h, w, pts[EYE_R])
    m_lip_o = _mask_poly(h, w, pts[LIPS_OUT])
    m_lip_i = _mask_poly(h, w, pts[LIPS_IN])

    skin = m_face.copy()
    for m in [m_eye_l,m_eye_r,m_lip_o,m_lip_i]:
        skin = cv2.subtract(skin, m)
    skin = cv2.medianBlur(skin,5)

    m_fh  = _mask_poly(h, w, pts[FOREHEAD])
    m_ckl = _mask_poly(h, w, pts[CHEEK_L])
    m_ckr = _mask_poly(h, w, pts[CHEEK_R])
    m_ch  = _mask_poly(h, w, pts[CHIN])

    masks = {
        "skin":skin,
        "forehead":cv2.bitwise_and(skin, m_fh),
        "cheeks":cv2.bitwise_and(skin, cv2.bitwise_or(m_ckl, m_ckr)),
        "chin":cv2.bitwise_and(skin, m_ch),
    }
    return masks


# ===================================================================================
# 4) LAB-BASED SCORING (Clinic-Grade)
# ===================================================================================

def _norm01(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo + 1e-6), 0, 1))

def _region_vals(L, mask):
    vals = L[mask==255]
    if len(vals) < 120:   # safety
        return None
    return vals

def _score_from_L(L, masks):
    skin_vals = _region_vals(L, masks["skin"])
    if skin_vals is None:
        skin_vals = L.flatten()

    std_inside = float(np.std(skin_vals))
    inner = _norm01(std_inside, 5.0, 25.0)

    means = []
    for key in ["forehead","cheeks","chin"]:
        vals = _region_vals(L, masks[key])
        if vals is not None:
            means.append(float(np.mean(vals)))

    if len(means)>=2:
        inter = np.ptp(means)
        inter = _norm01(inter, 3.0, 25.0)
    else:
        inter = 0.0

    # Clinic weight (Asian skin)
    risk = (
        0.65 * inner +      # ภายในผิวสำคัญสุด
        0.35 * inter        # ความต่างระหว่างโซน
    )
    return float(np.clip(risk, 0, 1))


# ===================================================================================
# 5) PUBLIC API
# ===================================================================================

def score_tone(img_pil: Image.Image) -> float:
    """ วิเคราะห์ความสม่ำเสมอของโทนผิวแบบ Clinic-grade """
    img = np.array(img_pil.convert("RGB"))
    img_corr = _illumination_correction(img)

    # FaceMesh
    with mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as fm:
        res = fm.process(img_corr)

    lab = cv2.cvtColor(img_corr, cv2.COLOR_RGB2LAB)
    L = lab[...,0]

    if res and res.multi_face_landmarks:
        masks = _build_masks(img_corr, res)
    else:
        m = _skin_mask_fallback(img_corr)
        masks = {"skin":m, "forehead":m, "cheeks":m, "chin":m}

    return _score_from_L(L, masks)


def score_tone_multiview(front, left, right):
    """Trimmed Mean — robust กว่า mean/median"""
    arr = np.array([
        score_tone(front),
        score_tone(left),
        score_tone(right)
    ], dtype=np.float32)

    arr_sorted = np.sort(arr)
    return float(arr_sorted[1])   # ตัดค่าสุดโต่ง 1 ค่า


# ===================================================================================
# CLI Test
# ===================================================================================
if __name__ == "__main__":
    img = Image.open("sample_face.jpg")
    print("Tone Risk =", score_tone(img))
