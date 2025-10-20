import cv2, numpy as np
from insightface.app import FaceAnalysis
from app.core.config import settings

_face = FaceAnalysis(name='buffalo_l', providers=[settings.INSIGHTFACE_PROVIDER])
_face.prepare(ctx_id=0, det_size=(settings.DETECT_SIZE, settings.DETECT_SIZE))

# ====== Utils ======
def _brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(hsv[...,2].mean())

def quick_check(path: str):
    img = cv2.imread(path)
    faces = _face.get(img)
    b = _brightness(img)
    return {"face_ok": len(faces) > 0, "lighting_ok": 110 <= b <= 180, "brightness": round(b, 2)}

# ====== Score functions ======
def _score_smoothness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return int(np.clip(100 - (var/5), 0, 100))

def _score_redness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,_ = cv2.split(hsv)
    red_mask = (h<10)|(h>170)
    red = np.mean(s[red_mask]) if np.any(red_mask) else 0
    return int(np.clip(100 - red*0.5, 0, 100))

def _score_tone(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sd = gray.std()
    return int(np.clip(100 - (sd/2), 0, 100))

def _score_oiliness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    spec = (hsv[...,2] > 230).mean() * 100
    score = 100 - abs(spec - 15) * 3
    return int(np.clip(score, 0, 100))

def _score_eyebag(img):
    h,w,_ = img.shape
    roi = img[int(h*0.55):int(h*0.75), int(w*0.25):int(w*0.75)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    darkness = 255 - gray.mean()
    return int(np.clip(100 - darkness*0.5, 0, 100))

def _score_acne(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a = lab[...,1]
    blem = (a>155).mean()*100
    return int(np.clip(100 - blem*2, 0, 100))


# ====== Core analyze ======
def analyze_full(path: str, angle="auto"):
    img = cv2.imread(path)
    qc = quick_check(path)
    if not qc["face_ok"]:
        return None, qc

    radar = {
        "smoothness": _score_smoothness(img),
        "redness": _score_redness(img),
        "tone": _score_tone(img),
        "oiliness": _score_oiliness(img),
        "eyebag": _score_eyebag(img),
        "acne": _score_acne(img),
    }
    return radar, qc


# ====== New: Analyze from 3 angles ======
def analyze_faces_from_files(files: list[bytes]):
    results = []
    for img_bytes in files:
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        radar = {
            "smoothness": _score_smoothness(frame),
            "redness": _score_redness(frame),
            "tone": _score_tone(frame),
            "oiliness": _score_oiliness(frame),
            "eyebag": _score_eyebag(frame),
            "acne": _score_acne(frame),
        }
        results.append(radar)

    # รวมค่าเฉลี่ย 3 มุม
    avg = {k: int(np.mean([r[k] for r in results])) for k in results[0]}
    return {"average": avg, "details": results}
