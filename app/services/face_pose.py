from __future__ import annotations
import os
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from insightface.app import FaceAnalysis

# ============================================
# CONFIG
# ============================================
MODEL_PATH = os.getenv("FACE_POSE_ONNX", "models/mb1_120x120.onnx")
CSV_FILE = "pose_calibration.csv"

# โหลด InsightFace สำหรับตรวจจับใบหน้า
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640), providers=["CPUExecutionProvider"])


# ============================================
# โหลดโมเดล ONNX
# ============================================
_session: ort.InferenceSession | None = None
def _get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ ไม่พบโมเดล ONNX: {MODEL_PATH}")
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        print(f"✅ โหลดโมเดล ONNX สำเร็จ: {MODEL_PATH}")
    return _session


# ============================================
# อ่านและเตรียมภาพก่อนเข้าโมเดล
# ============================================
def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, (120, 120))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


# ============================================
# โหลดค่า yaw/pitch จาก CSV (เฉลี่ยแต่ละมุม)
# ============================================
def load_thresholds(csv_path=CSV_FILE):
    df = pd.read_csv(csv_path, header=None)

    if len(df.columns) == 6:
        df.columns = ["timestamp", "pose_class", "yaw", "pitch", "roll", "predicted"]
    elif len(df.columns) == 5:
        df.columns = ["timestamp", "pose_class", "yaw", "pitch", "roll"]
    else:
        raise ValueError("❌ รูปแบบ CSV ไม่ถูกต้อง")

    df["pose_class"] = df["pose_class"].astype(str).str.strip().str.lower()
    thresholds = {}
    for name in ["front", "left", "right", "up", "down"]:
        subset = df[df["pose_class"] == name]
        if len(subset) > 0:
            thresholds[name] = {
                "yaw": subset["yaw"].mean(),
                "pitch": subset["pitch"].mean(),
            }
    print("✅ โหลดค่าเกณฑ์เฉลี่ยจาก CSV สำเร็จ:")
    for k, v in thresholds.items():
        print(f"  {k:>6}: yaw={v['yaw']:.2f}, pitch={v['pitch']:.2f}")
    return thresholds


# โหลด thresholds ทันทีตอนเริ่มต้น
thresholds = load_thresholds()


# ============================================
# วิเคราะห์มุมศีรษะจากใบหน้า
# ============================================
def infer_pose_from_image(img_bgr: np.ndarray) -> dict | None:
    session = _get_session()
    inp = _preprocess(img_bgr)
    input_name = session.get_inputs()[0].name

    try:
        outputs = session.run(None, {input_name: inp})
    except Exception as e:
        print(f"❌ ONNX inference ล้มเหลว: {e}")
        return None

    params = outputs[0][0]
    yaw, pitch, roll = float(params[0]), float(params[1]), float(params[2])
    return {"yaw": yaw, "pitch": pitch, "roll": roll}


# ============================================
# คำนวณมุมจาก yaw/pitch เทียบกับ CSV
# ============================================
def classify_pose(yaw, pitch, ref=thresholds):
    min_dist = 9999
    best_label = "unknown"
    for name, r in ref.items():
        dist = np.sqrt((yaw - r["yaw"]) ** 2 + (pitch - r["pitch"]) ** 2)
        if dist < min_dist:
            min_dist = dist
            best_label = name
    return best_label, min_dist


# ============================================
# ฟังก์ชันหลักที่ใช้ใน endpoint
# ============================================
def detect_pose_from_bytes(image_bytes: bytes) -> dict:
    """
    วิเคราะห์มุมศีรษะจากภาพที่ frontend ส่งมา (bytes)
    - ตรวจจับใบหน้าด้วย InsightFace
    - ครอปใบหน้า
    - วิเคราะห์มุมด้วย ONNX
    - เทียบกับ CSV เพื่อบอกมุม front/left/right/up/down
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    faces = face_app.get(frame)
    if not faces:
        return {"pose": "none", "message": "ไม่พบใบหน้าในภาพ"}

    face = max(faces, key=lambda f: f.det_score)
    x1, y1, x2, y2 = map(int, face.bbox)
    face_crop = frame[y1:y2, x1:x2]

    pose = infer_pose_from_image(face_crop)
    if not pose:
        return {"pose": "unknown", "message": "ไม่สามารถวิเคราะห์มุมได้"}

    yaw, pitch = pose["yaw"], pose["pitch"]
    label, dist = classify_pose(yaw, pitch, thresholds)
    return {
        "pose": label,
        "yaw": yaw,
        "pitch": pitch,
        "roll": pose["roll"],
        "distance": round(dist, 2),
    }
