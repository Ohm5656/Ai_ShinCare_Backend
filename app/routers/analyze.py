from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
from insightface.app import FaceAnalysis
import cv2, numpy as np
from app.services.face_pose import infer_pose_from_image, classify_pose

router = APIRouter(tags=["Analyze"])


# ===================================================
# โหลดโมเดล InsightFace สำหรับตรวจจับใบหน้า
# ===================================================
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))


# ===================================================
# Helper: วิเคราะห์คุณภาพผิวจากใบหน้าที่ครอปมา
# ===================================================
def analyze_skin(face_crop):
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    wrinkle = np.clip(np.std(cv2.Laplacian(gray, cv2.CV_64F)) / 80, 0, 1)

    hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
    redness = np.clip((np.mean(face_crop[:, :, 2]) - 110) / 80, 0, 1)

    lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
    tone = np.clip((np.std(lab[:, :, 1]) + np.std(lab[:, :, 2])) / 40, 0, 1)

    oil = np.clip(np.mean(gray >= np.percentile(gray, 95)) * 3, 0, 1)

    h, w = face_crop.shape[:2]
    roi = face_crop[int(h * 0.25):int(h * 0.4), int(w * 0.2):int(w * 0.8)]
    eye_bag = np.clip(1 - np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)) / 255, 0, 1)

    acne = np.clip(redness * 0.8 + oil * 0.2, 0, 1)

    return {
        "ริ้วรอย": wrinkle,
        "รอยแดง": redness,
        "สีผิว": tone,
        "ความมัน": oil,
        "ใต้ตา": eye_bag,
        "สิว": acne,
    }


# ===================================================
# ✅ วิเคราะห์ “มุมใบหน้า” จากรูปเดียว (ใช้ใน Loop)
# ===================================================
@router.post("/pose")
async def analyze_pose(file: UploadFile = File(...)):
    try:
        # -----------------------------
        # โหลดภาพจาก frontend
        # -----------------------------
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="ไม่สามารถอ่านภาพได้")

        # -----------------------------
        # ตรวจจับใบหน้า
        # -----------------------------
        faces = face_app.get(img)
        face_ok = len(faces) > 0

        # -----------------------------
        # ตรวจวัดความสว่าง
        # -----------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))  # ✅ cast เป็น float ปกติ
        light_ok = bool(brightness > 60)   # ✅ cast เป็น bool ปกติ

        # -----------------------------
        # ถ้าไม่พบใบหน้า
        # -----------------------------
        if not face_ok:
            print(f"[DEBUG] ❌ No face detected | brightness={brightness:.1f}")
            return {
                "pose": "none",
                "face_ok": False,
                "light_ok": light_ok,
                "brightness": brightness
            }

        # -----------------------------
        # ถ้ามีใบหน้า → วิเคราะห์มุม
        # -----------------------------
        face = max(faces, key=lambda x: x.det_score)
        x1, y1, x2, y2 = map(int, face.bbox)
        face_crop = img[y1:y2, x1:x2]

        pose_data = infer_pose_from_image(face_crop)
        pose_label = str(classify_pose(float(pose_data["yaw"]), float(pose_data["pitch"])))  # ✅ force เป็น string

        print(f"[DEBUG] ✅ Pose={pose_label}, yaw={pose_data['yaw']:.2f}, pitch={pose_data['pitch']:.2f}, "
              f"brightness={brightness:.1f}, faces={len(faces)}")

        # -----------------------------
        # ส่งผลกลับไป frontend
        # -----------------------------
        return {
            "pose": pose_label,
            "face_ok": True,
            "light_ok": light_ok,
            "yaw": float(pose_data["yaw"]),
            "pitch": float(pose_data["pitch"]),
            "brightness": brightness
        }

    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()

# ===================================================
# ✅ วิเคราะห์ “ผิวหน้า” จากภาพทั้ง 3 มุม
# ===================================================
@router.post("/skin")
async def analyze_skin_api(files: List[UploadFile] = File(...)):
    try:
        if len(files) < 3:
            raise HTTPException(status_code=400, detail="ต้องส่งภาพ 3 มุม (front, left, right)")

        skin_scores = []

        for file in files:
            contents = await file.read()
            img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            faces = face_app.get(img)
            if not faces:
                continue

            face = max(faces, key=lambda x: x.det_score)
            x1, y1, x2, y2 = map(int, face.bbox)
            face_crop = img[y1:y2, x1:x2]
            skin_scores.append(analyze_skin(face_crop))

        # รวมค่าเฉลี่ย
        avg = {k: float(np.mean([d[k] for d in skin_scores])) for k in skin_scores[0]}

        return {"ok": True, "results": avg, "message": "วิเคราะห์ผิวสำเร็จ"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"การวิเคราะห์ล้มเหลว: {e}")
