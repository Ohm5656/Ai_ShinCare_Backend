from fastapi import APIRouter, File, UploadFile, HTTPException
from insightface.app import FaceAnalysis
import cv2, numpy as np, math, os

# 🔹 Import service 3DDFA_V2 ที่โอห์มสร้างไว้
from app.services.face_pose import infer_pose_from_image, classify_pose

router = APIRouter(prefix="/analyze", tags=["Analyze"])

# ======================================
# โหลดโมเดล InsightFace (สำหรับตรวจจับใบหน้า)
# ======================================
app_insight = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app_insight.prepare(ctx_id=0, det_size=(640, 640))


# ======================================
# ฟังก์ชันย่อย (วิเคราะห์ผิว)
# ======================================
def analyze_skin(face_crop):
    """วิเคราะห์พารามิเตอร์ผิวหน้า 6 ด้าน"""
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


def describe_skin(name, score):
    """ตีความค่าตัวเลขเป็นข้อความภาษาไทย"""
    s = score * 100
    if name == "สิว":
        if s < 30:
            return "ผิวสะอาด ไม่มีสิว"
        elif s < 60:
            return "มีสิวเล็กน้อย"
        else:
            return "มีสิวชัดเจน ควรลดความมัน"
    elif name == "ริ้วรอย":
        if s < 30:
            return "ผิวเรียบเนียน"
        elif s < 60:
            return "ริ้วรอยเล็กน้อย"
        else:
            return "ริ้วรอยชัด ควรบำรุง"
    elif name == "ความมัน":
        if s < 30:
            return "ผิวสมดุล"
        elif s < 60:
            return "ค่อนข้างมัน"
        else:
            return "ผิวมันมาก"
    elif name == "รอยแดง":
        if s < 30:
            return "ผิวปกติ"
        elif s < 60:
            return "รอยแดงเล็กน้อย"
        else:
            return "รอยแดงชัด"
    elif name == "สีผิว":
        if s < 30:
            return "สีผิวสม่ำเสมอ"
        elif s < 60:
            return "สีผิวเริ่มไม่สม่ำเสมอ"
        else:
            return "สีผิวหมองคล้ำ"
    elif name == "ใต้ตา":
        if s < 30:
            return "ใต้ตาสดใส"
        elif s < 60:
            return "คล้ำเล็กน้อย"
        else:
            return "คล้ำมาก"
    return "ปกติ"


# ======================================
# Endpoint วิเคราะห์ผิว + มุมใบหน้า
# ======================================
@router.post("/skin")
async def analyze_skin_api(file: UploadFile = File(...)):
    """
    📷 รับภาพใบหน้า → ตรวจจับใบหน้า → วิเคราะห์ผิว + มุมใบหน้า (3DDFA_V2)
    """
    try:
        # อ่านข้อมูลภาพ
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="ไม่สามารถอ่านภาพได้")

        # ----------------------------------------
        # 🔹 ตรวจจับใบหน้าด้วย InsightFace
        # ----------------------------------------
        faces = app_insight.get(img)
        if not faces:
            return {"ok": False, "message": "ไม่พบใบหน้าในภาพ"}

        # ใช้ใบหน้าที่มั่นใจที่สุด
        face = max(faces, key=lambda x: x.det_score)
        x1, y1, x2, y2 = map(int, face.bbox)
        face_crop = img[y1:y2, x1:x2]

        # ----------------------------------------
        # 🔹 วิเคราะห์ผิว (6 พารามิเตอร์)
        # ----------------------------------------
        results = analyze_skin(face_crop)
        descriptions = {k: describe_skin(k, v) for k, v in results.items()}

        # ----------------------------------------
        # 🔹 วิเคราะห์มุมใบหน้าด้วย 3DDFA_V2 (ONNX)
        # ----------------------------------------
        pose_data = infer_pose_from_image(face_crop)
        pose_label = pose_data.get("label", "unknown")

        # รวมผลลัพธ์ทั้งหมด
        summary_text = (
            "สรุปผลเบื้องต้น: ผิวโดยรวมอยู่ในเกณฑ์ดี "
            "ควรรักษาความชุ่มชื้นและทาครีมกันแดดสม่ำเสมอ"
        )

        return {
            "ok": True,
            "pose": {
                "yaw": pose_data["yaw"],
                "pitch": pose_data["pitch"],
                "roll": pose_data["roll"],
                "label": pose_label
            },
            "results": results,
            "descriptions": descriptions,
            "summary": summary_text,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"การวิเคราะห์ล้มเหลว: {e}")
    finally:
        await file.close()
