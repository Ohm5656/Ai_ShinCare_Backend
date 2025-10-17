import numpy as np
import cv2

def bytes_to_bgr(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def calc_brightness(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def classify_pose_from_yaw(yaw: float) -> str:
    # ปรับได้ตามต้องการ
    FRONT_THRESH = 10.0
    SIDE_THRESH  = 15.0

    if abs(yaw) < FRONT_THRESH:
        return "หน้าตรง"
    elif yaw > SIDE_THRESH:
        return "หันซ้าย"
    elif yaw < -SIDE_THRESH:
        return "หันขวา"
    else:
        return "ไม่แน่ใจ"
