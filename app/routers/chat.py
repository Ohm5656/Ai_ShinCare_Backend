from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.deps import db_session, get_current_user
from app.db.models import Scan
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()

@router.post("/advise", response_model=ChatResponse)
def advise(data: ChatRequest, db: Session = Depends(db_session), user=Depends(get_current_user)):
    latest = db.query(Scan).filter(Scan.user_id==user.id).order_by(Scan.created_at.desc()).first()
    if not latest:
        return ChatResponse(
            reply="ยังไม่มีผลวิเคราะห์ล่าสุด ลองสแกนก่อนนะครับ",
            suggestions=["เริ่มสแกนใบหน้า", "ดูวิธีถ่ายให้แสงเหมาะสม"]
        )
    tips = []
    if latest.redness < 60: tips.append("ลองใช้ครีมลดรอยแดงที่มี Niacinamide")
    if latest.oiliness < 40: tips.append("บำรุงด้วยมอยส์เจอร์ไรเซอร์เพิ่มความชุ่มชื้น")
    if latest.acne < 60: tips.append("เลือกโฟมล้างหน้าสูตรอ่อนโยน ลดการอุดตัน")
    reply = "สรุปจากผลล่าสุด: " + (latest.summary or "ผิวโดยรวมปกติดี") + (" | " + " • ".join(tips) if tips else "")
    return ChatResponse(
        reply=reply,
        suggestions=["แนะนำผลิตภัณฑ์บำรุงผิว", "สาเหตุผิวแดง?", "ลดสิวอย่างไร?"]
    )
