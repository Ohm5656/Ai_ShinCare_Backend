from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import SessionLocal
from models.scan_model import ScanHistory

router = APIRouter(prefix="/scan", tags=["scan"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/save")
def save_scan(data: dict, db: Session = Depends(get_db)):
    scan = ScanHistory(
        overall_score = data["overall_score"],
        wrinkles = data["dimension_scores"]["wrinkles"],
        sagging = data["dimension_scores"]["sagging"],
        pigmentation = data["dimension_scores"]["pigmentation"],
        acne = data["dimension_scores"]["acne"],
        redness = data["dimension_scores"]["redness"],
        texture = data["dimension_scores"]["texture"],
        tone = data["dimension_scores"]["tone"],
        top_issue = data.get("top_issue", ""),
        improvement = data.get("improvement", 0),
        profile = data["profile"],
    )

    db.add(scan)
    db.commit()
    db.refresh(scan)

    return {"status": "ok", "scan_id": scan.id}
