from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import SessionLocal
from models.scan_model import ScanHistory

router = APIRouter(prefix="/history", tags=["history"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 1) Summary
@router.get("/summary")
def get_summary(db: Session = Depends(get_db)):
    scans = db.query(ScanHistory).order_by(ScanHistory.date).all()

    if not scans:
        return {
            "totalScans": 0,
            "averageScore": 0,
            "latestScore": 0,
            "improvement": 0
        }

    total = len(scans)
    avg = sum(s.overall_score for s in scans) / total
    latest = scans[-1].overall_score
    improvement = latest - scans[0].overall_score

    return {
        "totalScans": total,
        "averageScore": round(avg, 2),
        "latestScore": latest,
        "improvement": round(improvement, 2)
    }


# 2) Chart Data
@router.get("/scores")
def get_scores(range: str, db: Session = Depends(get_db)):
    scans = db.query(ScanHistory).order_by(ScanHistory.date).all()

    result = [
        {
            "date": s.date.strftime("%Y-%m-%d"),
            "overall": s.overall_score,
            "wrinkles": s.wrinkles,
            "sagging": s.sagging,
            "darkSpots": s.pigmentation,
            "acne": s.acne,
            "redness": s.redness,
            "pores": s.texture,
            "evenness": s.tone
        }
        for s in scans
    ]
    return result


# 3) Timeline
@router.get("/scans")
def get_scans(db: Session = Depends(get_db)):
    scans = db.query(ScanHistory).order_by(ScanHistory.date.desc()).limit(20).all()

    data = []
    for s in scans:
        data.append({
            "id": s.id,
            "date": s.date.strftime("%Y-%m-%d"),
            "score": s.overall_score,
            "topIssue": s.top_issue or "Overall skin condition",
            "improvement": s.improvement,
            "thumbnail": "📷"
        })
    return data


# 4) Before/After Progress
@router.get("/progress")
def get_progress(db: Session = Depends(get_db)):
    scans = db.query(ScanHistory).order_by(ScanHistory.date).all()
    if len(scans) < 2:
        return []

    first = scans[0]
    last = scans[-1]
    diff = last.overall_score - first.overall_score

    return [{
        "id": 1,
        "start": first.date.strftime("%Y-%m-%d"),
        "end": last.date.strftime("%Y-%m-%d"),
        "improvement": round(diff, 2),
        "emoji": "🌈"
    }]
