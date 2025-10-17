from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app.core.deps import db_session, get_current_user
from app.db.models import Scan
from app.services.smoother import ema

router = APIRouter()

@router.get("/summary")
def summary(period: str = "7d", db: Session = Depends(db_session), user=Depends(get_current_user)):
    days = {"7d":7, "15d":15, "30d":30}.get(period, 7)
    since = datetime.utcnow() - timedelta(days=days)
    rows = db.query(Scan).filter(Scan.user_id==user.id, Scan.created_at>=since).order_by(Scan.created_at.asc()).all()
    scores = [r.score_total for r in rows]
    if not scores:
        return {"avg": 0, "delta": 0, "best_dimension": None, "count": 0}

    smoothed = ema(scores, 0.4)
    avg = int(sum(scores)/len(scores))
    delta = int(scores[-1] - scores[0])
    dims = ["smoothness","redness","tone","oiliness","eyebag","acne"]
    med = {d:int(sum(getattr(r,d) for r in rows)/len(rows)) for d in dims}
    best_dim = max(med, key=med.get)
    return {"avg": avg, "delta": delta, "best_dimension": best_dim, "count": len(rows)}

@router.get("/list")
def list_scans(page: int = 1, page_size: int = 10, db: Session = Depends(db_session), user=Depends(get_current_user)):
    q = db.query(Scan).filter(Scan.user_id==user.id).order_by(Scan.created_at.desc())
    total = q.count()
    items = q.offset((page-1)*page_size).limit(page_size).all()
    data = [{
        "id": str(r.id), "created_at": r.created_at.isoformat(), "score_total": r.score_total,
        "summary": r.summary
    } for r in items]
    return {"total": total, "page": page, "page_size": page_size, "items": data}
