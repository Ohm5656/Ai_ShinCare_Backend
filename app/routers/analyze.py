from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import uuid4

from app.core.deps import db_session, get_current_user
from app.utils.storage import save_upload
from app.services.analyzer import quick_check, analyze_full
from app.services.scoring import combine_score, skin_type_from, summarize
from app.db.models import Scan

router = APIRouter()

@router.post("/check")
async def check(file: UploadFile = File(...)):
    path = save_upload(file, subdir="checks", prefix=str(uuid4()))
    return quick_check(path)

@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    angle: str = "auto",
    db: Session = Depends(db_session),
    user=Depends(get_current_user)
):
    path = save_upload(file, subdir="raw", prefix=str(uuid4()))
    radar, qc = analyze_full(path, angle=angle)
    if radar is None:
        raise HTTPException(status_code=400, detail="ไม่พบใบหน้าในภาพ")

    total = combine_score(radar)
    skin_type = skin_type_from(radar)
    highlights, improvements, summary = summarize(radar, total)

    scan = Scan(
        user_id=user.id, lighting_ok=qc["lighting_ok"], face_ok=qc["face_ok"],
        angle=angle, source_image=path, score_total=total,
        smoothness=radar["smoothness"], redness=radar["redness"], tone=radar["tone"],
        oiliness=radar["oiliness"], eyebag=radar["eyebag"], acne=radar["acne"],
        summary=summary
    )
    db.add(scan); db.commit(); db.refresh(scan)

    return {
        "scan_id": str(scan.id),
        "score_total": total,
        "radar": radar,
        "skin_type": skin_type,
        "highlights": highlights,
        "improvements": improvements,
        "summary": summary,
        **qc
    }
