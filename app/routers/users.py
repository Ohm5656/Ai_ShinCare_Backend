from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.deps import db_session, get_current_user
from app.db.models import Profile
from app.schemas.user import ProfileOut, ProfileUpdate

router = APIRouter()

@router.get("/me", response_model=ProfileOut)
def me(db: Session = Depends(db_session), user=Depends(get_current_user)):
    prof = db.query(Profile).filter(Profile.user_id==user.id).first()
    return prof

@router.put("/me", response_model=ProfileOut)
def update_me(data: ProfileUpdate, db: Session = Depends(db_session), user=Depends(get_current_user)):
    prof = db.query(Profile).filter(Profile.user_id==user.id).first()
    if not prof:
        prof = Profile(user_id=user.id)
        db.add(prof)
    for k,v in data.model_dump(exclude_unset=True).items():
        setattr(prof, k, v)
    db.commit()
    db.refresh(prof)
    return prof
