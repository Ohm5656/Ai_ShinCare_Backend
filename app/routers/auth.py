from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import uuid4

from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse
from app.core.security import hash_password, verify_password, create_access_token
from app.core.deps import db_session
from app.db.models import User, Profile

router = APIRouter()

@router.post("/register", response_model=TokenResponse)
def register(data: RegisterRequest, db: Session = Depends(db_session)):
    if db.query(User).filter(User.email==data.email).first():
        raise HTTPException(400, "Email already registered")
    user = User(id=str(uuid4()), email=data.email, password_hash=hash_password(data.password))
    db.add(user)
    db.add(Profile(user_id=user.id))
    db.commit()
    token = create_access_token(str(user.id))
    return TokenResponse(access_token=token)

@router.post("/login", response_model=TokenResponse)
def login(data: LoginRequest, db: Session = Depends(db_session)):
    user = db.query(User).filter(User.email==data.email).first()
    if not user or not user.password_hash or not verify_password(data.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token(str(user.id))
    return TokenResponse(access_token=token)
