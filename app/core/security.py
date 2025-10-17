from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGO = "HS256"

def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)

def verify_password(pw: str, hashed: str) -> bool:
    return pwd_context.verify(pw, hashed)

def create_access_token(sub: str, minutes: int | None = None) -> str:
    exp = datetime.utcnow() + timedelta(minutes=minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": sub, "exp": exp}
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGO)

def decode_token(token: str):
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGO])
    except JWTError:
        return None
