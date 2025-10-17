from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from uuid import uuid4
from datetime import datetime

from app.db.base import Base

# helper: UUID column ที่ใช้ได้กับ SQLite/PG (เก็บเป็น TEXT ใน SQLite)
def UUID_col():
    try:
        return Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    except Exception:
        return Column(String, primary_key=True, default=lambda: str(uuid4()))

def UUID_fk():
    try:
        return Column(UUID(as_uuid=True), ForeignKey("users.id"))
    except Exception:
        return Column(String, ForeignKey("users.id"))

class User(Base):
    __tablename__ = "users"
    id = UUID_col()
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=True)
    provider = Column(String, default="local")
    is_premium = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    profile = relationship("Profile", back_populates="user", uselist=False, cascade="all,delete")
    scans = relationship("Scan", back_populates="user", cascade="all,delete")

class Profile(Base):
    __tablename__ = "profiles"
    user_id = UUID_fk()
    display_name = Column(String)
    gender = Column(String)  # male/female/other
    age = Column(Integer)
    skin_type = Column(String)  # dry/oily/combination/sensitive
    goal = Column(Text)
    avatar_url = Column(Text)

    user = relationship("User", back_populates="profile")
    __table_args__ = ({'sqlite_autoincrement': True},)

class Scan(Base):
    __tablename__ = "scans"
    id = UUID_col()
    user_id = UUID_fk()
    created_at = Column(DateTime, default=datetime.utcnow)
    lighting_ok = Column(Boolean)
    face_ok = Column(Boolean)
    angle = Column(String)
    source_image = Column(Text)
    result_image = Column(Text)
    score_total = Column(Integer)
    smoothness = Column(Integer)
    redness = Column(Integer)
    tone = Column(Integer)
    oiliness = Column(Integer)
    eyebag = Column(Integer)
    acne = Column(Integer)
    summary = Column(Text)

    user = relationship("User", back_populates="scans")
