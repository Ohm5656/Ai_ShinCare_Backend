from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from database import Base

class ScanHistory(Base):
    __tablename__ = "scan_history"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime(timezone=True), server_default=func.now())

    overall_score = Column(Float)
    wrinkles = Column(Float)
    sagging = Column(Float)
    pigmentation = Column(Float)
    acne = Column(Float)
    redness = Column(Float)
    texture = Column(Float)
    tone = Column(Float)

    top_issue = Column(String)
    improvement = Column(Float, default=0)

    profile = Column(JSON)     # sex, age_range, skin_type, sensitive, concerns

    # Save before/after links if needed later
    images = Column(JSON, default={})
