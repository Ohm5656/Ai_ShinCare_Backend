from pydantic import BaseModel
from typing import Dict, List

class CheckResponse(BaseModel):
    face_ok: bool
    lighting_ok: bool
    brightness: float

class AnalyzeResponse(BaseModel):
    scan_id: str
    score_total: int
    radar: Dict[str, int]
    skin_type: str
    highlights: List[str]
    improvements: List[str]
    summary: str
    lighting_ok: bool
    face_ok: bool
