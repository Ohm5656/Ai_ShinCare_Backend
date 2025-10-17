from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    question: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    suggestions: List[str]
