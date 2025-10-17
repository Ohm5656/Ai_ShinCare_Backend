from pydantic import BaseModel
from typing import Optional

class ProfileOut(BaseModel):
    display_name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    skin_type: Optional[str] = None
    goal: Optional[str] = None
    avatar_url: Optional[str] = None
    class Config:
        from_attributes = True

class ProfileUpdate(ProfileOut):
    pass
