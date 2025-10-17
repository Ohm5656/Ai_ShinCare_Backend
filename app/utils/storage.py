import os, shutil
from uuid import uuid4
from app.core.config import settings

os.makedirs(settings.STORAGE_DIR, exist_ok=True)

def save_upload(file, subdir="raw", prefix=None) -> str:
    folder = os.path.join(settings.STORAGE_DIR, subdir)
    os.makedirs(folder, exist_ok=True)
    name = f"{prefix or uuid4()}_{file.filename}"
    path = os.path.join(folder, name)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return path
