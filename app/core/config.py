from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_ENV: str = "development"
    SECRET_KEY: str = "change_me"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    CORS_ORIGINS: str = "*"  # ใส่โดเมน frontend จริงภายหลัง

    # ทดสอบในเครื่องด้วย SQLite (ง่ายสุด) | ผลิตจริงใช้ PostgreSQL
    DATABASE_URL: str = "sqlite:///./dev.db"
    # ตัวอย่าง PostgreSQL: postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME

    STORAGE_DIR: str = "./uploads"

    INSIGHTFACE_PROVIDER: str = "CPUExecutionProvider"
    DETECT_SIZE: int = 640

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
