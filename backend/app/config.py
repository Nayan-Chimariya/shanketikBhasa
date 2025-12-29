from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Sign Language Learning API"
    DEBUG: bool = True

    # Database (MySQL)
    DATABASE_URL: str = "mysql+pymysql://root:password@localhost:{port}/{db_name}"

    # JWT Authentication
    SECRET_KEY: str = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    ALGORITHM: str = "XXXXX"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS
    ALLOWED_ORIGINS: list = ["http://localhost:XXXX", "http://localhost:XXXX"]

    # ML Model
    MODEL_PATH: str = "sign_language_model.keras"

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True
    )


settings = Settings()
