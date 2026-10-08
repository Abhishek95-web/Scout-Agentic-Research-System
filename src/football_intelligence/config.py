from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Football Intelligence"
    environment: str = "development"

    database_path: Path = PROJECT_ROOT / "data" / "football_intelligence.db"

    groq_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None

    groq_model: str = "openai/gpt-oss-120b"
    tavily_max_results: int = Field(default=20, ge=1, le=100)
    request_timeout_seconds: int = Field(default=20, ge=1, le=120)


@lru_cache
def get_settings() -> Settings:
    return Settings()
