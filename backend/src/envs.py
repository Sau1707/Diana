from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application settings
    app_name: str = "DIANA prototype"
    frontend_dist: Path = Path(__file__).resolve().parents[2] / "frontend" / "dist"

    model_config = SettingsConfigDict(env_prefix="DIANA_", env_file=".env", extra="ignore")


env = Settings()
