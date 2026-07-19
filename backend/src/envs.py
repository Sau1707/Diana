from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


SESSION_SECRET_FILE = Path(__file__).resolve().parents[1] / ".session-secret"

# Prefer the private secret generated inside the production image when no environment secret is set.
DEFAULT_SESSION_SECRET = (
    SESSION_SECRET_FILE.read_text(encoding="utf-8").strip()
    if SESSION_SECRET_FILE.is_file()
    else "local-only-change-this-session-secret"
)


class Settings(BaseSettings):
    # Application settings
    app_name: str = "DIANA prototype"
    deployment: str = Field(default="local", validation_alias=AliasChoices("DIANA_DEPLOYMENT", "VERCEL_ENV"))
    secure_cookies: bool = False
    session_max_age: int = 28800
    frontend_dist: Path = Path(__file__).resolve().parents[2] / "frontend" / "dist"

    # Session settings
    session_secret: SecretStr = SecretStr(DEFAULT_SESSION_SECRET)

    model_config = SettingsConfigDict(env_prefix="DIANA_", env_file=".env", extra="ignore")


env = Settings()
