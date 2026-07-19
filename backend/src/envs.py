from pathlib import Path
from typing import Self

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application settings
    app_name: str = "DIANA prototype"
    deployment: str = Field(default="local", validation_alias=AliasChoices("DIANA_DEPLOYMENT", "VERCEL_ENV"))
    secure_cookies: bool = False
    session_max_age: int = 28800
    frontend_dist: Path = Path(__file__).resolve().parents[2] / "frontend" / "dist"

    # Prototype account settings
    session_secret: SecretStr = SecretStr("local-only-change-this-session-secret")
    scientist_username: str = "scientist"
    scientist_password: SecretStr = SecretStr("diana-scientist")
    participant_username: str = "participant"
    participant_password: SecretStr = SecretStr("diana-participant")

    model_config = SettingsConfigDict(env_prefix="DIANA_", env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def validate_deployed_credentials(self) -> Self:
        """Reject documented local credentials in public Vercel environments."""

        # Prevent a preview or production deployment from accepting known demo secrets.
        if self.deployment in {"preview", "production"}:
            defaults_present = (
                self.session_secret.get_secret_value() == "local-only-change-this-session-secret"
                or self.participant_password.get_secret_value() == "diana-participant"
                or self.scientist_password.get_secret_value() == "diana-scientist"
            )
            if defaults_present:
                raise ValueError("Set DIANA session and account secrets before deploying to Vercel")
        return self


env = Settings()
