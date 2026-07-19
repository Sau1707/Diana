from typing import Literal

from pydantic import BaseModel, Field


AuthRole = Literal["participant", "scientist"]


class LoginRequest(BaseModel):
    # Submitted credentials
    role: AuthRole
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=200)


class AuthSession(BaseModel):
    # Safe session details
    role: AuthRole
    username: str
