import secrets

from fastapi import APIRouter, HTTPException, Request, Response, status

from backend.src.envs import env
from backend.src.dtypes.auth import AuthSession, LoginRequest


router = APIRouter(prefix="/api/auth", tags=["authentication"])


@router.post("/login", response_model=AuthSession)
def login(credentials: LoginRequest, request: Request):
    """Validate a configured account and create a signed browser session."""

    # Select only the configured credentials for the requested application role.
    expected_username = env.participant_username if credentials.role == "participant" else env.scientist_username
    expected_password = env.participant_password if credentials.role == "participant" else env.scientist_password
    username_valid = secrets.compare_digest(credentials.username.encode(), expected_username.encode())
    password_valid = secrets.compare_digest(credentials.password.encode(), expected_password.get_secret_value().encode())
    if not username_valid or not password_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    # Replace any prior role session without retaining credentials in the cookie.
    request.session.clear()
    request.session.update({"role": credentials.role, "username": credentials.username})
    return {"role": credentials.role, "username": credentials.username}


@router.get("/session", response_model=AuthSession)
def session(request: Request):
    """Return the authenticated role stored in the signed session cookie."""

    # Reject missing or malformed session values before returning them to the client.
    role = request.session.get("role")
    username = request.session.get("username")
    if role not in {"participant", "scientist"} or not isinstance(username, str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return {"role": role, "username": username}


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request):
    """Clear the current signed browser session."""

    # Remove all role and identity state from the session cookie.
    request.session.clear()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
