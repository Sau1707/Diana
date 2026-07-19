from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from backend.src.envs import env
from backend.src.routes import auth


class HealthResponse(BaseModel):
    # API status fields
    status: str
    frontend_built: bool


DIST = env.frontend_dist.resolve()

app = FastAPI(
    title=env.app_name,
    description="Backend and production frontend host for the DIANA prototype.",
    version="0.1.0",
)
app.add_middleware(
    SessionMiddleware,
    secret_key=env.session_secret.get_secret_value(),
    session_cookie="diana_session",
    max_age=env.session_max_age,
    same_site="lax",
    https_only=env.secure_cookies or env.deployment in {"preview", "production"},
)
app.include_router(auth.router)


@app.get("/api/health", response_model=HealthResponse)
def health():
    """Return the backend and frontend build status."""

    # Report whether the production SPA is available.
    return {"status": "ok", "frontend_built": (DIST / "index.html").is_file()}


@app.get("/api", include_in_schema=False)
@app.get("/api/{path:path}", include_in_schema=False)
def missing_api(path: str = ""):
    """Keep unknown API requests out of the frontend SPA fallback."""

    # Preserve JSON API semantics for endpoints that do not exist.
    raise HTTPException(status_code=404, detail=f"API endpoint not found: /api/{path}")


# Mount immutable Vite assets before the SPA fallback when a build is present.
if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def serve_spa(path: str):
    """Serve built frontend files and fall back to the SPA entry point."""

    # Return public files such as the favicon without bypassing the build directory.
    requested = (DIST / path).resolve()
    if requested.is_relative_to(DIST.resolve()) and requested.is_file():
        return FileResponse(requested)

    # Let React Router resolve all remaining non-API application routes.
    index = DIST / "index.html"
    if index.is_file():
        return FileResponse(index)

    raise HTTPException(
        status_code=503,
        detail="Frontend build not found. Run `npm run build` in frontend first.",
    )
