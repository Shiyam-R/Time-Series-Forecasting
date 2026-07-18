"""
app/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application factory.

Start the server:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Interactive docs:
    http://localhost:8000/docs      — Swagger UI
    http://localhost:8000/redoc     — ReDoc
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import API_TITLE, API_VERSION, API_DESCRIPTION, ENVIRONMENT, CORS_ALLOWED_ORIGINS
from app.exceptions import RossmannAPIError
from app.model_loader import load_artifacts
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Load all model artifacts exactly once before the API accepts requests.
    Runs artifact cleanup (if needed) on shutdown.
    """
    logger.info("=" * 60)
    logger.info("Rossmann Forecasting API — starting up …")
    logger.info("Environment: %s", ENVIRONMENT)
    logger.info("=" * 60)

    try:
        load_artifacts()
        logger.info("Startup complete. API is ready to serve requests.")
    except Exception as exc:
        logger.critical("Startup failed: %s", exc, exc_info=True)
        raise

    yield  # API is live here

    logger.info("Rossmann Forecasting API — shutting down.")


# ── Application Factory ───────────────────────────────────────────────────────
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title       = API_TITLE,
        version     = API_VERSION,
        description = API_DESCRIPTION,
        lifespan    = lifespan,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Origins are read from CORS_ALLOWED_ORIGINS (see app/config.py). Defaults
    # to ["*"] to preserve prior behavior exactly — set the env var to a
    # comma-separated allow-list to tighten this for a real deployment.
    application.add_middleware(
        CORSMiddleware,
        allow_origins     = CORS_ALLOWED_ORIGINS,
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ── Centralised exception handler ─────────────────────────────────────────
    @application.exception_handler(RossmannAPIError)
    async def rossmann_error_handler(
        request: Request, exc: RossmannAPIError
    ) -> JSONResponse:
        """
        Convert any RossmannAPIError subclass into a structured JSON response.
        All custom exceptions carry their own HTTP status code.
        """
        logger.warning(
            "Handled error [%s] — %s",
            exc.status_code, exc.detail,
        )
        return JSONResponse(
            status_code = exc.status_code,
            content     = {
                "status":  "error",
                "message": exc.message,
                "detail":  exc.detail,
                "code":    exc.status_code,
            },
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    application.include_router(router)

    return application


app = create_app()