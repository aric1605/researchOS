"""ResearchOS FastAPI Application."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app import __version__
from app.config import settings
from app.db.connection import check_postgres_health, run_migrations, close_db_pool
from app.clients.vectorstore import check_qdrant_health
from app.api.routes import router as api_router, ensure_worker_started

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("researchos")

static_dir = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting ResearchOS v{__version__}...")
    # Initialize background worker loop
    ensure_worker_started()
    # Run database migrations if DATABASE_URL is configured
    try:
        await run_migrations()
    except Exception as e:
        logger.warning(f"Could not run migrations on startup: {e}")
    yield
    logger.info("Shutting down ResearchOS...")
    await close_db_pool()


app = FastAPI(
    title="ResearchOS API",
    description="Multi-Agent Autonomous Research System",
    version=__version__,
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API endpoints
app.include_router(api_router)


@app.get("/", tags=["UI"])
async def serve_index():
    """Serves the single-page application frontend."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "ResearchOS API running. Static frontend not found."}


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint checking PostgreSQL and Qdrant connectivity."""
    pg_health = await check_postgres_health()
    qdrant_health = check_qdrant_health()

    is_healthy = (
        pg_health.get("status") in ("healthy", "unconfigured")
        and qdrant_health.get("status") in ("healthy", "unconfigured")
    )

    return {
        "status": "healthy" if is_healthy else "degraded",
        "version": __version__,
        "services": {
            "postgres": pg_health,
            "qdrant": qdrant_health,
        },
    }


# Static files mount
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
