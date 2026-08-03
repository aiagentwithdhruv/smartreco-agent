"""SmartReco FastAPI application.

Run: uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.db import init_db
from app.routers import auth, pages
from app.templating import STATIC_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="SmartReco",
    description="Behavioral AI recommendation agent — SmartReco Build Challenge 2026 (Mesh API).",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(auth.router)
app.include_router(pages.router)


@app.get("/healthz", tags=["ops"])
def healthz() -> dict[str, str]:
    """Liveness probe — also the quickest way to confirm the app booted."""
    return {"status": "ok"}
