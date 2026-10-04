"""PixelMend FastAPI backend main application entrypoint."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.backend.config import get_settings
from src.app.backend.routers import health, task1, task2, task3, task4
from src.app.backend.schemas.common import RootInfoResponse
from src.app.backend.services.inference import session_manager

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: preload ONNX model sessions at startup."""
    session_manager.load_all_models()
    yield
    session_manager.clear()


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "Multi-workspace deep learning computer vision REST API supporting "
        "Universal Restoration, Hard-Routing, Soft MoE, and Face-to-Sketch generation."
    ),
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(task1.router)
app.include_router(task2.router)
app.include_router(task3.router)
app.include_router(task4.router)


@app.get("/", response_model=RootInfoResponse, tags=["General"])
def read_root() -> RootInfoResponse:
    """API root informational summary and documentation links."""
    return RootInfoResponse(
        app_name=settings.app_name,
        version=settings.version,
        docs_url="/docs",
        health_url="/health",
        status="running",
    )
