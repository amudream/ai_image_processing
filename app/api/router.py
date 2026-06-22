from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import assets, generation_jobs, health, imports, outputs, visual_units

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(imports.router, tags=["imports"])
api_router.include_router(assets.router, prefix="/assets", tags=["assets"])
api_router.include_router(visual_units.router, prefix="/visual-units", tags=["visual-units"])
api_router.include_router(
    generation_jobs.router, prefix="/generation-jobs", tags=["generation-jobs"]
)
api_router.include_router(outputs.router, prefix="/outputs", tags=["outputs"])
