"""
Health Check and System Diagnostics Endpoint.
"""

from datetime import datetime
from fastapi import APIRouter
from app.schemas import HealthResponse
from app.services.predictor import predictor_service
from app.config import settings

router = APIRouter(prefix="", tags=["System"])

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Readiness Probe",
    description="Returns API status, uptime, version, and model readiness status."
)
async def health_check():
    model_loaded = predictor_service.is_model_loaded
    model_name = None
    if model_loaded and predictor_service.pipeline.metadata:
        model_name = predictor_service.pipeline.metadata.get("model_name")

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version=settings.VERSION,
        model_loaded=model_loaded,
        model_name=model_name,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )
