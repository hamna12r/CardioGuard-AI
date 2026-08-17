"""
Metrics and Model Metadata Endpoints for CardioGuard AI.
Exposes evaluation metrics, confusion matrices, and ROC visualizations.
"""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.schemas import ModelMetricsResponse
from app.services.predictor import predictor_service
from app.config import settings

router = APIRouter(prefix="", tags=["Model Metrics"])

@router.get(
    "/model-metrics",
    response_model=ModelMetricsResponse,
    summary="Get Model Performance & Feature Importances",
    description="Retrieves the champion model evaluation metrics, benchmark comparison across all 4 candidate models, confusion matrix, and feature importance rankings."
)
async def get_model_metrics():
    try:
        metrics = predictor_service.get_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model metrics: {str(e)}")

@router.get(
    "/roc-curve",
    summary="Get Multi-Model ROC Curve Plot",
    description="Returns the serialized PNG image showing ROC-AUC curves comparison for all evaluated algorithms."
)
async def get_roc_curve_plot():
    plot_path = os.path.join(settings.ARTIFACTS_DIR, "roc_curve.png")
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="ROC plot not found. Run model training first.")
    return FileResponse(plot_path, media_type="image/png")
