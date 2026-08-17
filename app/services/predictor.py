"""
Predictor service coordinator for CardioGuard AI.
Integrates the inference pipeline, XAI explainer, and clinical advisor into a unified service.
"""

import os
from datetime import datetime
from typing import Dict, Any, Tuple
import pandas as pd

from model.pipeline import CardioPipeline
from app.services.explainer import ExplainerService
from app.services.advisor import AdvisorService
from app.config import settings
from app.schemas import PredictionResponse, BatchPredictionResponse, BatchSummary, ModelMetricsResponse

class PredictorService:
    def __init__(self):
        self.pipeline = CardioPipeline(artifacts_dir=settings.ARTIFACTS_DIR)

    @property
    def is_model_loaded(self) -> bool:
        return self.pipeline.is_ready

    def predict_patient(self, patient_dict: Dict[str, Any]) -> PredictionResponse:
        # Run inference
        base_result = self.pipeline.predict_single(patient_dict)
        
        # Extract metadata feature importances
        feature_importances = self.pipeline.metadata.get("feature_importances", {}) if self.pipeline.metadata else {}

        # Compute XAI breakdown
        breakdown = ExplainerService.explain_patient_risk(
            patient_values=patient_dict,
            risk_probability=base_result["risk_probability"],
            feature_importances=feature_importances
        )

        # Generate clinical recommendations
        recommendations = AdvisorService.generate_recommendations(
            patient_values=patient_dict,
            risk_tier=base_result["risk_tier"],
            risk_probability=base_result["risk_probability"]
        )

        return PredictionResponse(
            prediction=base_result["prediction"],
            prediction_label=base_result["prediction_label"],
            risk_probability=base_result["risk_probability"],
            risk_percentage=base_result["risk_percentage"],
            confidence=base_result["confidence"],
            risk_tier=base_result["risk_tier"],
            risk_color=base_result["risk_color"],
            risk_breakdown=breakdown,
            recommendations=recommendations,
            patient_values=patient_dict,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    def predict_batch_csv(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, BatchPredictionResponse]:
        result_df, summary = self.pipeline.predict_batch(df)

        preview_records = result_df.head(10).to_dict(orient="records")

        response = BatchPredictionResponse(
            summary=BatchSummary(
                total_records=summary["total_records"],
                high_risk_count=summary["high_risk_count"],
                low_risk_count=summary["low_risk_count"],
                high_risk_percentage=summary["high_risk_percentage"],
                mean_risk_score=summary["mean_risk_score"]
            ),
            preview=preview_records,
            csv_download_ready=True
        )

        return result_df, response

    def get_metrics(self) -> ModelMetricsResponse:
        if not self.pipeline.metadata:
            raise RuntimeError("Model metadata is not loaded.")

        meta = self.pipeline.metadata
        best_metrics = meta.get("best_model_metrics", {})

        return ModelMetricsResponse(
            model_name=meta.get("model_name", "Unknown"),
            model_type=meta.get("model_type", "Unknown"),
            trained_at=meta.get("trained_at", ""),
            dataset_size=meta.get("dataset_size", 0),
            training_samples=meta.get("training_samples", 0),
            test_samples=meta.get("test_samples", 0),
            accuracy=best_metrics.get("accuracy", 0.0),
            precision=best_metrics.get("precision", 0.0),
            recall=best_metrics.get("recall", 0.0),
            f1_score=best_metrics.get("f1_score", 0.0),
            roc_auc=best_metrics.get("roc_auc", 0.0),
            brier_score=best_metrics.get("brier_score", 0.0),
            confusion_matrix=best_metrics.get("confusion_matrix", []),
            feature_importances=meta.get("feature_importances", {}),
            all_model_benchmarks=meta.get("all_model_benchmarks", {}),
            dataset_statistics=meta.get("dataset_statistics", {})
        )

predictor_service = PredictorService()
