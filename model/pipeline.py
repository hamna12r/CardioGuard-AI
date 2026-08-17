"""
Inference and Preprocessing Pipeline for CardioGuard AI.
Handles input feature transformation, model loading, prediction formatting,
and feature importance calculations.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

# Feature column definitions in exact model order
FEATURE_COLUMNS = [
    "age",
    "gender",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "glucose",
    "bmi",
    "smoking",
    "alcohol_intake",
    "physical_activity",
    "resting_ecg",
    "max_heart_rate",
    "chest_pain_type",
    "exercise_angina"
]

FEATURE_DISPLAY_NAMES = {
    "age": "Age (years)",
    "gender": "Biological Sex",
    "systolic_bp": "Systolic Blood Pressure (mmHg)",
    "diastolic_bp": "Diastolic Blood Pressure (mmHg)",
    "cholesterol": "Total Serum Cholesterol (mg/dL)",
    "glucose": "Fasting Blood Glucose (mg/dL)",
    "bmi": "Body Mass Index (kg/m²)",
    "smoking": "Tobacco Smoking",
    "alcohol_intake": "Alcohol Consumption",
    "physical_activity": "Physical Activity Level",
    "resting_ecg": "Resting ECG Results",
    "max_heart_rate": "Max Heart Rate (bpm)",
    "chest_pain_type": "Chest Pain Classification",
    "exercise_angina": "Exercise-Induced Angina"
}

FEATURE_BENCHMARKS = {
    "age": {"normal_min": 18, "normal_max": 50, "unit": "years"},
    "systolic_bp": {"normal_min": 90, "normal_max": 120, "unit": "mmHg"},
    "diastolic_bp": {"normal_min": 60, "normal_max": 80, "unit": "mmHg"},
    "cholesterol": {"normal_min": 125, "normal_max": 200, "unit": "mg/dL"},
    "glucose": {"normal_min": 70, "normal_max": 100, "unit": "mg/dL"},
    "bmi": {"normal_min": 18.5, "normal_max": 24.9, "unit": "kg/m²"},
    "max_heart_rate": {"normal_min": 120, "normal_max": 180, "unit": "bpm"}
}

class CardioPipeline:
    def __init__(self, artifacts_dir: Optional[str] = None):
        if artifacts_dir is None:
            artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.scaler = None
        self.metadata = None
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = os.path.join(self.artifacts_dir, "best_model.joblib")
        scaler_path = os.path.join(self.artifacts_dir, "scaler.joblib")
        meta_path = os.path.join(self.artifacts_dir, "model_metadata.json")

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        # Ensure all columns present and in correct order
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required feature: {col}")
        ordered_df = df[FEATURE_COLUMNS]
        return self.scaler.transform(ordered_df)

    def predict_single(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError("CardioPipeline artifacts are not loaded. Train the model first.")

        df = pd.DataFrame([patient_dict])[FEATURE_COLUMNS]
        X_scaled = self.scaler.transform(df)
        
        prob = float(self.model.predict_proba(X_scaled)[0, 1])
        prediction = int(prob >= 0.5)

        # Risk Stratification Tier
        if prob < 0.25:
            risk_tier = "Low Risk"
            risk_color = "#10b981"  # Emerald green
        elif prob < 0.50:
            risk_tier = "Moderate Risk"
            risk_color = "#f59e0b"  # Amber
        elif prob < 0.75:
            risk_tier = "High Risk"
            risk_color = "#f97316"  # Orange
        else:
            risk_tier = "Critical Risk"
            risk_color = "#ef4444"  # Rose red

        confidence = prob if prediction == 1 else (1.0 - prob)

        return {
            "prediction": prediction,
            "prediction_label": "High Cardiovascular Risk" if prediction == 1 else "Low Cardiovascular Risk",
            "risk_probability": round(prob, 4),
            "risk_percentage": round(prob * 100, 2),
            "confidence": round(confidence * 100, 2),
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "patient_values": patient_dict
        }

    def predict_batch(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if not self.is_ready:
            raise RuntimeError("CardioPipeline artifacts are not loaded. Train the model first.")

        # Validate columns
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Batch dataset missing columns: {missing}")

        X_scaled = self.scaler.transform(df[FEATURE_COLUMNS])
        probs = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)

        result_df = df.copy()
        result_df["cvd_risk_probability"] = np.round(probs, 4)
        result_df["cvd_risk_percentage"] = np.round(probs * 100, 2)
        result_df["cvd_prediction"] = preds
        result_df["risk_tier"] = [
            "Low Risk" if p < 0.25 else "Moderate Risk" if p < 0.50 else "High Risk" if p < 0.75 else "Critical Risk"
            for p in probs
        ]

        summary = {
            "total_records": len(df),
            "high_risk_count": int(preds.sum()),
            "low_risk_count": int(len(df) - preds.sum()),
            "high_risk_percentage": round(float(preds.mean() * 100), 2),
            "mean_risk_score": round(float(probs.mean() * 100), 2)
        }

        return result_df, summary
