"""
Pydantic Data Schemas for CardioGuard AI.
Ensures rigorous validation, schema constraints, and clear OpenAPI documentation.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime

class PatientInputSchema(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 58,
                "gender": 1,
                "systolic_bp": 145,
                "diastolic_bp": 92,
                "cholesterol": 245,
                "glucose": 125,
                "bmi": 29.3,
                "smoking": 1,
                "alcohol_intake": 0,
                "physical_activity": 0,
                "resting_ecg": 1,
                "max_heart_rate": 135,
                "chest_pain_type": 0,
                "exercise_angina": 1
            }
        }
    )

    age: int = Field(..., ge=18, le=100, description="Patient age in years (18 - 100)")
    gender: int = Field(..., ge=0, le=1, description="Biological sex (1 = Male, 0 = Female)")
    systolic_bp: int = Field(..., ge=80, le=240, description="Systolic Blood Pressure in mmHg (80 - 240)")
    diastolic_bp: int = Field(..., ge=50, le=140, description="Diastolic Blood Pressure in mmHg (50 - 140)")
    cholesterol: int = Field(..., ge=100, le=500, description="Total Serum Cholesterol in mg/dL (100 - 500)")
    glucose: int = Field(..., ge=50, le=400, description="Fasting Blood Glucose in mg/dL (50 - 400)")
    bmi: float = Field(..., ge=14.0, le=60.0, description="Body Mass Index in kg/m² (14.0 - 60.0)")
    smoking: int = Field(..., ge=0, le=1, description="Tobacco smoking habit (0 = No, 1 = Yes)")
    alcohol_intake: int = Field(..., ge=0, le=1, description="Alcohol consumption (0 = No, 1 = Yes)")
    physical_activity: int = Field(..., ge=0, le=1, description="Regular physical exercise (0 = Inactive, 1 = Active)")
    resting_ecg: int = Field(..., ge=0, le=2, description="Resting ECG (0 = Normal, 1 = ST-T Abnormality, 2 = LVH)")
    max_heart_rate: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved during stress test (bpm)")
    chest_pain_type: int = Field(..., ge=0, le=3, description="Chest pain type (0=Typical Angina, 1=Atypical, 2=Non-Anginal, 3=Asymptomatic)")
    exercise_angina: int = Field(..., ge=0, le=1, description="Exercise-induced angina present (0 = No, 1 = Yes)")

    @field_validator("diastolic_bp")
    @classmethod
    def validate_bp(cls, v, info):
        systolic = info.data.get("systolic_bp")
        if systolic and v >= systolic:
            raise ValueError("Diastolic blood pressure must be lower than systolic blood pressure.")
        return v

class RiskFactorBreakdown(BaseModel):
    feature: str
    display_name: str
    patient_value: Any
    benchmark: str
    status: str  # Normal, Elevated, High, Critical
    impact_level: str  # Low, Moderate, High
    relative_contribution_pct: float
    description: str

class ClinicalRecommendation(BaseModel):
    category: str  # Lifestyle, Pharmacotherapy / Medical, Monitoring, Diet
    priority: str  # Immediate, High, Medium, Low
    title: str
    action_item: str
    clinical_rationale: str

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = Low Risk / Negative, 1 = High Risk / Positive")
    prediction_label: str
    risk_probability: float
    risk_percentage: float
    confidence: float
    risk_tier: str  # Low Risk, Moderate Risk, High Risk, Critical Risk
    risk_color: str
    risk_breakdown: List[RiskFactorBreakdown]
    recommendations: List[ClinicalRecommendation]
    patient_values: Dict[str, Any]
    timestamp: str

class BatchSummary(BaseModel):
    total_records: int
    high_risk_count: int
    low_risk_count: int
    high_risk_percentage: float
    mean_risk_score: float

class BatchPredictionResponse(BaseModel):
    summary: BatchSummary
    preview: List[Dict[str, Any]]
    csv_download_ready: bool
    filename: Optional[str] = None

class ModelMetricsResponse(BaseModel):
    model_name: str
    model_type: str
    trained_at: str
    dataset_size: int
    training_samples: int
    test_samples: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    brier_score: float
    confusion_matrix: List[List[int]]
    feature_importances: Dict[str, Any]
    all_model_benchmarks: Dict[str, Any]
    dataset_statistics: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_name: Optional[str] = None
    timestamp: str
