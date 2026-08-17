"""
Unit tests for CardioGuard AI ML pipeline and model artifacts.
"""

import os
import pytest
import pandas as pd
import numpy as np

from model.pipeline import CardioPipeline, FEATURE_COLUMNS
from data.make_dataset import generate_cardiovascular_dataset

@pytest.fixture
def sample_patient():
    return {
        "age": 55,
        "gender": 1,
        "systolic_bp": 140,
        "diastolic_bp": 90,
        "cholesterol": 240,
        "glucose": 115,
        "bmi": 28.5,
        "smoking": 1,
        "alcohol_intake": 0,
        "physical_activity": 0,
        "resting_ecg": 1,
        "max_heart_rate": 138,
        "chest_pain_type": 0,
        "exercise_angina": 1
    }

def test_dataset_generation():
    df = generate_cardiovascular_dataset(n_samples=100, random_state=42)
    assert len(df) == 100
    assert "cardiovascular_disease" in df.columns
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert df["cardiovascular_disease"].isin([0, 1]).all()

def test_pipeline_loading():
    pipeline = CardioPipeline()
    assert pipeline.is_ready
    assert pipeline.model is not None
    assert pipeline.scaler is not None
    assert pipeline.metadata is not None

def test_pipeline_single_prediction(sample_patient):
    pipeline = CardioPipeline()
    result = pipeline.predict_single(sample_patient)
    
    assert "prediction" in result
    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert 0.0 <= result["risk_percentage"] <= 100.0
    assert result["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
    assert "risk_color" in result

def test_pipeline_batch_prediction():
    pipeline = CardioPipeline()
    test_df = generate_cardiovascular_dataset(n_samples=25, random_state=99)[FEATURE_COLUMNS]
    
    result_df, summary = pipeline.predict_batch(test_df)
    assert len(result_df) == 25
    assert "cvd_risk_probability" in result_df.columns
    assert "cvd_prediction" in result_df.columns
    assert summary["total_records"] == 25
    assert summary["high_risk_count"] + summary["low_risk_count"] == 25
