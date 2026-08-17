"""
Integration tests for CardioGuard AI FastAPI endpoints.
"""

import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data

def test_root_dashboard():
    response = client.get("/")
    assert response.status_code == 200
    assert "CardioGuard" in response.text
    assert "<!DOCTYPE html>" in response.text

def test_predict_single_endpoint():
    payload = {
        "age": 60,
        "gender": 1,
        "systolic_bp": 150,
        "diastolic_bp": 95,
        "cholesterol": 260,
        "glucose": 130,
        "bmi": 30.5,
        "smoking": 1,
        "alcohol_intake": 0,
        "physical_activity": 0,
        "resting_ecg": 1,
        "max_heart_rate": 130,
        "chest_pain_type": 0,
        "exercise_angina": 1
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["prediction"] in [0, 1]
    assert "risk_probability" in data
    assert "risk_percentage" in data
    assert "risk_tier" in data
    assert "risk_breakdown" in data
    assert len(data["risk_breakdown"]) > 0
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0

def test_predict_invalid_payload():
    payload = {
        "age": 10,  # Below minimum 18
        "gender": 5   # Invalid
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422

def test_model_metrics_endpoint():
    response = client.get("/api/v1/model-metrics")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "accuracy" in data
    assert "roc_auc" in data
    assert "feature_importances" in data
    assert "all_model_benchmarks" in data

def test_roc_curve_endpoint():
    response = client.get("/api/v1/roc-curve")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

def test_sample_csv_download():
    response = client.get("/api/v1/sample-csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "age,gender,systolic_bp" in response.text

def test_predict_batch_endpoint():
    csv_content = (
        "age,gender,systolic_bp,diastolic_bp,cholesterol,glucose,bmi,smoking,alcohol_intake,physical_activity,resting_ecg,max_heart_rate,chest_pain_type,exercise_angina\n"
        "45,1,120,80,180,90,24.5,0,0,1,0,165,2,0\n"
        "65,1,160,100,280,170,32.0,1,1,0,2,115,0,1\n"
    )
    files = {"file": ("test_cohort.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/api/v1/predict-batch", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["total_records"] == 2
    assert len(data["preview"]) == 2
    assert "cvd_risk_probability" in data["preview"][0]
