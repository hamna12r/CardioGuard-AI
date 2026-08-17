"""
Unit tests for Pydantic schema validation and clinical boundary checks.
"""

import pytest
from pydantic import ValidationError
from app.schemas import PatientInputSchema

def test_valid_patient_schema():
    valid_data = {
        "age": 52,
        "gender": 0,
        "systolic_bp": 130,
        "diastolic_bp": 85,
        "cholesterol": 210,
        "glucose": 95,
        "bmi": 26.2,
        "smoking": 0,
        "alcohol_intake": 0,
        "physical_activity": 1,
        "resting_ecg": 0,
        "max_heart_rate": 155,
        "chest_pain_type": 2,
        "exercise_angina": 0
    }
    patient = PatientInputSchema(**valid_data)
    assert patient.age == 52
    assert patient.gender == 0

def test_invalid_age_raises_error():
    invalid_data = {
        "age": 12,  # Under 18
        "gender": 1,
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "cholesterol": 200,
        "glucose": 90,
        "bmi": 24.0,
        "smoking": 0,
        "alcohol_intake": 0,
        "physical_activity": 1,
        "resting_ecg": 0,
        "max_heart_rate": 150,
        "chest_pain_type": 1,
        "exercise_angina": 0
    }
    with pytest.raises(ValidationError):
        PatientInputSchema(**invalid_data)

def test_diastolic_higher_than_systolic_raises_error():
    invalid_data = {
        "age": 45,
        "gender": 1,
        "systolic_bp": 110,
        "diastolic_bp": 130,  # Invalid: Diastolic > Systolic
        "cholesterol": 200,
        "glucose": 90,
        "bmi": 24.0,
        "smoking": 0,
        "alcohol_intake": 0,
        "physical_activity": 1,
        "resting_ecg": 0,
        "max_heart_rate": 150,
        "chest_pain_type": 1,
        "exercise_angina": 0
    }
    with pytest.raises(ValidationError):
        PatientInputSchema(**invalid_data)
