"""
Prediction API Endpoints for CardioGuard AI.
Handles single patient risk evaluation and batch CSV processing.
"""

import io
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, Response
from fastapi.responses import StreamingResponse

from app.schemas import PatientInputSchema, PredictionResponse, BatchPredictionResponse
from app.services.predictor import predictor_service
from model.pipeline import FEATURE_COLUMNS

router = APIRouter(prefix="", tags=["Predictions"])

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Cardiovascular Disease Risk for Single Patient",
    description="Accepts physiological biomarkers, vital signs, and lifestyle indicators to predict CVD risk, compute XAI factor breakdown, and generate tailored clinical recommendations."
)
async def predict_single_patient(patient: PatientInputSchema):
    try:
        result = predictor_service.predict_patient(patient.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@router.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    summary="Batch Predict CVD Risk via CSV Upload",
    description="Upload a CSV file containing multiple patient records. Returns risk stratification summary and preview table."
)
async def predict_batch_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Check required columns
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required columns in CSV: {missing}. Required columns: {FEATURE_COLUMNS}"
            )

        result_df, response = predictor_service.predict_batch_csv(df)
        response.filename = file.filename
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")

@router.get(
    "/sample-csv",
    summary="Download Sample Patient Batch CSV Template",
    description="Downloads a validated sample CSV with 10 patient records ready for batch prediction testing."
)
async def get_sample_csv():
    sample_data = {
        "age": [45, 62, 53, 71, 38, 59, 49, 66, 55, 68],
        "gender": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        "systolic_bp": [122, 155, 138, 168, 115, 142, 128, 160, 134, 150],
        "diastolic_bp": [78, 96, 88, 102, 74, 90, 82, 98, 86, 94],
        "cholesterol": [185, 260, 225, 285, 170, 240, 195, 275, 210, 255],
        "glucose": [88, 140, 105, 175, 82, 118, 94, 160, 98, 145],
        "bmi": [24.2, 31.5, 27.8, 33.1, 22.0, 29.4, 25.1, 32.0, 26.5, 30.2],
        "smoking": [0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        "alcohol_intake": [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        "physical_activity": [1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        "resting_ecg": [0, 1, 0, 2, 0, 1, 0, 2, 0, 1],
        "max_heart_rate": [165, 122, 140, 110, 178, 132, 155, 118, 148, 125],
        "chest_pain_type": [2, 0, 1, 0, 3, 0, 2, 0, 1, 0],
        "exercise_angina": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(sample_data)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cardioguard_sample_batch.csv"}
    )
