"""
Configuration settings for CardioGuard AI Application.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env")
    
    PROJECT_NAME: str = "CardioGuard AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DESCRIPTION: str = (
        "Production-grade Clinical AI System for Cardiovascular Disease Risk Stratification, "
        "Explainable AI (XAI) Attribution, and Evidence-Based Preventive Health Guidance."
    )
    
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ARTIFACTS_DIR: str = os.path.join(BASE_DIR, "model", "artifacts")
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

settings = Settings()
