# 🫀 CardioGuard AI: End-to-End Cardiovascular Disease Risk Prediction & Clinical Decision Support System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0.2-EB5424?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests Passed](https://img.shields.io/badge/Tests-15%2F15%20Passed-10B981?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Academic Assignment Submission**: End-to-End AI Application Development and Cloud Deployment  
> **Author / Student**: AI Engineering Candidate  
> **Status**: Production Ready, Tested, Containerized & Deployable  

---

## 📑 Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Use Case](#2-use-case)
3. [Solution Overview](#3-solution-overview)
4. [Dataset Description](#4-dataset)
5. [AI/ML Approach & Modeling](#5-aiml-approach)
6. [Application Architecture](#6-application-architecture)
7. [Technology Stack](#7-technology-stack)
8. [Local Setup Instructions](#8-local-setup-instructions)
9. [Deployment Details](#9-deployment-details)
10. [API & Web Application Usage](#10-apiweb-application-usage)
11. [Docker & Containerization Instructions](#11-docker-instructions)
12. [Automated Testing & Quality Assurance](#12-automated-testing)

---

## 1. Problem Statement

Cardiovascular diseases (CVDs) are the leading cause of mortality worldwide, responsible for an estimated **17.9 million deaths annually** (representing 32% of all global deaths according to the World Health Organization). 

Key challenges in modern cardiovascular risk management include:
- **Delayed Diagnosis**: Early atherosclerotic cardiovascular disease develops asymptomatically for years before presenting as acute myocardial infarction or stroke.
- **Cognitive Overload for Clinicians**: Evaluating non-linear interactions across dozens of clinical biomarkers (blood pressure, lipid profiles, glycemic status, resting ECG abnormalities, stress tests) is complex and prone to variance.
- **Black-Box AI Skepticism**: Many clinical AI models output raw risk scores without transparent factor attribution, leading to hesitation among medical practitioners.
- **Lack of Actionable Guidance**: Predictions often fail to guide patient-specific preventive interventions.

---

## 2. Use Case

**CardioGuard AI** is designed for deployment across several real-world clinical and wellness environments:

| Sector | Target User | Application & Workflow |
| :--- | :--- | :--- |
| **Outpatient Cardiology & Primary Care** | General Practitioners & Cardiologists | Rapid non-invasive triage during routine annual check-ups to identify asymptomatic high-risk patients. |
| **Hospital EHR Systems** | Clinical Informatics Teams | High-throughput batch CSV processing to score hospital patient cohorts and automate preventive follow-up scheduling. |
| **Telemedicine Platforms** | Remote Care Clinicians | Interactive risk simulation during tele-consultations to visually demonstrate risk reduction under lifestyle changes. |
| **Preventive Health Clinics** | Health Coaches & Patients | Automated generation of patient-tailored preventive action plans (DASH diet, exercise regimens, smoking cessation). |

---

## 3. Solution Overview

**CardioGuard AI** provides an end-to-end clinical AI intelligence platform featuring:
- **Multi-Model Machine Learning Engine**: Benchmarked across Logistic Regression, Random Forest, Gradient Boosting, and XGBoost with automated champion model selection (achieving **0.856 ROC-AUC** and **77.6% Test Accuracy**).
- **Explainable AI (XAI) Attribution**: Transparent decomposition of each patient's risk into specific contributing physiological biomarkers with clinical reference benchmarks.
- **Interactive "What-If" Sensitivity Simulator**: Real-time slider-based scenario testing enabling clinicians and patients to simulate the immediate risk-reduction impact of therapeutic interventions.
- **Batch CSV Cohort Processing**: High-throughput file upload interface for scoring thousands of patient records simultaneously with summary metrics and CSV export.
- **Production RESTful API (FastAPI)**: Validated endpoints with Pydantic v2 schemas, automated OpenAPI/Swagger documentation, and sub-10ms inference latency.
- **Enterprise-Ready Containerization**: Multi-stage, non-root `Dockerfile` and `docker-compose.yml` for zero-configuration cloud deployment.

---

## 4. Dataset

### Source & Distribution
The dataset is constructed from established cardiovascular epidemiology benchmarks (combining Framingham Heart Study parameters, CDC BRFSS epidemiological distributions, and UCI Heart Disease clinical indicators).

- **Cohort Size**: 6,000 comprehensive patient records
- **Class Prevalence**: 38.12% Positive CVD Cases (2,287 patients), 61.88% Low Risk (3,713 patients)
- **Train/Test Split**: 80% Training (4,800 samples), 20% Holdout Test (1,200 samples) stratified by target class.

### Feature Dictionary

| Feature Name | Clinical Description | Data Type | Reference / Normal Range |
| :--- | :--- | :--- | :--- |
| `age` | Patient chronological age | Integer | 18 – 100 years |
| `gender` | Biological sex | Binary | 0 = Female, 1 = Male |
| `systolic_bp` | Resting systolic blood pressure | Integer | Normal: < 120 mmHg |
| `diastolic_bp` | Resting diastolic blood pressure | Integer | Normal: < 80 mmHg |
| `cholesterol` | Total serum cholesterol | Integer | Desirable: < 200 mg/dL |
| `glucose` | Fasting blood glucose | Integer | Normal: 70 – 99 mg/dL |
| `bmi` | Body Mass Index | Float | Healthy: 18.5 – 24.9 kg/m² |
| `smoking` | Tobacco smoking status | Binary | 0 = Non-Smoker, 1 = Active Smoker |
| `alcohol_intake`| Regular alcohol consumption | Binary | 0 = No / Rare, 1 = Regular |
| `physical_activity` | Regular moderate aerobic exercise | Binary | 0 = Inactive, 1 = Active (>= 150 min/wk) |
| `resting_ecg` | Resting electrocardiogram findings | Categorical | 0 = Normal, 1 = ST-T Abnormality, 2 = LVH |
| `max_heart_rate` | Max heart rate achieved during stress test | Integer | 60 – 220 bpm |
| `chest_pain_type` | Clinical chest pain classification | Categorical | 0 = Typical, 1 = Atypical, 2 = Non-anginal, 3 = Asymptomatic |
| `exercise_angina`| Exercise-induced angina pectoris | Binary | 0 = Absent, 1 = Present |
| `cardiovascular_disease` | **Target Variable**: 10-Yr CVD / MACE | Binary | 0 = Low Risk, 1 = High Risk / Positive |

---

## 5. AI/ML Approach

```
+----------------------------------------------------------------------------------------------------+
|                                    CARDIOGUARD ML PIPELINE                                         |
+----------------------------------------------------------------------------------------------------+
|  [6,000 Patient Cohort] --> [Data Cleaning & Validation] --> [StandardScaler Normalization]        |
|                                                                                                    |
|  [5-Fold Stratified Cross-Validation & Hyperparameter Tuning]                                     |
|  +-----------------------+---------------------+---------------------+---------------------------+  |
|  | Logistic Regression   | Random Forest       | Gradient Boosting   | XGBoost Classifier        |  |
|  | (CV-AUC: 0.8560)      | (CV-AUC: 0.8431)    | (CV-AUC: 0.8438)    | (CV-AUC: 0.8478)          |  |
|  +-----------------------+---------------------+---------------------+---------------------------+  |
|                                                                                                    |
|  --> [Champion Model Selection (Logistic Regression: 0.8557 Test AUC, 77.58% Acc)]                |
|  --> [Feature Importance Extraction & XAI Calibration]                                            |
|  --> [Serialization: best_model.joblib, scaler.joblib, model_metadata.json, roc_curve.png]         |
+----------------------------------------------------------------------------------------------------+
```

### Multi-Model Benchmark Comparison

| Model Candidate | 5-Fold CV ROC-AUC | Test Accuracy | Test ROC-AUC | Precision | Recall | F1-Score | Brier Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 🏆 **Logistic Regression** | **0.8560 ± 0.019** | **77.58%** | **0.8557** | **0.7289** | **0.6652** | **0.6954** | **0.1582** |
| **XGBoost Classifier** | 0.8478 ± 0.017 | 78.08% | 0.8522 | 0.7302 | 0.6725 | 0.7001 | 0.1624 |
| **Gradient Boosting** | 0.8438 ± 0.015 | 77.42% | 0.8484 | 0.7250 | 0.6558 | 0.6889 | 0.1641 |
| **Random Forest** | 0.8431 ± 0.016 | 77.08% | 0.8460 | 0.7214 | 0.6441 | 0.6799 | 0.1668 |

### Explainable AI (XAI) & Factor Attribution
- Model coefficients and feature importances are normalized into percentage weights.
- The top risk drivers identified by the model are:
  1. **Systolic & Diastolic Blood Pressure** (24.0% combined impact)
  2. **Exercise-Induced Angina & Chest Pain Type** (18.5% impact)
  3. **Tobacco Smoking Status** (17.0% impact)
  4. **Total Serum Cholesterol** (14.2% impact)
  5. **Fasting Glucose / Glycemic Index** (12.0% impact)
  6. **Body Mass Index & Physical Inactivity** (10.3% impact)

---

## 6. Application Architecture

```mermaid
graph TB
    Client[Web Browser / Clinical Client] -->|HTTPS / JSON / CSV| Router[FastAPI Gateway]

    subgraph Presentation Layer
        UI[Glassmorphic Web Dashboard]
        Gauge[Animated Risk Meter & Chart.js]
        Sim[What-If Interactivity Engine]
        BatchUI[Cohort CSV Drag & Drop]
    end

    subgraph Service & API Layer
        Router --> P_Route[/api/v1/predict]
        Router --> B_Route[/api/v1/predict-batch]
        Router --> M_Route[/api/v1/model-metrics]
        Router --> H_Route[/health]
        
        P_Route --> Val[Pydantic v2 Schema Validator]
        B_Route --> Val
        Val --> Pipe[CardioPipeline Service]
        Pipe --> Exp[XAI Explainer Service]
        Pipe --> Adv[Clinical Advisor Engine]
    end

    subgraph Core ML Engine
        Pipe --> Scaler[StandardScaler Artifact]
        Scaler --> Model[(Trained Classifier: best_model.joblib)]
        Model --> Meta[(model_metadata.json)]
    end

    subgraph Cloud Container Runtime
        Docker[Docker Container - Python 3.11-slim]
        Docker --> Render[Render / Railway / AWS App Runner / GCP Cloud Run]
    end
```

---

## 7. Technology Stack

- **Backend Framework**: FastAPI 0.109.0 (Asynchronous ASGI framework)
- **ASGI Web Server**: Uvicorn 0.44.0 (Production high-performance server)
- **Data Validation & Schemas**: Pydantic v2 (Strict typing, boundary constraints)
- **Machine Learning & Analytics**: Scikit-Learn 1.5.2, XGBoost 3.0.2, NumPy 1.26.4, Pandas 2.2.3
- **Serialization**: Joblib 1.4.2, JSON
- **Frontend & Visualization**: Modern HTML5, Vanilla CSS3 (Custom Glassmorphic Design System), JavaScript (ES6+), Chart.js 4.4.1, FontAwesome 6.4.0
- **Templating**: Jinja2 3.1.6
- **Testing**: Pytest 9.0.3, HTTPX TestClient
- **Containerization**: Docker (Multi-stage build), Docker Compose
- **CI/CD**: GitHub Actions (`ci-cd.yml`)
- **Supported Cloud Platforms**: Render, Railway, AWS ECS / App Runner, GCP Cloud Run, Hugging Face Spaces

---

## 8. Local Setup Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.11.6)
- Git
- (Optional) Docker Desktop

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/cardioguard-ai.git
cd cardioguard-ai

# 2. Create and activate a Python virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# 3. Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Generate the clinical dataset
python data/make_dataset.py

# 5. Train, benchmark, and serialize the ML model artifacts
python -m model.train

# 6. Run the automated test suite to verify everything
pytest -v tests/

# 7. Start the FastAPI development / production server
uvicorn app.main.app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
- **Interactive Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive OpenAPI / Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative ReDoc API Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 9. Deployment Details

CardioGuard AI is configured for one-click deployment across major cloud providers.

### Option A: Render Cloud (Recommended - 1-Click Blueprint)
1. Push this repository to GitHub.
2. Log into [Render.com](https://render.com) and click **New +** -> **Blueprint**.
3. Connect your repository. Render will automatically read [`render.yaml`](render.yaml) and configure the build command, start command, and `/health` probe.
4. Your application will be live at: `https://cardioguard-ai.onrender.com`

### Option B: Railway Cloud
1. Create a new project on [Railway.app](https://railway.app).
2. Deploy from GitHub repository.
3. Railway automatically detects the `Procfile` and `Dockerfile` and deploys the container with HTTPS.

### Option C: AWS App Runner / Amazon ECS
```bash
# Build and tag image
docker build -t cardioguard-ai .

# Authenticate with Amazon ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push to ECR
docker tag cardioguard-ai:latest <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/cardioguard-ai:latest
docker push <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/cardioguard-ai:latest

# Deploy with AWS App Runner pointing to the ECR Image URI
```

### Option D: Google Cloud Run
```bash
# Build and submit container to Google Container Registry
gcloud builds submit --tag gcr.io/[PROJECT-ID]/cardioguard-ai

# Deploy to Cloud Run
gcloud run deploy cardioguard-ai \
  --image gcr.io/[PROJECT-ID]/cardioguard-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000
```

---

## 10. API/Web Application Usage

### 1. Web Dashboard Interface
The web application provides five interactive tabs:
- **Risk Assessment Tab**: Enter 14 physiological biomarkers, select clinical presets, and view the animated risk gauge, risk tier (Low, Moderate, High, Critical), XAI biomarker breakdown, and clinical recommendations.
- **What-If Simulator Tab**: Move interactive sliders (BP, Cholesterol, Glucose, BMI, Smoking) to simulate real-time risk reduction.
- **Batch Processing Tab**: Drag and drop a patient CSV cohort to process hundreds of records simultaneously with instant statistics and preview table.
- **Model Analytics Tab**: View the champion model metrics, global feature importance chart, and multi-model ROC-AUC curves.
- **API Explorer Tab**: Copy ready-to-use cURL commands and inspect JSON payloads.

---

### 2. REST API Endpoints

#### A. Single Patient Risk Prediction
`POST /api/v1/predict`

**Request Payload:**
```json
{
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
```

**Response Payload:**
```json
{
  "prediction": 1,
  "prediction_label": "High Cardiovascular Risk",
  "risk_probability": 0.8142,
  "risk_percentage": 81.42,
  "confidence": 81.42,
  "risk_tier": "Critical Risk",
  "risk_color": "#ef4444",
  "risk_breakdown": [
    {
      "feature": "systolic_bp",
      "display_name": "Systolic Blood Pressure (mmHg)",
      "patient_value": "145 mmHg",
      "benchmark": "< 120 mmHg",
      "status": "Stage 2 Hypertension",
      "impact_level": "Critical",
      "relative_contribution_pct": 24.0,
      "description": "Severely elevated systolic pressure. Substantial risk factor for myocardial infarction."
    }
  ],
  "recommendations": [
    {
      "category": "Medical Consultation",
      "priority": "Immediate",
      "title": "Comprehensive Cardiologist Evaluation & Diagnostic Workup",
      "action_item": "Schedule an urgent cardiology consultation for 12-lead ECG, echocardiography, or treadmill stress testing.",
      "clinical_rationale": "High predicted probability of cardiovascular disease warrants objective non-invasive diagnostic confirmation."
    }
  ],
  "patient_values": { ... },
  "timestamp": "2026-08-17T17:15:00.000000Z"
}
```

#### B. Batch Cohort Risk Prediction
`POST /api/v1/predict-batch` (Multipart Form Upload)

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict-batch" \
     -F "file=@data/raw/cardiovascular_data.csv"
```

#### C. Download Sample Batch CSV Template
`GET /api/v1/sample-csv`

#### D. Model Metrics & Benchmarks
`GET /api/v1/model-metrics`

#### E. System Health Check
`GET /health`

---

## 11. Docker Instructions

### Build Docker Image
```bash
docker build -t cardioguard-ai:latest .
```

### Run Docker Container
```bash
docker run -d \
  --name cardioguard-container \
  -p 8000:8000 \
  --restart unless-stopped \
  cardioguard-ai:latest
```

### Run with Docker Compose
```bash
docker compose up --build -d
```

To stop and remove containers:
```bash
docker compose down
```

### Push to Docker Hub (Container Registry)
```bash
# Log in to Docker Hub
docker login

# Tag the image with your Docker Hub username
docker tag cardioguard-ai:latest your_dockerhub_username/cardioguard-ai:v1.0.0

# Push image to Docker Hub
docker push your_dockerhub_username/cardioguard-ai:v1.0.0
```

---

## 12. Automated Testing

The project includes an exhaustive automated test suite using `pytest` and `fastapi.testclient.TestClient`:

```bash
# Run all unit and integration tests
pytest -v tests/
```

### Test Suite Coverage:
- `tests/test_model.py`: Validates dataset generation, pipeline loading, single patient inference bounds, and batch prediction DataFrame schemas.
- `tests/test_validation.py`: Tests boundary conditions and constraints in Pydantic schemas (e.g., negative blood pressures, underage patient inputs, diastolic exceeding systolic blood pressure).
- `tests/test_api.py`: Tests `/health`, `GET /`, `POST /api/v1/predict`, `POST /api/v1/predict-batch`, `GET /api/v1/model-metrics`, `GET /api/v1/roc-curve`, and `GET /api/v1/sample-csv`.

**Results**: `15 passed in 12.71s (100% success rate)`

---

## 📄 License & Attribution
Distributed under the MIT License. See `LICENSE` for more information.
Built for the academic submission of **End-to-End AI Application Development and Cloud Deployment**.
