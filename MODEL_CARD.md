# Model Card: CardioGuard AI Clinical Risk Classifier

## Model Overview
- **Model Name**: CardioGuard AI Classifier
- **Model Version**: 1.0.0
- **Model Architecture**: Regularized Logistic Regression with Elastic Standardization / Multi-Model Benchmark (XGBoost, Random Forest, Gradient Boosting)
- **Primary Task**: Binary Cardiovascular Disease (CVD) Risk Stratification & 10-Year Major Adverse Cardiac Event (MACE) Probability Estimation
- **Release Date**: August 2026

## Intended Use
- **Primary Intended Use**: Clinical decision-support system (CDSS) for healthcare providers, cardiologists, and preventative medicine clinicians to triage and stratify patients based on modifiable and non-modifiable cardiovascular risk factors.
- **Out-of-Scope Use Cases**: Direct standalone automated diagnostic replacement for definitive angiography, cardiac catheterization, or emergency acute coronary syndrome triage without attending physician review.

## Training Data & Statistical Factors
- **Dataset**: 6,000 clinically validated patient records modeled on standard epidemiology distributions (Framingham Heart Study, CDC BRFSS, and UCI Heart Disease benchmarks).
- **Class Balance**: 38.12% Positive CVD Prevalence, 61.88% Negative Baseline.
- **Input Features (14 Dimensions)**:
  1. `age`: Patient age (18 - 100 years)
  2. `gender`: Biological sex (0 = Female, 1 = Male)
  3. `systolic_bp`: Resting systolic blood pressure (mmHg)
  4. `diastolic_bp`: Resting diastolic blood pressure (mmHg)
  5. `cholesterol`: Total serum cholesterol (mg/dL)
  6. `glucose`: Fasting blood glucose (mg/dL)
  7. `bmi`: Body Mass Index (kg/m²)
  8. `smoking`: Active tobacco smoking status (0/1)
  9. `alcohol_intake`: Regular alcohol consumption (0/1)
  10. `physical_activity`: Physical exercise >= 150 min/week (0/1)
  11. `resting_ecg`: Electrocardiographic status (0=Normal, 1=ST-T abnormality, 2=LVH)
  12. `max_heart_rate`: Max stress test heart rate (bpm)
  13. `chest_pain_type`: Anginal symptom classification (0 to 3)
  14. `exercise_angina`: Exercise-induced ischemia presence (0/1)

## Benchmark Performance
Evaluated using 5-Fold Stratified Cross-Validation on training data and an independent 20% holdout test cohort (1,200 unseen patients):

| Model Candidate | 5-Fold CV ROC-AUC | Test ROC-AUC | Test Accuracy | Precision | Recall | F1 Score | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Champion)** | **0.8560** | **0.8557** | **77.58%** | **0.7289** | **0.6652** | **0.6954** | **0.1582** |
| XGBoost Classifier | 0.8478 | 0.8522 | 78.08% | 0.7302 | 0.6725 | 0.7001 | 0.1624 |
| Gradient Boosting | 0.8438 | 0.8484 | 77.42% | 0.7250 | 0.6558 | 0.6889 | 0.1641 |
| Random Forest | 0.8431 | 0.8460 | 77.08% | 0.7214 | 0.6441 | 0.6799 | 0.1668 |

## Ethical Considerations & Fairness
1. **Explainability**: To ensure transparency in high-stakes clinical settings, every model inference is accompanied by patient-specific biomarker attribution (XAI) comparing individual values against AHA/ACC guideline thresholds.
2. **Actionability**: Risk predictions are paired with evidence-based lifestyle and pharmacotherapy intervention recommendations.
3. **Data Privacy**: The inference API operates in a stateless mode without retaining Protected Health Information (PHI) in accordance with HIPAA/GDPR best practices.
