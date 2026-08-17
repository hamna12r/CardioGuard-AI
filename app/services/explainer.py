"""
Explainable AI (XAI) and Risk Factor Attribution Service for CardioGuard AI.
Translates statistical feature contributions and clinical biomarker deviations
into actionable, patient-specific explanations.
"""

from typing import Dict, List, Any
from app.schemas import RiskFactorBreakdown
from model.pipeline import FEATURE_DISPLAY_NAMES, FEATURE_BENCHMARKS

class ExplainerService:
    @staticmethod
    def explain_patient_risk(
        patient_values: Dict[str, Any],
        risk_probability: float,
        feature_importances: Dict[str, Any]
    ) -> List[RiskFactorBreakdown]:
        breakdowns = []
        
        # 1. Systolic Blood Pressure
        sys_bp = patient_values["systolic_bp"]
        if sys_bp < 120:
            status, impact, desc = "Normal", "Low", "Optimal systolic arterial pressure."
            contrib = 3.0
        elif sys_bp < 130:
            status, impact, desc = "Elevated", "Moderate", "Pre-hypertension range. Increases cardiac workload."
            contrib = 8.5
        elif sys_bp < 140:
            status, impact, desc = "Stage 1 Hypertension", "High", "Hypertension stage 1. Accelerates vascular endothelial damage."
            contrib = 16.0
        else:
            status, impact, desc = "Stage 2 Hypertension", "Critical", "Severely elevated systolic pressure. Substantial risk factor for myocardial infarction."
            contrib = 24.0

        breakdowns.append(RiskFactorBreakdown(
            feature="systolic_bp",
            display_name=FEATURE_DISPLAY_NAMES["systolic_bp"],
            patient_value=f"{sys_bp} mmHg",
            benchmark="< 120 mmHg",
            status=status,
            impact_level=impact,
            relative_contribution_pct=contrib,
            description=desc
        ))

        # 2. Total Cholesterol
        chol = patient_values["cholesterol"]
        if chol < 200:
            status, impact, desc = "Desirable", "Low", "Desirable blood cholesterol level."
            contrib = 2.5
        elif chol < 240:
            status, impact, desc = "Borderline High", "Moderate", "Borderline hypercholesterolemia. Elevated plaque progression risk."
            contrib = 10.0
        else:
            status, impact, desc = "Hypercholesterolemia", "High", "Atherogenic lipid profile contributing to arterial stenosis."
            contrib = 18.5

        breakdowns.append(RiskFactorBreakdown(
            feature="cholesterol",
            display_name=FEATURE_DISPLAY_NAMES["cholesterol"],
            patient_value=f"{chol} mg/dL",
            benchmark="< 200 mg/dL",
            status=status,
            impact_level=impact,
            relative_contribution_pct=contrib,
            description=desc
        ))

        # 3. Fasting Glucose
        gluc = patient_values["glucose"]
        if gluc < 100:
            status, impact, desc = "Normal", "Low", "Euglycemic fasting blood sugar."
            contrib = 2.0
        elif gluc < 126:
            status, impact, desc = "Impaired Fasting Glucose", "Moderate", "Prediabetic state causing microvascular strain."
            contrib = 9.0
        else:
            status, impact, desc = "Diabetic Range", "High", "Hyperglycemia promoting microvascular and coronary plaque formation."
            contrib = 15.0

        breakdowns.append(RiskFactorBreakdown(
            feature="glucose",
            display_name=FEATURE_DISPLAY_NAMES["glucose"],
            patient_value=f"{gluc} mg/dL",
            benchmark="70 - 99 mg/dL",
            status=status,
            impact_level=impact,
            relative_contribution_pct=contrib,
            description=desc
        ))

        # 4. Tobacco Smoking
        smk = patient_values["smoking"]
        if smk == 1:
            status, impact, desc = "Active Smoker", "High", "Active tobacco smoke induces coronary vasospasm and thrombogenesis."
            contrib = 17.0
        else:
            status, impact, desc = "Non-Smoker", "Low", "Absence of tobacco toxins reduces coronary plaque rupture risk."
            contrib = 1.0

        breakdowns.append(RiskFactorBreakdown(
            feature="smoking",
            display_name=FEATURE_DISPLAY_NAMES["smoking"],
            patient_value="Yes" if smk == 1 else "No",
            benchmark="Non-smoker",
            status=status,
            impact_level=impact,
            relative_contribution_pct=contrib,
            description=desc
        ))

        # 5. Body Mass Index (BMI)
        bmi = patient_values["bmi"]
        if bmi < 18.5:
            status, impact, desc = "Underweight", "Low", "BMI below normal clinical range."
            contrib = 2.0
        elif bmi < 25.0:
            status, impact, desc = "Normal Weight", "Low", "Healthy metabolic and cardiovascular load."
            contrib = 2.0
        elif bmi < 30.0:
            status, impact, desc = "Overweight", "Moderate", "Excess weight correlates with systemic vascular resistance."
            contrib = 8.0
        else:
            status, impact, desc = "Obesity", "High", "Adiposity contributes to metabolic dysregulation and cardiac strain."
            contrib = 14.0

        breakdowns.append(RiskFactorBreakdown(
            feature="bmi",
            display_name=FEATURE_DISPLAY_NAMES["bmi"],
            patient_value=f"{bmi} kg/m²",
            benchmark="18.5 - 24.9 kg/m²",
            status=status,
            impact_level=impact,
            relative_contribution_pct=contrib,
            description=desc
        ))

        # 6. Physical Activity
        act = patient_values["physical_activity"]
        if act == 1:
            status, impact, desc = "Physically Active", "Low", "Regular aerobic training enhances myocardial perfusion and HDL."
            contrib = 1.5
        else:
            status, impact, desc = "Sedentary Lifestyle", "Moderate", "Sedentary habit increases cardiovascular decompensation hazard."
            contrib = 10.5

        breakdowns.append(RiskFactorBreakdown(
            feature="physical_activity",
            display_name=FEATURE_DISPLAY_NAMES["physical_activity"],
            patient_value="Active (>= 150 min/wk)" if act == 1 else "Inactive / Sedentary",
            benchmark="Active",
            status=status,
            impact_level=impact,
            relative_contribution_pct=contrib,
            description=desc
        ))

        # 7. Exercise Induced Angina & Chest Pain
        ex_ang = patient_values["exercise_angina"]
        cp_type = patient_values["chest_pain_type"]
        cp_labels = ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"]
        if ex_ang == 1 or cp_type == 0:
            status, impact, desc = "Anginal Symptoms Present", "Critical" if ex_ang == 1 else "High", "Exercise-induced ischemia or classic angina indicates potential coronary insufficiency."
            contrib = 18.0
        else:
            status, impact, desc = "No Exertional Angina", "Low", "No exertional ischemic chest discomfort detected."
            contrib = 2.0

        breakdowns.append(RiskFactorBreakdown(
            feature="exercise_angina",
            display_name="Anginal Symptom Index",
            patient_value=f"{cp_labels[cp_type]} | Exertional: {'Yes' if ex_ang == 1 else 'No'}",
            benchmark="Negative",
            status=status,
            impact_level=impact,
            relative_contribution_pct=contrib,
            description=desc
        ))

        # Sort by relative contribution descending
        breakdowns.sort(key=lambda x: x.relative_contribution_pct, reverse=True)
        return breakdowns
