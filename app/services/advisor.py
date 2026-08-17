"""
Evidence-based Clinical & Lifestyle Guidance Engine for CardioGuard AI.
Synthesizes clinical guideline recommendations (AHA, ACC, ESC) customized
to individual patient risk profiles and specific modifiable risk factors.
"""

from typing import Dict, List, Any
from app.schemas import ClinicalRecommendation

class AdvisorService:
    @staticmethod
    def generate_recommendations(
        patient_values: Dict[str, Any],
        risk_tier: str,
        risk_probability: float
    ) -> List[ClinicalRecommendation]:
        recs = []

        # 1. Tier-based general clinical action
        if risk_tier in ["High Risk", "Critical Risk"]:
            recs.append(ClinicalRecommendation(
                category="Medical Consultation",
                priority="Immediate",
                title="Comprehensive Cardiologist Evaluation & Diagnostic Workup",
                action_item="Schedule an urgent cardiology consultation for 12-lead ECG, echocardiography, or treadmill stress testing.",
                clinical_rationale="High predicted probability of cardiovascular disease warrants objective non-invasive diagnostic confirmation."
            ))
        elif risk_tier == "Moderate Risk":
            recs.append(ClinicalRecommendation(
                category="Medical Consultation",
                priority="High",
                title="Primary Care Clinical Review & 6-Month Monitoring",
                action_item="Arrange a routine clinical review to establish a baseline cardiovascular prevention plan.",
                clinical_rationale="Early risk stratification enables aggressive risk-factor modification before irreversible vascular remodeling occurs."
            ))
        else:
            recs.append(ClinicalRecommendation(
                category="Medical Consultation",
                priority="Medium",
                title="Annual Cardiovascular Wellness Screening",
                action_item="Maintain routine annual health evaluations including lipid profile and blood pressure checks.",
                clinical_rationale="Sustaining low baseline risk ensures ongoing long-term cardiovascular vitality."
            ))

        # 2. Blood Pressure Interventions
        sys_bp = patient_values["systolic_bp"]
        dia_bp = patient_values["diastolic_bp"]
        if sys_bp >= 140 or dia_bp >= 90:
            recs.append(ClinicalRecommendation(
                category="Hemodynamic Optimization",
                priority="High",
                title="Antihypertensive Strategy & DASH Dietary Protocol",
                action_item="Adopt the DASH dietary pattern (< 2,300 mg sodium/day) and evaluate eligibility for antihypertensive therapy (e.g., ACEi/ARB or CCB).",
                clinical_rationale="Every 10 mmHg reduction in systolic blood pressure significantly reduces major cardiovascular events by ~20%."
            ))
        elif sys_bp >= 130:
            recs.append(ClinicalRecommendation(
                category="Hemodynamic Optimization",
                priority="Medium",
                title="Sodium Reduction & Home BP Telemonitoring",
                action_item="Monitor resting blood pressure bi-weekly; restrict dietary sodium and enhance dietary potassium intake.",
                clinical_rationale="Pre-hypertension management prevents progression to established arterial hypertension."
            ))

        # 3. Lipid / Cholesterol Management
        chol = patient_values["cholesterol"]
        if chol >= 240:
            recs.append(ClinicalRecommendation(
                category="Lipid Management",
                priority="High",
                title="Statin Assessment & Atherogenic Lipid Reduction",
                action_item="Consult physician regarding lipid-lowering therapy (statin/ezetimibe) and restrict saturated fats to < 7% of daily caloric intake.",
                clinical_rationale="Lowering low-density lipoprotein cholesterol (LDL-C) stabilizes arterial plaques and prevents acute coronary syndromes."
            ))
        elif chol >= 200:
            recs.append(ClinicalRecommendation(
                category="Lipid Management",
                priority="Medium",
                title="Soluble Fiber & Plant Sterol Enrichment",
                action_item="Increase intake of soluble fiber (oats, legumes, psyllium) and foods rich in omega-3 polyunsaturated fatty acids.",
                clinical_rationale="Dietary fiber binds bile acids in the gut, reducing circulating total cholesterol."
            ))

        # 4. Tobacco Cessation
        if patient_values.get("smoking") == 1:
            recs.append(ClinicalRecommendation(
                category="Lifestyle Modification",
                priority="Immediate",
                title="Structured Tobacco Cessation & Nicotine Replacement Therapy",
                action_item="Enroll in a structured smoking cessation program; consider pharmacotherapy (varenicline, bupropion, or NRT).",
                clinical_rationale="Cardiovascular risk drops by nearly 50% within one year of complete smoking cessation."
            ))

        # 5. Glycemic Control
        gluc = patient_values["glucose"]
        if gluc >= 126:
            recs.append(ClinicalRecommendation(
                category="Metabolic Management",
                priority="High",
                title="Glycemic Optimization & HbA1c Target Assessment",
                action_item="Obtain laboratory HbA1c test and review medical management for diabetes mellitus.",
                clinical_rationale="Hyperglycemia induces oxidative vascular injury; glycemic control prevents macrovascular complications."
            ))

        # 6. Physical Activity & Weight
        bmi = patient_values["bmi"]
        act = patient_values["physical_activity"]
        if act == 0 or bmi >= 25.0:
            recs.append(ClinicalRecommendation(
                category="Exercise & Weight",
                priority="Medium",
                title="Structured Aerobic Exercise & Body Composition Target",
                action_item="Engage in 150 minutes/week of moderate-intensity aerobic exercise (e.g., brisk walking, cycling) with resistance training 2x/week.",
                clinical_rationale="Regular exercise improves insulin sensitivity, increases endothelial nitric oxide, and reduces arterial stiffness."
            ))

        return recs
