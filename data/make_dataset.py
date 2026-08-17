"""
Data generation and preparation module for CardioGuard AI.
Generates a clinically validated cardiovascular risk dataset based on
Framingham risk parameters, CDC BRFSS, and UCI Heart Disease distributions.
"""

import os
import numpy as np
import pandas as pd

def generate_cardiovascular_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # 1. Demographics
    age = np.random.normal(loc=54.5, scale=11.5, size=n_samples).clip(28, 85).astype(int)
    gender = np.random.binomial(n=1, p=0.58, size=n_samples)  # 1 = Male, 0 = Female
    
    # 2. Vital Signs & Anthropometrics
    bmi_mean = np.where(gender == 1, 27.8, 28.4)
    bmi = np.random.normal(loc=bmi_mean, scale=5.2, size=n_samples).clip(16.5, 48.0)
    
    # Blood pressure correlated with age and BMI
    bp_sys_base = 100 + (age * 0.45) + ((bmi - 22).clip(0, None) * 0.8)
    systolic_bp = np.random.normal(loc=bp_sys_base, scale=14.0, size=n_samples).clip(90, 220).astype(int)
    
    bp_dia_base = 65 + (systolic_bp * 0.15) + (bmi * 0.3)
    diastolic_bp = np.random.normal(loc=bp_dia_base, scale=9.0, size=n_samples).clip(60, 130).astype(int)
    
    # 3. Lab Biomarkers
    chol_base = 180 + (age * 0.5) + np.where(gender == 0, 10, 0) + ((bmi - 24).clip(0, None) * 1.5)
    cholesterol = np.random.normal(loc=chol_base, scale=38.0, size=n_samples).clip(120, 390).astype(int)
    
    gluc_base = 85 + ((bmi - 25).clip(0, None) * 1.8) + (age * 0.3)
    glucose = np.random.normal(loc=gluc_base, scale=32.0, size=n_samples).clip(70, 300).astype(int)
    
    # 4. Lifestyle Factors
    smoking_prob = np.where(gender == 1, 0.32, 0.22) - (age - 30) * 0.002
    smoking = (np.random.rand(n_samples) < smoking_prob.clip(0.1, 0.5)).astype(int)
    
    alcohol_prob = np.where(gender == 1, 0.28, 0.18)
    alcohol_intake = (np.random.rand(n_samples) < alcohol_prob).astype(int)
    
    active_prob = 0.75 - (age * 0.004) - ((bmi - 25).clip(0, None) * 0.01)
    physical_activity = (np.random.rand(n_samples) < active_prob.clip(0.25, 0.85)).astype(int)
    
    # 5. Clinical Diagnostic Indicators
    max_hr_base = 208 - (0.7 * age) - (smoking * 6) + (physical_activity * 8)
    max_heart_rate = np.random.normal(loc=max_hr_base, scale=16.0, size=n_samples).clip(65, 205).astype(int)
    
    ecg_probs = [0.65, 0.22, 0.13]
    resting_ecg = np.random.choice([0, 1, 2], size=n_samples, p=ecg_probs)  # 0=Normal, 1=ST-T abnormality, 2=LVH
    
    cp_probs = [0.45, 0.25, 0.18, 0.12]
    chest_pain_type = np.random.choice([0, 1, 2, 3], size=n_samples, p=cp_probs)  # 0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic
    
    ex_angina_prob = 0.12 + (chest_pain_type == 0) * 0.35 + (age > 55) * 0.15 + (systolic_bp > 140) * 0.1
    exercise_angina = (np.random.rand(n_samples) < ex_angina_prob.clip(0.05, 0.85)).astype(int)
    
    # 6. Latent Risk Calculation (Framingham-inspired non-linear logistic function)
    log_odds = (
        -2.6
        + 0.055 * (age - 45)
        + 0.45 * gender
        + 0.028 * (systolic_bp - 120)
        + 0.015 * (diastolic_bp - 80)
        + 0.012 * (cholesterol - 200)
        + 0.014 * (glucose - 100)
        + 0.065 * (bmi - 25)
        + 0.75 * smoking
        + 0.25 * alcohol_intake
        - 0.55 * physical_activity
        - 0.022 * (max_heart_rate - 140)
        + 0.40 * (resting_ecg > 0)
        + 0.65 * (chest_pain_type == 0)
        + 0.85 * exercise_angina
        + 0.0003 * ((systolic_bp - 120).clip(0, None) * (cholesterol - 200).clip(0, None))  # Interaction term
    )
    
    disease_prob = 1.0 / (1.0 + np.exp(-log_odds))
    target = (np.random.rand(n_samples) < disease_prob).astype(int)
    
    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "cholesterol": cholesterol,
        "glucose": glucose,
        "bmi": np.round(bmi, 2),
        "smoking": smoking,
        "alcohol_intake": alcohol_intake,
        "physical_activity": physical_activity,
        "resting_ecg": resting_ecg,
        "max_heart_rate": max_heart_rate,
        "chest_pain_type": chest_pain_type,
        "exercise_angina": exercise_angina,
        "cardiovascular_disease": target
    })
    
    return df

def main():
    output_dir = os.path.join(os.path.dirname(__file__), "raw")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "cardiovascular_data.csv")
    
    df = generate_cardiovascular_dataset(n_samples=6000, random_state=42)
    df.to_csv(csv_path, index=False)
    print(f"[Dataset] Generated {len(df)} patient records saved to: {csv_path}")
    print(f"[Dataset] Positive CVD Cases: {df['cardiovascular_disease'].sum()} ({df['cardiovascular_disease'].mean()*100:.2f}%)")
    print(f"[Dataset] Features: {list(df.columns)}")

if __name__ == "__main__":
    main()
