"""
Model Training and Benchmarking Pipeline for CardioGuard AI.
Trains multiple ML classifiers (Logistic Regression, Random Forest,
Gradient Boosting, XGBoost), performs Stratified Cross-Validation,
selects the best model, extracts feature importance, and serializes artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

from model.pipeline import FEATURE_COLUMNS, FEATURE_DISPLAY_NAMES
from model.evaluate import evaluate_classifier, save_evaluation_plots

def run_training_pipeline(data_path: str, artifacts_dir: str):
    print("=" * 65)
    print("  CARDIOGUARD AI - MODEL TRAINING & BENCHMARKING PIPELINE")
    print("=" * 65)

    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Load Dataset
    print(f"\n[1/5] Loading cardiovascular dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"      Total records: {len(df):,} | Features: {len(FEATURE_COLUMNS)}")

    X = df[FEATURE_COLUMNS]
    y = df["cardiovascular_disease"]

    # 2. Train / Test Split
    print("\n[2/5] Splitting dataset (80% Train, 20% Holdout Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Train Set: {len(X_train)} samples | Test Set: {len(X_test)} samples")

    # 3. Feature Standardization
    print("\n[3/5] Standardizing numerical features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Model Candidates Definition
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_split=6, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42
        ),
        "XGBoost Classifier": xgb.XGBClassifier(
            n_estimators=140,
            learning_rate=0.07,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42
        )
    }

    # 5. Cross-Validation & Model Evaluation
    print("\n[4/5] Running 5-Fold Stratified Cross-Validation and Benchmarking...")
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_results = {}
    fitted_models = {}

    for name, clf in models.items():
        print(f"\n  >> Evaluating: {name}")
        # Cross-validation score
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=kfold, scoring="roc_auc")
        print(f"     5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Fit on full training split
        clf.fit(X_train_scaled, y_train)
        fitted_models[name] = clf

        # Evaluate on test set
        eval_metrics = evaluate_classifier(clf, X_test_scaled, y_test.values, model_name=name)
        eval_metrics["cv_roc_auc_mean"] = round(float(cv_scores.mean()), 4)
        eval_metrics["cv_roc_auc_std"] = round(float(cv_scores.std()), 4)
        benchmark_results[name] = eval_metrics

        print(f"     Test Accuracy: {eval_metrics['accuracy']:.4f} | ROC-AUC: {eval_metrics['roc_auc']:.4f} | F1: {eval_metrics['f1_score']:.4f}")

    # Select Best Model based on Test ROC-AUC
    best_model_name = max(benchmark_results, key=lambda k: benchmark_results[k]["roc_auc"])
    best_model = fitted_models[best_model_name]
    best_metrics = benchmark_results[best_model_name]

    print("\n" + "-" * 65)
    print(f"  [CHAMPION MODEL SELECTED] {best_model_name}")
    print(f"     Test ROC-AUC:  {best_metrics['roc_auc']:.4f}")
    print(f"     Test Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"     Test F1 Score: {best_metrics['f1_score']:.4f}")
    print("-" * 65)

    # 6. Extract Feature Importances
    feature_importances = {}
    if hasattr(best_model, "feature_importances_"):
        raw_importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        raw_importances = np.abs(best_model.coef_[0])
    else:
        raw_importances = np.ones(len(FEATURE_COLUMNS))

    # Normalize to 100%
    normalized_importances = (raw_importances / np.sum(raw_importances)) * 100
    for col, imp in zip(FEATURE_COLUMNS, normalized_importances):
        feature_importances[col] = {
            "display_name": FEATURE_DISPLAY_NAMES.get(col, col),
            "importance_pct": round(float(imp), 2)
        }

    # Sort descending
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1]["importance_pct"], reverse=True)
    sorted_feature_dict = {k: v for k, v in sorted_features}

    # 7. Save Evaluation Plots
    print("\n[5/5] Generating evaluation curves & saving artifacts...")
    save_evaluation_plots(fitted_models, X_test_scaled, y_test.values, artifacts_dir)

    # 8. Serialize Best Model & Scaler
    model_path = os.path.join(artifacts_dir, "best_model.joblib")
    scaler_path = os.path.join(artifacts_dir, "scaler.joblib")
    meta_path = os.path.join(artifacts_dir, "model_metadata.json")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_name": best_model_name,
        "model_type": str(type(best_model).__name__),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "dataset_size": len(df),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": FEATURE_COLUMNS,
        "feature_importances": sorted_feature_dict,
        "best_model_metrics": best_metrics,
        "all_model_benchmarks": benchmark_results,
        "dataset_statistics": {
            "mean_age": round(float(df["age"].mean()), 1),
            "mean_systolic_bp": round(float(df["systolic_bp"].mean()), 1),
            "mean_cholesterol": round(float(df["cholesterol"].mean()), 1),
            "mean_glucose": round(float(df["glucose"].mean()), 1),
            "positive_prevalence_pct": round(float(df["cardiovascular_disease"].mean() * 100), 2)
        }
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Success] Artifacts serialized successfully:")
    print(f"  - Model:     {model_path}")
    print(f"  - Scaler:    {scaler_path}")
    print(f"  - Metadata:  {meta_path}")
    print("=" * 65)

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_file = os.path.join(base_dir, "data", "raw", "cardiovascular_data.csv")
    artifacts_dir = os.path.join(base_dir, "model", "artifacts")
    run_training_pipeline(data_file, artifacts_dir)
