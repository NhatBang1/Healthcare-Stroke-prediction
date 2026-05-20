import os
import sys
import json
import warnings
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add current folder to path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from data_preprocessing import (
    ROOT_DIR, CALORIES_PATH, STROKE_PATH, FEATURE_SETS,
    deduplicate_patients, build_calories_model,
    build_preprocessing_artifacts, preprocess_stroke_data
)
from model_pipeline import (
    run_evaluation_review_loop, generate_synthetic_data, export_model_onnx
)
from inference import StrokePredictor

warnings.filterwarnings("ignore")

def main():
    print("====================================================")
    print("STARTING END-TO-END STROKE prediction ML PIPELINE")
    print("====================================================\n")

    # 1. Load raw data
    print(f"Loading calories data from: {CALORIES_PATH}")
    df_cal = pd.read_csv(CALORIES_PATH)
    print(f"Loading stroke data from: {STROKE_PATH}")
    df_str_raw = pd.read_csv(STROKE_PATH)

    # 2. Build calories model
    print("\n[Step 2] Training Calories Regressor (XGBRegressor)...")
    calories_model = build_calories_model(df_cal)
    
    # Evaluate Calories model performance
    df_cal_eval = df_cal.copy()
    from data_preprocessing import normalize_gender, numeric_medians, apply_numeric_fill, CALORIES_FEATURE_COLUMNS
    df_cal_eval["Gender"] = normalize_gender(df_cal_eval["Gender"])
    df_cal_eval["BMI"] = df_cal_eval["Weight"] / ((df_cal_eval["Height"] / 100) ** 2)
    medians = numeric_medians(df_cal_eval, CALORIES_FEATURE_COLUMNS + ["Calories"])
    df_cal_eval = apply_numeric_fill(df_cal_eval, medians)
    pred_calories = calories_model.predict(df_cal_eval[CALORIES_FEATURE_COLUMNS])
    mae_cal = np.mean(np.abs(df_cal_eval["Calories"] - pred_calories))
    print(f"Calories Model MAE on self-check: {mae_cal:.4f}")

    # 3. Deduplicate Stroke Patients by patient_id
    print("\n[Step 3] Deduplicating Stroke Patient Records...")
    df_str = deduplicate_patients(df_str_raw)

    # 4. Train-Validation-Test Split on Patient Level
    print("\n[Step 4] Splitting Patients into Train, Validation, and Test sets (70/15/15)...")
    train_df_raw, temp_df_raw = train_test_split(
        df_str, test_size=0.30, stratify=df_str["stroke_target"], random_state=42
    )
    val_df_raw, test_df_raw = train_test_split(
        temp_df_raw, test_size=0.50, stratify=temp_df_raw["stroke_target"], random_state=42
    )
    print(f"Train size: {len(train_df_raw)}, Val size: {len(val_df_raw)}, Test size: {len(test_df_raw)}")

    # 5. Build Preprocessing Artifacts based ONLY on training data
    print("\n[Step 5] Building Preprocessing Imputation Artifacts...")
    preprocessing_artifacts = build_preprocessing_artifacts(train_df_raw)

    # 6. Apply preprocessing to Train, Val, and Test
    print("\n[Step 6] Preprocessing and feature engineering datasets...")
    train_processed, y_train, preprocessing_artifacts = preprocess_stroke_data(
        train_df_raw, calories_model, preprocessing_artifacts
    )
    val_processed, y_val, _ = preprocess_stroke_data(
        val_df_raw, calories_model, preprocessing_artifacts
    )
    test_processed, y_test, _ = preprocess_stroke_data(
        test_df_raw, calories_model, preprocessing_artifacts
    )
    
    # Store list of dummy encoded features
    dummy_features = [col for col in train_processed.columns if col not in ["patient_id", "stroke_target"]]
    preprocessing_artifacts["encoded_columns"] = dummy_features

    # 7. Generate Synthetic Dataset
    print("\n[Step 7] Generating raw synthetic patient dataset for evaluation review loop...")
    df_synth_raw = generate_synthetic_data(train_df_raw, num_samples=1500, random_state=42)
    
    # Preprocess raw synthetic dataset using the training artifacts
    synth_processed, y_synth, _ = preprocess_stroke_data(
        df_synth_raw, calories_model, preprocessing_artifacts
    )

    # 8. Run Model Optimization and Review Loop across multiple Feature Sets
    print("\n[Step 8] Running automated model search and evaluation review loops...")
    
    # We will run the review loop for the best-performing feature subsets
    best_overall_auc = 0
    best_overall_model = None
    best_overall_name = ""
    best_overall_features = []
    best_overall_threshold = 0.5
    best_overall_metrics = {}
    best_feature_set_name = ""

    # Test two of the strongest feature sets
    candidate_feature_sets = {
        "clinical_only": FEATURE_SETS["clinical_only"],
        "baseline": FEATURE_SETS["baseline"]
    }

    for feat_set_name, feat_list in candidate_feature_sets.items():
        print(f"\nEvaluating Feature Set: {feat_set_name}...")
        
        # Filter available features
        selected_features = [col for col in feat_list if col in train_processed.columns]
        
        X_train_f = train_processed[selected_features]
        X_val_f = val_processed[selected_features]
        X_test_f = test_processed[selected_features]
        X_synth_f = synth_processed[selected_features]
        
        model, name, threshold, metrics = run_evaluation_review_loop(
            X_train_f, y_train, X_val_f, y_val, X_test_f, y_test, X_synth_f, y_synth, selected_features
        )
        
        # Select overall best model based on synthetic + real test average AUC-ROC
        avg_auc = (metrics["real"]["auc_roc"] + metrics["synthetic"]["auc_roc"]) / 2
        if avg_auc > best_overall_auc:
            best_overall_auc = avg_auc
            best_overall_model = model
            best_overall_name = name
            best_overall_features = selected_features
            best_overall_threshold = threshold
            best_overall_metrics = metrics
            best_feature_set_name = feat_set_name

    print("\n==========================================")
    print("FINAL SELECTION SUMMARY")
    print("==========================================")
    print(f"Best Feature Set: {best_feature_set_name}")
    print(f"Best Model Name: {best_overall_name}")
    print(f"Best Decision Threshold: {best_overall_threshold:.3f}")
    print(f"Real Test AUC-ROC: {best_overall_metrics['real']['auc_roc']:.4f} | F1: {best_overall_metrics['real']['f1']:.4f}")
    print(f"Synthetic Test AUC-ROC: {best_overall_metrics['synthetic']['auc_roc']:.4f} | F1: {best_overall_metrics['synthetic']['f1']:.4f}")

    # 9. Save Artifacts
    artifacts_dir = ROOT_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Step 9] Saving final artifacts to: {artifacts_dir}")
    
    # Save models
    joblib.dump(calories_model, artifacts_dir / "calories_model.joblib")
    joblib.dump(best_overall_model, artifacts_dir / "stroke_model.joblib")
    
    # Save preprocessing metadata
    metadata = {
        "numeric_fill": preprocessing_artifacts["numeric_fill"],
        "categorical_fill": preprocessing_artifacts["categorical_fill"],
        "categorical_columns": preprocessing_artifacts["categorical_columns"],
        "feature_columns": best_overall_features,
        "decision_threshold": best_overall_threshold,
        "best_model_name": best_overall_name,
        "best_feature_set": best_feature_set_name,
        "metrics": best_overall_metrics
    }
    
    with open(artifacts_dir / "preprocess_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print("Artifacts saved successfully.")

    # 10. Export to ONNX
    onnx_path = artifacts_dir / "stroke_model.onnx"
    print(f"\n[Step 10] Exporting final model to ONNX format...")
    export_model_onnx(best_overall_model, len(best_overall_features), onnx_path)

    # 11. Run Demo Inference
    print("\n[Step 11] Performing demo inference to verify StrokePredictor class...")
    predictor = StrokePredictor(artifacts_dir)
    
    demo_patients = [
        {
            "patient_id": 99991,
            "gender": "Female",
            "age": 67.0,
            "hypertension": 0,
            "heart_disease": 1,
            "avg_glucose_level": 228.69,
            "bmi": 36.6,
            "avg_RestingBP": 130.0,
            "heart_cholesterol": 240.0,
            "avg_MaxHR": 160.0,
            "avg_Oldpeak": 1.2,
            "heart_disease_rate": 0.35,
            "ever_married": "Yes",
            "work_type": "Private",
            "Residence_type": "Urban",
            "smoking_status": "formerly smoked"
        },
        {
            "patient_id": 99992,
            "gender": "Male",
            "age": 30.0,
            "hypertension": 0,
            "heart_disease": 0,
            "avg_glucose_level": 82.5,
            "bmi": 24.1,
            "avg_RestingBP": 115.0,
            "heart_cholesterol": 180.0,
            "avg_MaxHR": 175.0,
            "avg_Oldpeak": 0.0,
            "heart_disease_rate": 0.0,
            "ever_married": "No",
            "work_type": "Self-employed",
            "Residence_type": "Rural",
            "smoking_status": "never smoked"
        }
    ]
    
    inference_results = predictor.predict_risk(demo_patients)
    print("\nInference Results:")
    print(json.dumps(inference_results, indent=4))
    
    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    main()
