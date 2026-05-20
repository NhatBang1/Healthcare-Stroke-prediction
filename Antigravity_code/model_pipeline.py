import os
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

# Define target metric thresholds for performance review loop
TARGET_AUC = 0.85
TARGET_F1 = 0.28
MAX_ITERATIONS = 3
RANDOM_STATE = 42

def optimize_decision_threshold(y_true, y_prob):
    """Scan thresholds to find the one that yields the best F1 score."""
    thresholds = np.linspace(0.05, 0.95, 181)
    best_f1 = -1
    best_threshold = 0.5
    for t in thresholds:
        f1 = f1_score(y_true, y_prob >= t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    return float(best_threshold), float(best_f1)

def evaluate_model_on_data(model, X, y, threshold):
    """Evaluate a model with a given decision threshold and return core metrics."""
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.decision_function(X)
    
    y_pred = (y_prob >= threshold).astype(int)
    
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y, y_prob))
    }

def generate_synthetic_data(df_train_raw, num_samples=1000, random_state=42):
    """
    Generate synthetic raw patient data based on empirical distributions and statistics
    from the training raw data. This is used for out-of-distribution evaluation.
    """
    np.random.seed(random_state)
    synthetic_data = {}
    
    # Generate demographic and clinical variables using training distributions
    for col in df_train_raw.columns:
        if col == "patient_id":
            synthetic_data[col] = np.arange(100000, 100000 + num_samples)
        elif not pd.api.types.is_numeric_dtype(df_train_raw[col]) or df_train_raw[col].nunique(dropna=True) <= 5:
            # Discrete/Categorical/Small integer columns: sample from observed probabilities
            value_counts = df_train_raw[col].value_counts(dropna=True, normalize=True)
            choices = value_counts.index.tolist()
            probs = value_counts.values.tolist()
            synthetic_data[col] = np.random.choice(choices, size=num_samples, p=probs)
        else:
            # Numeric columns: sample from normal distribution matched to mean and std
            mean_val = df_train_raw[col].mean()
            std_val = df_train_raw[col].std()
            if pd.isna(mean_val) or pd.isna(std_val):
                # Fallbacks based on typical clinical statistics
                if col == "age":
                    mean_val, std_val = 50.0, 18.0
                elif col == "bmi":
                    mean_val, std_val = 28.0, 7.0
                elif col == "avg_glucose_level":
                    mean_val, std_val = 105.0, 45.0
                else:
                    mean_val, std_val = 100.0, 15.0
            
            vals = np.random.normal(loc=mean_val, scale=std_val, size=num_samples)
            
            # Post-processing caps to ensure biological plausibility
            if col == "age":
                vals = np.clip(vals, 0.5, 95.0)
            elif col == "bmi":
                vals = np.clip(vals, 10.0, 75.0)
            elif col == "avg_glucose_level":
                vals = np.clip(vals, 50.0, 300.0)
            elif col in ["avg_RestingBP", "avg_ap_hi"]:
                vals = np.clip(vals, 70.0, 200.0)
            elif col == "avg_ap_lo":
                vals = np.clip(vals, 40.0, 130.0)
            elif col == "avg_MaxHR":
                vals = np.clip(vals, 60.0, 220.0)
            elif col == "avg_Oldpeak":
                vals = np.clip(vals, 0.0, 6.0)
            elif col == "heart_disease_rate":
                vals = np.clip(vals, 0.0, 1.0)
            elif col == "avg_height":
                vals = np.clip(vals, 140.0, 200.0)
            elif col == "avg_weight":
                vals = np.clip(vals, 40.0, 150.0)
            elif col == "heart_cholesterol":
                vals = np.clip(vals, 100.0, 450.0)
                
            synthetic_data[col] = vals

    synthetic_df = pd.DataFrame(synthetic_data)
    
    # Recalculate BMI based on height/weight consistency if available
    # BMI = weight / (height/100)^2
    synthetic_df["bmi"] = synthetic_df["avg_weight"] / ((synthetic_df["avg_height"] / 100) ** 2)
    synthetic_df["bmi"] = clean_outliers_bmi(synthetic_df["bmi"])
    
    return synthetic_df

def clean_outliers_bmi(series):
    values = pd.to_numeric(series, errors="coerce")
    return values.mask((values < 10) | (values > 80)).fillna(28.0)

def search_hyperparameters(model_name, X_train, y_train):
    """Run randomized search to tune hyperparameters for the underperforming models."""
    print(f"--- Hyperparameter Tuning: {model_name} ---")
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)

    if model_name == "XGBoost":
        param_dist = {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "reg_lambda": [1.0, 5.0, 10.0, 20.0]
        }
        xgb = XGBClassifier(objective="binary:logistic", scale_pos_weight=scale_pos_weight, 
                            eval_metric="logloss", random_state=RANDOM_STATE)
        search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=15, 
                                    scoring="roc_auc", cv=3, random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(X_train, y_train)
        return search.best_estimator_
        
    elif model_name == "LogisticRegression":
        param_dist = {
            "logisticregression__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "logisticregression__penalty": ["l2"]
        }
        pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE))
        search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=5, 
                                    scoring="roc_auc", cv=3, random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(X_train, y_train)
        return search.best_estimator_

    elif model_name == "GradientBoosting":
        param_dist = {
            "n_estimators": [50, 100, 150],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0]
        }
        gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
        search = RandomizedSearchCV(gb, param_distributions=param_dist, n_iter=10, 
                                    scoring="roc_auc", cv=3, random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(X_train, y_train)
        return search.best_estimator_
        
    else:
        # For RandomForest or others
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5, 6],
            "min_samples_leaf": [4, 8, 12]
        }
        rf = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
        search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, 
                                    scoring="roc_auc", cv=3, random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(X_train, y_train)
        return search.best_estimator_

def run_evaluation_review_loop(X_train, y_train, X_val, y_val, X_test, y_test, 
                               X_synth, y_synth, feature_cols):
    """
    Core Evaluation & Review Loop.
    1. Trains multiple candidate models.
    2. Optimizes thresholds on validation data.
    3. Reviews accuracy, recall, and F1-score on both test and synthetic test sets.
    4. If results do not meet targets, automatically tunes hyperparameters, rebuilds, and restarts the review.
    5. Returns the best performing model.
    """
    iteration = 1
    best_overall_auc = 0
    best_overall_model = None
    best_overall_name = ""
    best_overall_features = []
    best_overall_threshold = 0.5
    best_overall_metrics = {}

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)

    # Dictionary of standard candidate models
    models = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, 
                                 scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=5, class_weight="balanced", random_state=RANDOM_STATE)
    }

    # Track if models need hyperparameter tuning (starting with False, will trigger if loop fails)
    tune_models = False

    while iteration <= MAX_ITERATIONS:
        print(f"\n==========================================")
        print(f"REVIEW LOOP ITERATION {iteration}/{MAX_ITERATIONS}")
        print(f"==========================================")
        
        current_best_auc = 0
        current_best_model = None
        current_best_name = ""
        current_best_threshold = 0.5
        current_best_metrics = {}

        for name, model in list(models.items()):
            print(f"\nTraining model: {name} (Tuning={tune_models})")
            
            if tune_models:
                # Run tuning search to rebuild the model
                model = search_hyperparameters(name, X_train, y_train)
                models[name] = model  # update dictionary

            model.fit(X_train, y_train)
            
            # Predict validation set probabilities to optimize decision threshold
            if hasattr(model, "predict_proba"):
                y_prob_val = model.predict_proba(X_val)[:, 1]
                y_prob_test = model.predict_proba(X_test)[:, 1]
                y_prob_synth = model.predict_proba(X_synth)[:, 1]
            else:
                y_prob_val = model.decision_function(X_val)
                y_prob_test = model.decision_function(X_test)
                y_prob_synth = model.decision_function(X_synth)

            threshold, val_f1 = optimize_decision_threshold(y_val, y_prob_val)
            
            # Evaluate on standard test set
            test_metrics = evaluate_model_on_data(model, X_test, y_test, threshold)
            
            # Evaluate on synthetic dataset
            synth_metrics = evaluate_model_on_data(model, X_synth, y_synth, threshold)
            
            print(f"Model: {name} | Threshold: {threshold:.3f}")
            print(f"  [Real Test]      Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f} | Recall: {test_metrics['recall']:.4f} | AUC: {test_metrics['auc_roc']:.4f}")
            print(f"  [Synthetic Test] Acc: {synth_metrics['accuracy']:.4f} | F1: {synth_metrics['f1']:.4f} | Recall: {synth_metrics['recall']:.4f} | AUC: {synth_metrics['auc_roc']:.4f}")
            
            # Use average AUC across real and synthetic test data for model selection
            avg_auc = (test_metrics["auc_roc"] + synth_metrics["auc_roc"]) / 2
            
            if avg_auc > current_best_auc:
                current_best_auc = avg_auc
                current_best_model = model
                current_best_name = name
                current_best_threshold = threshold
                current_best_metrics = {
                    "real": test_metrics,
                    "synthetic": synth_metrics
                }

        print(f"\nIteration Best Model: {current_best_name} with Average AUC: {current_best_auc:.4f}")
        
        # Check if the iteration's best model satisfies the quality threshold
        real_f1 = current_best_metrics["real"]["f1"]
        real_auc = current_best_metrics["real"]["auc_roc"]
        synth_f1 = current_best_metrics["synthetic"]["f1"]
        synth_auc = current_best_metrics["synthetic"]["auc_roc"]
        
        passed_quality_gate = (real_auc >= TARGET_AUC) and (real_f1 >= TARGET_F1)
                              
        if passed_quality_gate:
            print(f"\n>>> Quality Gate PASSED! Best model satisfies all requirements.")
            best_overall_model = current_best_model
            best_overall_name = current_best_name
            best_overall_auc = current_best_auc
            best_overall_threshold = current_best_threshold
            best_overall_metrics = current_best_metrics
            break
        else:
            print(f"\n>>> Quality Gate FAILED! Real AUC: {real_auc:.4f} (Target: {TARGET_AUC}), Real F1: {real_f1:.4f} (Target: {TARGET_F1}).")
            print("Triggering auto-review & hyperparameter search to rebuild models...")
            
            # Save the current best as the fallback overall best in case subsequent iterations don't improve
            if current_best_auc > best_overall_auc:
                best_overall_model = current_best_model
                best_overall_name = current_best_name
                best_overall_auc = current_best_auc
                best_overall_threshold = current_best_threshold
                best_overall_metrics = current_best_metrics
            
            tune_models = True
            iteration += 1

    if iteration > MAX_ITERATIONS:
        print(f"\n>>> Max iterations reached! Fallback to the overall best model discovered.")
        
    print(f"\nFINAL SELECTION: {best_overall_name}")
    print(f"  Decision Threshold: {best_overall_threshold:.3f}")
    print(f"  Real Test Metrics:      {best_overall_metrics['real']}")
    print(f"  Synthetic Test Metrics: {best_overall_metrics['synthetic']}")
    
    return best_overall_model, best_overall_name, best_overall_threshold, best_overall_metrics

def export_model_onnx(model, num_features, output_path):
    """Export the trained model to ONNX format for deployment compatibility."""
    try:
        from skl2onnx.common.data_types import FloatTensorType
        from onnxmltools import convert_xgboost
        from skl2onnx import convert_sklearn
    except ImportError:
        print(f"Skip ONNX export: install onnxmltools, skl2onnx and onnxruntime.")
        return False

    initial_types = [("input", FloatTensorType([None, num_features]))]
    
    try:
        if "XGBClassifier" in str(type(model)):
            booster = model.get_booster()
            original_names = booster.feature_names
            booster.feature_names = [f"f{i}" for i in range(num_features)]
            try:
                onnx_model = convert_xgboost(model, initial_types=initial_types, target_opset=15)
            finally:
                booster.feature_names = original_names
        else:
            # Handle scikit-learn models (LogisticRegression, RandomForest, etc.)
            onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=15)
            
        output_path.write_bytes(onnx_model.SerializeToString())
        print(f"Saved ONNX: {output_path}")
        return True
    except Exception as e:
        print(f"Error during ONNX conversion: {e}")
        return False
