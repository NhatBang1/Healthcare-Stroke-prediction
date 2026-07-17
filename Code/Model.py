from pathlib import Path
import json
import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold

warnings.filterwarnings("ignore")

try:
    import shap
    from xgboost import XGBClassifier, XGBRegressor
except ImportError as exc:
    raise ImportError(
        "Model.py requires xgboost and shap. Activate MLwPC and install requirements first."
    ) from exc


ROOT_DIR = Path(__file__).resolve().parent.parent
CALORIES_PATH = ROOT_DIR / "Calo_burn" / "calories.csv"
STROKE_PATH = ROOT_DIR / "Stroke_dt" / "Stroke_dt.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
RANDOM_STATE = 42

CALORIES_FEATURE_COLUMNS = ["Gender", "Age", "BMI"]

STROKE_NUMERIC_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "avg_RestingBP",
    "avg_MaxHR",
    "avg_Oldpeak",
    "heart_disease_rate",
    "cardio_cholesterol",
    "cardio_gluc",
    "cardio_rate",
    "smoke_flag",
    "alco_flag"
]

STROKE_CATEGORICAL_COLUMNS = [
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

LEAKY_OR_NOISY_COLUMNS = [
    "avg_weight",
    "avg_height",
    "avg_ap_hi",
    "avg_ap_lo",
    "heart_cholesterol"
]

# 25 features after drop leaky and dummy encoding
FINAL_FEATURE_COLUMNS = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'avg_RestingBP', 'avg_MaxHR', 'avg_Oldpeak', 'heart_disease_rate',
    'cardio_cholesterol', 'cardio_gluc', 'cardio_rate', 'gender',
    'smoke_flag', 'alco_flag', 'Estimated_calories', 'ever_married_Yes',
    'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed',
    'work_type_children', 'Residence_type_Urban',
    'smoking_status_never smoked', 'smoking_status_smokes',
    'smoking_status_unknown'
]

EDGE_FEATURE_COLUMNS = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "Estimated_calories"
]


def normalize_gender(series):
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map({"male": 0, "female": 1})


def clean_bmi(series):
    values = pd.to_numeric(series, errors="coerce")
    return values.mask((values < 10) | (values > 80))


def mode_value(series, fallback="Unknown"):
    values = series.dropna()
    if values.empty:
        return fallback
    return values.mode(dropna=True).iloc[0]


def deduplicate_patients(df):
    if "patient_id" not in df.columns:
        return df.drop_duplicates().reset_index(drop=True)

    aggregations = {}
    for col in df.columns:
        if col == "patient_id":
            continue
        if col == "stroke_target":
            aggregations[col] = "max"
        elif pd.api.types.is_numeric_dtype(df[col]):
            aggregations[col] = "median"
        else:
            aggregations[col] = mode_value

    deduped = df.groupby("patient_id", as_index=False).agg(aggregations)
    print(
        f"Deduplicated stroke data: {len(df):,} rows -> "
        f"{len(deduped):,} unique patients"
    )
    return deduped


def numeric_medians(df, columns):
    medians = {}
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        median_value = values.median()
        medians[col] = 0.0 if pd.isna(median_value) else float(median_value)
    return medians


def categorical_modes(df, columns):
    modes = {}
    for col in columns:
        if col not in df.columns:
            continue
        modes[col] = str(mode_value(df[col]))
    return modes


def apply_numeric_fill(df, fill_values):
    for col, fill_value in fill_values.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_value)
    return df


def apply_categorical_fill(df, fill_values):
    for col, fill_value in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    return df


def build_calories_model(df_cal):
    required_columns = ["Gender", "Age", "Height", "Weight", "Calories"]
    missing_columns = sorted(set(required_columns) - set(df_cal.columns))
    if missing_columns:
        raise ValueError(f"Calories dataset is missing columns: {missing_columns}")

    df_cal = df_cal.copy()
    df_cal["Gender"] = normalize_gender(df_cal["Gender"])
    df_cal["BMI"] = df_cal["Weight"] / ((df_cal["Height"] / 100) ** 2)
    df_cal = apply_numeric_fill(
        df_cal,
        numeric_medians(df_cal, CALORIES_FEATURE_COLUMNS + ["Calories"]),
    )

    X = df_cal[CALORIES_FEATURE_COLUMNS]
    y = df_cal["Calories"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )

    model = XGBRegressor(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=3.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
    )
    try:
        model.fit(X_train, y_train)
        model_name = "XGBRegressor"
    except Exception:
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        model_name = "RandomForestRegressor fallback"

    mae = np.mean(np.abs(y_test - model.predict(X_test)))
    print(f"MAE Calories model ({model_name}): {mae:.4f}")
    return model


def build_preprocessing_artifacts(df_str):
    df_str = df_str.copy()
    df_str["gender"] = normalize_gender(df_str["gender"])
    df_str["bmi"] = clean_bmi(df_str["bmi"])
    numeric_fill = numeric_medians(
        df_str,
        STROKE_NUMERIC_COLUMNS + ["gender", "stroke_target"],
    )
    categorical_fill = categorical_modes(df_str, STROKE_CATEGORICAL_COLUMNS)
    return {
        "numeric_fill": numeric_fill,
        "categorical_fill": categorical_fill,
        "categorical_columns": STROKE_CATEGORICAL_COLUMNS,
        "excluded_columns": LEAKY_OR_NOISY_COLUMNS,
        "selected_features": FINAL_FEATURE_COLUMNS,
        "calories_feature_columns": CALORIES_FEATURE_COLUMNS,
    }


def preprocess_stroke_data(df_str, calories_model, artifacts=None, require_target=True):
    required_columns = ["gender", "age", "bmi"]
    if require_target:
        required_columns.append("stroke_target")
    missing_columns = sorted(set(required_columns) - set(df_str.columns))
    if missing_columns:
        raise ValueError(f"Stroke dataset is missing columns: {missing_columns}")

    df_str = df_str.copy()
    artifacts = artifacts or build_preprocessing_artifacts(df_str)

    df_str["gender"] = normalize_gender(df_str["gender"])
    df_str["bmi"] = clean_bmi(df_str["bmi"])
    df_str = apply_numeric_fill(df_str, artifacts["numeric_fill"])
    df_str = apply_categorical_fill(df_str, artifacts["categorical_fill"])

    energy_features = df_str[["gender", "age", "bmi"]].rename(
        columns={"gender": "Gender", "age": "Age", "bmi": "BMI"}
    )
    df_str["Estimated_calories"] = calories_model.predict(energy_features)
    df_str = df_str.drop(columns=LEAKY_OR_NOISY_COLUMNS, errors="ignore")

    categorical_columns = [
        col for col in artifacts["categorical_columns"] if col in df_str.columns
    ]
    df_str = pd.get_dummies(df_str, columns=categorical_columns, drop_first=True, dtype=int)

    y = None
    if "stroke_target" in df_str.columns:
        df_str["stroke_target"] = pd.to_numeric(df_str["stroke_target"], errors="coerce")
        fill_value = artifacts["numeric_fill"].get("stroke_target", 0)
        df_str["stroke_target"] = df_str["stroke_target"].fillna(fill_value).astype(int)
        y = df_str["stroke_target"]

    X_all = df_str.drop(columns=["stroke_target", "patient_id"], errors="ignore")
    selected_features = artifacts.get("selected_features", FINAL_FEATURE_COLUMNS)
    X_all = X_all.reindex(columns=selected_features, fill_value=0)
    artifacts["feature_columns"] = selected_features

    remaining_nan = int(X_all.isna().sum().sum() + (0 if y is None else y.isna().sum()))
    if remaining_nan:
        nan_columns = X_all.columns[X_all.isna().any()].tolist()
        raise ValueError(f"Van con NaN trong du lieu: {nan_columns}")

    non_numeric_columns = X_all.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        raise ValueError(f"Feature columns are not numeric after preprocessing: {non_numeric_columns}")

    return X_all, X_all.copy(), y, artifacts


def stratified_split(df):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=df["stroke_target"],
    )
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_df["stroke_target"],
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_stroke_model(y_train):
    # Stacking Classifier (Calibrated Sigmoid) final model
    base_estimators = [
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)),
        ('hgb', HistGradientBoostingClassifier(class_weight='balanced', random_state=RANDOM_STATE))
    ]
    
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(random_state=RANDOM_STATE),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    
    model = CalibratedClassifierCV(
        estimator=stacking,
        method='sigmoid',
        cv=5
    )
    return model


def build_stroke_model_edge(y_train):
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)
    
    return XGBClassifier(
        n_estimators=80,
        max_depth=2,
        learning_rate=0.05,
        min_child_weight=8,
        gamma=0.5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=12.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    )


def best_f1_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, y_prob >= threshold) for threshold in thresholds]
    best_index = int(np.argmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])


def optimize_clinical_threshold(y_true, y_prob, gb_val_fp, pr_auc_val):
    thresholds = np.arange(0.01, 1.00, 0.01)
    rows = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        bal_acc = (rec + spec) / 2
        
        score = (0.30 * rec + 
                 0.25 * acc + 
                 0.20 * spec + 
                 0.15 * bal_acc + 
                 0.05 * f1 + 
                 0.05 * pr_auc_val)
        
        penalty = 0
        if rec < 0.85: penalty += 0.08
        if acc < 0.80: penalty += 0.08
        if spec < 0.75: penalty += 0.05
        if fn > 6: penalty += 0.05
        if fp > gb_val_fp: penalty += 0.03
            
        final_score = score - penalty
        rows.append({
            'threshold': thresh,
            'recall': rec,
            'accuracy': acc,
            'specificity': spec,
            'strong_score': final_score,
            'Recall_ge_85': int(rec >= 0.85),
            'Acc_ge_80': int(acc >= 0.80),
            'Spec_ge_75': int(spec >= 0.75)
        })
        
    df_res = pd.DataFrame(rows)
    cond_all = (df_res['recall'] >= 0.85) & (df_res['accuracy'] >= 0.80) & (df_res['specificity'] >= 0.75)
    sat_all = df_res[cond_all]
    
    if len(sat_all) > 0:
        selected_row = sat_all.sort_values(by='strong_score', ascending=False).iloc[0]
    else:
        rec_ok_all = df_res[df_res['recall'] >= 0.85]
        if len(rec_ok_all) > 0:
            selected_row = rec_ok_all.sort_values(by=['accuracy', 'recall', 'strong_score'], ascending=[False, False, False]).iloc[0]
        else:
            selected_row = df_res.sort_values(by='strong_score', ascending=False).iloc[0]
            
    return float(selected_row['threshold'])


def evaluate_classifier(model, X, y, threshold, name):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\n===== {name} =====")
    print("AUC-ROC:", roc_auc_score(y, y_prob))
    print("Threshold:", threshold)
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred, digits=4))
    return y_prob, y_pred


def save_xgboost_native(model, path):
    if hasattr(model, "save_model"):
        model.save_model(path)


def export_xgboost_onnx(model, feature_count, output_path):
    try:
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        print(f"Skip ONNX export for {output_path.name}: install onnxmltools and skl2onnx.")
        return False

    initial_types = [("input", FloatTensorType([None, feature_count]))]
    booster = model.get_booster()
    original_feature_names = booster.feature_names
    booster.feature_names = [f"f{i}" for i in range(feature_count)]
    try:
        onnx_model = convert_xgboost(model, initial_types=initial_types, target_opset=15)
        output_path.write_bytes(onnx_model.SerializeToString())
    finally:
        booster.feature_names = original_feature_names
    print(f"Saved ONNX: {output_path}")
    return True


def save_artifacts(calories_model, stroke_model, stroke_model_clinical, artifacts, threshold, threshold_clinical):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dump(calories_model, ARTIFACTS_DIR / "calories_model.joblib")
    dump(stroke_model, ARTIFACTS_DIR / "stroke_model.joblib")
    dump(stroke_model_clinical, ARTIFACTS_DIR / "stroke_model_clinical.joblib")
    save_xgboost_native(calories_model, ARTIFACTS_DIR / "calories_model.json")
    save_xgboost_native(stroke_model, ARTIFACTS_DIR / "stroke_model.json")

    metadata = {
        **artifacts,
        "decision_threshold_edge": threshold,
        "decision_threshold_clinical": threshold_clinical,
        "stroke_model_input": "float32 tensor with shape [batch, feature_count]",
        "stroke_model_output": "class label and class probabilities",
        "notes": [
            "Leaky features (avg_weight, avg_height, avg_ap_hi, avg_ap_lo, heart_cholesterol) are excluded.",
            "BMI values outside [10, 80] are treated as missing before median imputation.",
            "Stroke rows are deduplicated by patient_id before train/validation/test split.",
            "Stacking Classifier (Calibrated Sigmoid) with 25 standard features is used for clinical reporting.",
            "XGBoost Classifier with 6 baseline features is deployed for C++ edge inference.",
        ],
    }
    (ARTIFACTS_DIR / "preprocess_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    export_xgboost_onnx(
        calories_model,
        len(CALORIES_FEATURE_COLUMNS),
        ARTIFACTS_DIR / "calories_model.onnx",
    )
    export_xgboost_onnx(
        stroke_model,
        len(EDGE_FEATURE_COLUMNS),
        ARTIFACTS_DIR / "stroke_model.onnx",
    )


def save_avg_ap_hi_diagnostics(df):
    if "avg_ap_hi" not in df.columns:
        return

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = df[["avg_ap_hi", "stroke_target"]].dropna().copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(8, 5))
    sns.stripplot(
        data=plot_df,
        x="stroke_target",
        y="avg_ap_hi",
        jitter=0.25,
        alpha=0.45,
        size=3,
    )
    sns.pointplot(
        data=plot_df,
        x="stroke_target",
        y="avg_ap_hi",
        estimator="mean",
        errorbar=("ci", 95),
        color="black",
        markers="D",
    )
    plt.title("avg_ap_hi distribution by stroke_target")
    plt.xlabel("stroke_target")
    plt.ylabel("avg_ap_hi")
    path = ARTIFACTS_DIR / "avg_ap_hi_by_target.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved avg_ap_hi diagnostic plot: {path}")

    plt.figure(figsize=(9, 5))
    plot_df["avg_ap_hi_bin"] = pd.qcut(plot_df["avg_ap_hi"], 10, duplicates="drop")
    rate = plot_df.groupby("avg_ap_hi_bin", observed=True)["stroke_target"].mean()
    rate.plot(kind="bar")
    plt.title("Stroke rate by avg_ap_hi decile")
    plt.xlabel("avg_ap_hi decile")
    plt.ylabel("stroke rate")
    plt.xticks(rotation=35, ha="right")
    path = ARTIFACTS_DIR / "avg_ap_hi_stroke_rate_decile.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved avg_ap_hi decile plot: {path}")


def save_red_flag_diagnostics(df):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    diagnostic_columns = ["avg_ap_hi", "avg_ap_lo", "heart_cholesterol", "bmi"]
    rows = []
    for col in diagnostic_columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "missing": int(values.isna().sum()),
                "non_positive": int((values <= 0).sum()),
                "min": values.min(),
                "median": values.median(),
                "max": values.max(),
                "corr_with_stroke": values.corr(df["stroke_target"]),
                "mean_non_stroke": values[df["stroke_target"] == 0].mean(),
                "mean_stroke": values[df["stroke_target"] == 1].mean(),
            }
        )

    path = ARTIFACTS_DIR / "red_flag_feature_diagnostics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved red-flag feature diagnostics: {path}")


def plot_shap_summary(model, X_test):
    print("\nGenerating SHAP summary plot...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    sample = X_test.sample(n=min(100, len(X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    beeswarm_path = ARTIFACTS_DIR / "shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP beeswarm: {beeswarm_path}")

    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = ARTIFACTS_DIR / "shap_summary_bar.png"
    plt.savefig(bar_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP bar: {bar_path}")


def train_pipeline(save_models=True, show_shap=True):
    df_cal = pd.read_csv(CALORIES_PATH)
    calories_model = build_calories_model(df_cal)

    print("\nPreprocessing raw datasets from scratch...")
    df_str_raw = pd.read_csv(STROKE_PATH)
    df_str = deduplicate_patients(df_str_raw)
    save_avg_ap_hi_diagnostics(df_str)
    save_red_flag_diagnostics(df_str)
    train_df, valid_df, test_df = stratified_split(df_str)

    X_train_pro, X_train, y_train, artifacts = preprocess_stroke_data(train_df, calories_model)
    X_valid_pro, X_valid, y_valid, _ = preprocess_stroke_data(valid_df, calories_model, artifacts=artifacts)
    X_test_pro, X_test, y_test, _ = preprocess_stroke_data(test_df, calories_model, artifacts=artifacts)

    # Reconstruct processed dataframes to save them
    train_df_save = train_df[['patient_id']].copy()
    train_df_save['stroke_target'] = y_train.values
    train_df_save = pd.concat([train_df_save, X_train], axis=1)

    val_df_save = valid_df[['patient_id']].copy()
    val_df_save['stroke_target'] = y_valid.values
    val_df_save = pd.concat([val_df_save, X_valid], axis=1)

    test_df_save = test_df[['patient_id']].copy()
    test_df_save['stroke_target'] = y_test.values
    test_df_save = pd.concat([test_df_save, X_test], axis=1)

    # Save to Report/, SIC_report/, and Code/
    for folder in ["Report", "SIC_report", "Code"]:
        dest_dir = ROOT_DIR / folder
        if dest_dir.exists():
            train_df_save.to_csv(dest_dir / "stroke_train_processed.csv", index=False)
            val_df_save.to_csv(dest_dir / "stroke_val_processed.csv", index=False)
            test_df_save.to_csv(dest_dir / "stroke_test_processed.csv", index=False)
            print(f"Saved preprocessed datasets to {folder}/")

    # 1. Train Stacking Classifier Clinical Model (25 features)
    print("\nTraining Clinical Stacking Classifier...")
    model_clinical = build_stroke_model(y_train)
    model_clinical.fit(X_train, y_train)

    # 2. Train XGBoost Edge Model (6 features)
    print("\nTraining Edge XGBoost Classifier...")
    X_train_edge = X_train[EDGE_FEATURE_COLUMNS]
    X_valid_edge = X_valid[EDGE_FEATURE_COLUMNS]
    X_test_edge = X_test[EDGE_FEATURE_COLUMNS]
    
    model_edge = build_stroke_model_edge(y_train)
    model_edge.fit(X_train_edge, y_train, verbose=False)

    # 3. Find thresholds
    # Stacking Clinical threshold optimization on validation set
    valid_prob_clinical = model_clinical.predict_proba(X_valid)[:, 1]
    
    # Fit a standard GB baseline model on validation to calculate clinical safety penalties
    gb_baseline = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    gb_baseline.fit(X_train, y_train)
    gb_val_probs = gb_baseline.predict_proba(X_valid)[:, 1]
    gb_val_preds = (gb_val_probs >= 0.05).astype(int)
    _, gb_val_fp, _, _ = confusion_matrix(y_valid, gb_val_preds).ravel()
    
    pr_auc_val_clinical = average_precision_score(y_valid, valid_prob_clinical)
    threshold_clinical = optimize_clinical_threshold(y_valid, valid_prob_clinical, gb_val_fp, pr_auc_val_clinical)
    print(f"\nOptimal Clinical Stacking threshold: {threshold_clinical:.2f}")

    # Edge model threshold by F1
    valid_prob_edge = model_edge.predict_proba(X_valid_edge)[:, 1]
    threshold_edge, valid_f1_edge = best_f1_threshold(y_valid, valid_prob_edge)
    print(f"Optimal Edge XGBoost threshold by F1: {threshold_edge:.3f} (F1={valid_f1_edge:.4f})")

    # Evaluate Clinical Stacking
    evaluate_classifier(model_clinical, X_valid, y_valid, threshold_clinical, "Validation Set - Clinical Stacking")
    evaluate_classifier(model_clinical, X_test, y_test, threshold_clinical, "Test Set - Clinical Stacking")

    # Evaluate Edge XGBoost
    evaluate_classifier(model_edge, X_valid_edge, y_valid, threshold_edge, "Validation Set - Edge XGBoost")
    evaluate_classifier(model_edge, X_test_edge, y_test, threshold_edge, "Test Set - Edge XGBoost")

    if save_models:
        save_artifacts(calories_model, model_edge, model_clinical, artifacts, threshold_edge, threshold_clinical)

    if show_shap:
        # Generate SHAP on a simpler tree estimator from Stacking or a fitted tree model
        # We fit a fast random forest on X_train for SHAP to avoid stack evaluation issues
        rf_shap = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=RANDOM_STATE)
        rf_shap.fit(X_train, y_train)
        plot_shap_summary(rf_shap, X_test)

    return {
        "calories_model": calories_model,
        "stroke_model": model_edge,
        "stroke_model_clinical": model_clinical,
        "artifacts": artifacts,
        "threshold_edge": threshold_edge,
        "threshold_clinical": threshold_clinical,
        "X_test_pro": X_test,
        "y_test": y_test,
    }


def main():
    train_pipeline(save_models=True, show_shap=True)


if __name__ == "__main__":
    main()
