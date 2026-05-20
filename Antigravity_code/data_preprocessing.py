import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# Define root paths
ROOT_DIR = Path(__file__).resolve().parent.parent
CALORIES_PATH = ROOT_DIR / "Calo_burn" / "calories.csv"
STROKE_PATH = ROOT_DIR / "Stroke_dt" / "Stroke_dt.csv"

# Preprocessing Constants
CALORIES_FEATURE_COLUMNS = ["Gender", "Age", "BMI"]

STROKE_NUMERIC_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "avg_RestingBP",
    "heart_cholesterol",
    "avg_MaxHR",
    "avg_Oldpeak",
    "heart_disease_rate",
    "avg_height",
    "avg_weight",
    "avg_ap_hi",
    "avg_ap_lo",
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

FEATURE_SETS = {
    "baseline": [
        "age",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi",
        "Estimated_calories"
    ],
    "clinical_only": [
        "age",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi",
        "avg_RestingBP",
        "heart_cholesterol",
        "avg_MaxHR",
        "avg_Oldpeak",
        "heart_disease_rate",
        "Estimated_calories"
    ],
    "all_except_redflags": [
        "gender", "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
        "avg_RestingBP", "avg_MaxHR", "avg_Oldpeak", "heart_disease_rate",
        "cardio_cholesterol", "cardio_gluc", "cardio_rate", "smoke_flag", "alco_flag",
        "Estimated_calories"
    ],
    "all_features": [
        "gender", "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
        "avg_RestingBP", "heart_cholesterol", "avg_MaxHR", "avg_Oldpeak", "heart_disease_rate",
        "avg_height", "avg_weight", "avg_ap_hi", "avg_ap_lo",
        "cardio_cholesterol", "cardio_gluc", "cardio_rate", "smoke_flag", "alco_flag",
        "Estimated_calories"
    ]
}

def normalize_gender(series):
    """Normalize gender values: Female/female -> 1, Male/male -> 0."""
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map({"male": 0, "female": 1})

def clean_bmi(series):
    """Clean out-of-range BMI values."""
    values = pd.to_numeric(series, errors="coerce")
    return values.mask((values < 10) | (values > 80))

def mode_value(series, fallback="Unknown"):
    """Calculate the mode of a series with a fallback."""
    values = series.dropna()
    if values.empty:
        return fallback
    return values.mode(dropna=True).iloc[0]

def deduplicate_patients(df):
    """
    Group by patient_id and aggregate records to resolve database replication.
    Numeric values are aggregated by median, categoricals by mode, stroke_target by max.
    """
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
    print(f"Deduplicated: {len(df):,} rows -> {len(deduped):,} unique patients")
    return deduped

def numeric_medians(df, columns):
    """Compute medians for a list of numeric columns."""
    medians = {}
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        median_value = values.median()
        medians[col] = 0.0 if pd.isna(median_value) else float(median_value)
    return medians

def categorical_modes(df, columns):
    """Compute modes for a list of categorical columns."""
    modes = {}
    for col in columns:
        if col not in df.columns:
            continue
        modes[col] = str(mode_value(df[col]))
    return modes

def apply_numeric_fill(df, fill_values):
    """Apply median values to fill missing values in numeric columns."""
    for col, fill_value in fill_values.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_value)
    return df

def apply_categorical_fill(df, fill_values):
    """Apply mode values to fill missing values in categorical columns."""
    for col, fill_value in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    return df

def build_calories_model(df_cal, random_state=42):
    """Train XGBRegressor on calories dataset to estimate energy expenditure."""
    required = ["Gender", "Age", "Height", "Weight", "Calories"]
    missing = sorted(set(required) - set(df_cal.columns))
    if missing:
        raise ValueError(f"Calories dataset is missing columns: {missing}")

    df_cal = df_cal.copy()
    df_cal["Gender"] = normalize_gender(df_cal["Gender"])
    df_cal["BMI"] = df_cal["Weight"] / ((df_cal["Height"] / 100) ** 2)
    
    # Fill any NaNs
    medians = numeric_medians(df_cal, CALORIES_FEATURE_COLUMNS + ["Calories"])
    df_cal = apply_numeric_fill(df_cal, medians)

    X = df_cal[CALORIES_FEATURE_COLUMNS]
    y = df_cal["Calories"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    model = XGBRegressor(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=3.0,
        objective="reg:squarederror",
        random_state=random_state,
    )
    
    try:
        model.fit(X_train, y_train)
    except Exception:
        # Fallback to RandomForest if XGBoost fails
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

    return model

def build_preprocessing_artifacts(df_str):
    """Learn imputation parameters from training data."""
    df_str = df_str.copy()
    df_str["gender"] = normalize_gender(df_str["gender"])
    df_str["bmi"] = clean_bmi(df_str["bmi"])

    numeric_fill = numeric_medians(df_str, STROKE_NUMERIC_COLUMNS + ["stroke_target"])
    categorical_fill = categorical_modes(df_str, STROKE_CATEGORICAL_COLUMNS)

    return {
        "numeric_fill": numeric_fill,
        "categorical_fill": categorical_fill,
        "categorical_columns": STROKE_CATEGORICAL_COLUMNS,
        "calories_feature_columns": CALORIES_FEATURE_COLUMNS
    }

def preprocess_stroke_data(df_str, calories_model, artifacts=None, require_target=True):
    """Clean, impute, feature engineer, and encode stroke patient dataset."""
    df_str = df_str.copy()

    if artifacts is None:
        artifacts = build_preprocessing_artifacts(df_str)

    # Normalize gender & clean BMI
    df_str["gender"] = normalize_gender(df_str["gender"])
    df_str["bmi"] = clean_bmi(df_str["bmi"])

    # Impute missing values using the training artifacts
    df_str = apply_numeric_fill(df_str, artifacts["numeric_fill"])
    df_str = apply_categorical_fill(df_str, artifacts["categorical_fill"])

    # Predict Estimated_calories based on Gender, Age, BMI
    energy_features = df_str[["gender", "age", "bmi"]].rename(
        columns={"gender": "Gender", "age": "Age", "bmi": "BMI"}
    )
    df_str["Estimated_calories"] = calories_model.predict(energy_features)

    # Target variable handling
    y = None
    if "stroke_target" in df_str.columns:
        df_str["stroke_target"] = pd.to_numeric(df_str["stroke_target"], errors="coerce")
        fill_val = artifacts["numeric_fill"].get("stroke_target", 0)
        df_str["stroke_target"] = df_str["stroke_target"].fillna(fill_val).astype(int)
        y = df_str["stroke_target"]

    # Save list of original columns before dummy variables encoding
    pre_dummies_cols = list(df_str.columns)

    # One-hot encode categoricals
    cat_cols_to_encode = [col for col in artifacts["categorical_columns"] if col in df_str.columns]
    df_str = pd.get_dummies(df_str, columns=cat_cols_to_encode, drop_first=True, dtype=int)

    # Add missing dummy columns that were learned in artifacts
    if "encoded_columns" in artifacts:
        for col in artifacts["encoded_columns"]:
            if col not in df_str.columns:
                df_str[col] = 0

    # Clean check
    remaining_nan = int(df_str.isna().sum().sum())
    if remaining_nan > 0:
        nan_cols = df_str.columns[df_str.isna().any()].tolist()
        raise ValueError(f"Preprocessed dataframe contains remaining NaN values in columns: {nan_cols}")

    return df_str, y, artifacts
