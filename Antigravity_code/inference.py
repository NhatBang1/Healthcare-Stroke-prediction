import json
import warnings
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

class StrokePredictor:
    """
    Standard deployment-ready inference class.
    Integrates preprocessing, calories regression, and stroke classification.
    """
    def __init__(self, artifacts_dir):
        self.artifacts_dir = Path(artifacts_dir)
        
        # Load artifacts
        self.calories_model = joblib.load(self.artifacts_dir / "calories_model.joblib")
        self.stroke_model = joblib.load(self.artifacts_dir / "stroke_model.joblib")
        
        with open(self.artifacts_dir / "preprocess_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        self.numeric_fill = self.metadata["numeric_fill"]
        self.categorical_fill = self.metadata["categorical_fill"]
        self.categorical_columns = self.metadata["categorical_columns"]
        self.feature_columns = self.metadata["feature_columns"]
        self.decision_threshold = self.metadata["decision_threshold"]

    def _normalize_gender(self, value):
        if pd.isna(value):
            return np.nan
        val_str = str(value).strip().lower()
        if val_str == "female":
            return 1
        elif val_str == "male":
            return 0
        return np.nan

    def _clean_bmi(self, value):
        try:
            val_num = float(value)
            if val_num < 10.0 or val_num > 80.0:
                return np.nan
            return val_num
        except (ValueError, TypeError):
            return np.nan

    def preprocess_patient(self, raw_df):
        """Preprocess raw patient data to match model features exactly."""
        df = raw_df.copy()
        
        # 1. Normalize gender and BMI caps
        df["gender"] = df["gender"].apply(self._normalize_gender)
        df["bmi"] = df["bmi"].apply(self._clean_bmi)
        
        # 2. Impute missing values using stored training medians/modes
        for col, fill_val in self.numeric_fill.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_val)
                
        for col, fill_val in self.categorical_fill.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_val)
                
        # 3. Estimate Calories Burned using the calories model
        # Create temporary features aligned with the calories model: Gender, Age, BMI
        energy_features = df[["gender", "age", "bmi"]].rename(
            columns={"gender": "Gender", "age": "Age", "bmi": "BMI"}
        )
        df["Estimated_calories"] = self.calories_model.predict(energy_features)
        
        # 4. Dummy encode categorical columns
        cat_cols = [col for col in self.categorical_columns if col in df.columns]
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
        
        # 5. Reindex columns to match training features exactly (handling missing categories)
        df_model = df.reindex(columns=self.feature_columns, fill_value=0)
        
        return df_model

    def predict_risk(self, patient_dict_list):
        """
        Predict stroke risk (probability and class) for a list of patient profiles.
        Input: list of dictionaries representing patients.
        Output: list of dictionaries with probabilities and binary flags.
        """
        # Convert list of dicts to DataFrame
        df_raw = pd.DataFrame(patient_dict_list)
        
        # Preprocess
        df_processed = self.preprocess_patient(df_raw)
        
        # Run classification model
        if hasattr(self.stroke_model, "predict_proba"):
            probs = self.stroke_model.predict_proba(df_processed)[:, 1]
        else:
            probs = self.stroke_model.decision_function(df_processed)
            
        preds = (probs >= self.decision_threshold).astype(int)
        
        results = []
        for i, (prob, pred) in enumerate(zip(probs, preds)):
            results.append({
                "patient_id": patient_dict_list[i].get("patient_id", None),
                "stroke_risk_probability": float(prob),
                "stroke_prediction_flag": int(pred),
                "decision_threshold_used": float(self.decision_threshold)
            })
            
        return results
