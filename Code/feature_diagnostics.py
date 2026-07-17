import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from Model import (
    ARTIFACTS_DIR,
    CALORIES_PATH,
    RANDOM_STATE,
    STROKE_PATH,
    build_calories_model,
    preprocess_stroke_data,
)

try:
    import shap
    from xgboost import XGBClassifier
except ImportError:
    shap = None
    XGBClassifier = None


DIAGNOSTIC_FEATURES = ["bmi", "avg_weight", "avg_height", "age", "stroke_target"]


def correlation_report(df):
    print("\n===== Data Quality =====")
    print("Rows:", len(df))
    print("Unique patient_id:", df["patient_id"].nunique())
    print("Full duplicate rows:", df.duplicated().sum())
    print("Missing values:")
    print(df[DIAGNOSTIC_FEATURES].isna().sum())

    print("\n===== Pearson Correlation =====")
    print(df[DIAGNOSTIC_FEATURES].corr(numeric_only=True).round(4))

    print("\n===== Spearman Correlation =====")
    print(df[DIAGNOSTIC_FEATURES].corr(method="spearman", numeric_only=True).round(4))

    print("\n===== Stroke Rate By avg_weight Decile =====")
    weight_bucket = pd.qcut(df["avg_weight"], 10, duplicates="drop")
    print(df.groupby(weight_bucket, observed=True)["stroke_target"].agg(["count", "mean"]).round(4))


def approximate_vif(X, columns):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    print("\n===== Approximate Multicollinearity Check =====")
    for target_col in columns:
        predictors = [col for col in columns if col != target_col]
        y = X[target_col]
        x = X[predictors]
        linear = LinearRegression().fit(x, y)
        linear_r2 = r2_score(y, linear.predict(x))
        linear_vif = np.inf if linear_r2 >= 0.999999 else 1 / (1 - linear_r2)

        forest = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)
        forest.fit(x, y)
        forest_r2 = r2_score(y, forest.predict(x))
        print(
            f"{target_col}: linear_R2={linear_r2:.4f}, "
            f"linear_VIF={linear_vif:.2f}, nonlinear_R2={forest_r2:.4f}"
        )


def train_xgb(X_train, y_train):
    min_class_count = int(y_train.value_counts().min())
    k_neighbors = max(1, min(5, min_class_count - 1))
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model.fit(X_resampled, y_resampled)
    return model


def evaluate_split(name, X_train, X_test, y_train, y_test):
    model = train_xgb(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "split": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
        "model": model,
    }


def leakage_report(df, calories_model):
    X_baseline, X_proposed, y, artifacts = preprocess_stroke_data(df, calories_model)

    train_idx, test_idx = train_test_split(
        y.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    random_result = evaluate_split(
        "random_row_split",
        X_proposed.loc[train_idx],
        X_proposed.loc[test_idx],
        y.loc[train_idx],
        y.loc[test_idx],
    )

    groups = df.loc[y.index, "patient_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    group_train_pos, group_test_pos = next(splitter.split(X_proposed, y, groups=groups))
    group_train_idx = X_proposed.index[group_train_pos]
    group_test_idx = X_proposed.index[group_test_pos]
    group_result = evaluate_split(
        "patient_group_split",
        X_proposed.loc[group_train_idx],
        X_proposed.loc[group_test_idx],
        y.loc[group_train_idx],
        y.loc[group_test_idx],
    )

    print("\n===== Leakage Check: Random Row Split vs Patient Group Split =====")
    for result in [random_result, group_result]:
        print(
            f"{result['split']}: "
            f"accuracy={result['accuracy']:.4f}, "
            f"f1={result['f1']:.4f}, "
            f"auc_roc={result['auc_roc']:.4f}"
        )

    return X_proposed, y, random_result["model"], artifacts


def feature_variant_report(df, calories_model):
    variants = {
        "all_features": [],
        "drop_avg_weight_height": ["avg_weight", "avg_height"],
        "drop_bmi_keep_weight_height": ["bmi"],
        "drop_all_body_size_except_bmi": ["avg_weight", "avg_height", "Estimated_calories"],
    }

    groups = df["patient_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)

    print("\n===== Feature Variant Check With Patient Group Split =====")
    rows = []
    for variant_name, drop_columns in variants.items():
        train_raw_pos, test_raw_pos = next(splitter.split(df, df["stroke_target"], groups=groups))
        train_df = df.iloc[train_raw_pos].copy()
        test_df = df.iloc[test_raw_pos].copy()
        _, X_train, y_train, artifacts = preprocess_stroke_data(train_df, calories_model)
        _, X_test, y_test, _ = preprocess_stroke_data(
            test_df, calories_model, artifacts=artifacts
        )

        drop_existing = [col for col in drop_columns if col in X_train.columns]
        X_train = X_train.drop(columns=drop_existing)
        X_test = X_test.drop(columns=drop_existing)

        result = evaluate_split(variant_name, X_train, X_test, y_train, y_test)
        rows.append({key: value for key, value in result.items() if key != "model"})
        print(
            f"{variant_name}: accuracy={result['accuracy']:.4f}, "
            f"f1={result['f1']:.4f}, auc_roc={result['auc_roc']:.4f}"
        )

    return pd.DataFrame(rows)


def shap_weight_report(X, model):
    if shap is None or XGBClassifier is None:
        print("\nSkip SHAP diagnostics: shap or xgboost is not installed.")
        return

    sample = X.sample(n=min(3000, len(X)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    shap_df = pd.DataFrame(shap_values, columns=sample.columns, index=sample.index)

    print("\n===== SHAP Direction Check =====")
    for col in ["bmi", "avg_weight", "avg_height", "Estimated_calories"]:
        if col not in sample.columns:
            continue
        corr = np.corrcoef(sample[col], shap_df[col])[0, 1]
        high = shap_df.loc[sample[col] >= sample[col].quantile(0.75), col].mean()
        low = shap_df.loc[sample[col] <= sample[col].quantile(0.25), col].mean()
        print(
            f"{col}: corr(value, SHAP)={corr:.4f}, "
            f"mean_SHAP_low_q={low:.4f}, mean_SHAP_high_q={high:.4f}"
        )


def main():
    df = pd.read_csv(STROKE_PATH)
    df_cal = pd.read_csv(CALORIES_PATH)
    calories_model = build_calories_model(df_cal)

    correlation_report(df)
    filled_body = df[["bmi", "avg_weight", "avg_height"]].copy()
    for col in filled_body.columns:
        filled_body[col] = pd.to_numeric(filled_body[col], errors="coerce")
        filled_body[col] = filled_body[col].fillna(filled_body[col].median())
    approximate_vif(filled_body, ["bmi", "avg_weight", "avg_height"])

    X_proposed, _, random_model, _ = leakage_report(df, calories_model)
    variant_results = feature_variant_report(df, calories_model)
    shap_weight_report(X_proposed, random_model)

    output_path = ARTIFACTS_DIR / "feature_variant_diagnostics.csv"
    variant_results.to_csv(output_path, index=False)
    print(f"\nSaved variant diagnostics: {output_path}")


if __name__ == "__main__":
    main()
