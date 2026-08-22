import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Model import (
    ARTIFACTS_DIR,
    CALORIES_PATH,
    RANDOM_STATE,
    STROKE_PATH,
    ROOT_DIR,
    best_f1_threshold,
    build_calories_model,
    deduplicate_patients,
    preprocess_stroke_data,
    stratified_split,
    build_preprocessing_artifacts,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


RESULTS_PATH = ARTIFACTS_DIR / "model_comparison.csv"


def positive_class_weight(y):
    negative_count = int((y == 0).sum())
    positive_count = int((y == 1).sum())
    return negative_count / max(positive_count, 1)


def get_models(y_train):
    class_weight = {0: 1.0, 1: positive_class_weight(y_train)}
    models = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=80,
            max_depth=2,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=80,
            max_leaf_nodes=8,
            learning_rate=0.05,
            l2_regularization=1.0,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
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
            scale_pos_weight=positive_class_weight(y_train),
            random_state=RANDOM_STATE,
        )

    return models


def positive_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def evaluate_one_model(name, model, X_train, y_train, X_valid, y_valid, X_test, y_test):
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started

    valid_score = positive_score(model, X_valid)
    threshold, valid_f1 = best_f1_threshold(y_valid, valid_score)

    started = time.perf_counter()
    test_score = positive_score(model, X_test)
    y_pred = (test_score >= threshold).astype(int)
    predict_seconds = time.perf_counter() - started

    return {
        "model": name,
        "n_features": X_train.shape[1],
        "threshold": threshold,
        "valid_f1": valid_f1,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, test_score),
        "avg_precision": average_precision_score(y_test, test_score),
        "pred_0": int(np.sum(y_pred == 0)),
        "pred_1": int(np.sum(y_pred == 1)),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "estimator": model,
    }


def load_prepared_data():
    df_cal = pd.read_csv(CALORIES_PATH)
    calories_model = build_calories_model(df_cal)

    processed_train_path = ROOT_DIR / "Report" / "stroke_train_processed.csv"
    processed_val_path = ROOT_DIR / "Report" / "stroke_val_processed.csv"
    processed_test_path = ROOT_DIR / "Report" / "stroke_test_processed.csv"

    if processed_train_path.exists() and processed_val_path.exists() and processed_test_path.exists():
        print("\nLoading preprocessed datasets directly from Report/...")
        train_df = pd.read_csv(processed_train_path)
        valid_df = pd.read_csv(processed_val_path)
        test_df = pd.read_csv(processed_test_path)

        y_train = train_df["stroke_target"]
        X_train = train_df.drop(columns=["patient_id", "stroke_target"], errors="ignore")

        y_valid = valid_df["stroke_target"]
        X_valid = valid_df.drop(columns=["patient_id", "stroke_target"], errors="ignore")

        y_test = test_df["stroke_target"]
        X_test = test_df.drop(columns=["patient_id", "stroke_target"], errors="ignore")

        artifacts = build_preprocessing_artifacts(pd.read_csv(STROKE_PATH))
        artifacts["feature_columns"] = X_train.columns.tolist()
    else:
        df_stroke = deduplicate_patients(pd.read_csv(STROKE_PATH))
        train_df, valid_df, test_df = stratified_split(df_stroke)

        _, X_train, y_train, artifacts = preprocess_stroke_data(train_df, calories_model)
        _, X_valid, y_valid, _ = preprocess_stroke_data(
            valid_df, calories_model, artifacts=artifacts
        )
        _, X_test, y_test, _ = preprocess_stroke_data(
            test_df, calories_model, artifacts=artifacts
        )
    return X_train, y_train, X_valid, y_valid, X_test, y_test, artifacts


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_valid, y_valid, X_test, y_test, artifacts = load_prepared_data()

    results = []
    best = None
    for name, model in get_models(y_train).items():
        print(f"\nTraining {name}...")
        result = evaluate_one_model(
            name, model, X_train, y_train, X_valid, y_valid, X_test, y_test
        )
        estimator = result.pop("estimator")
        results.append(result)
        if best is None or (result["f1"], result["auc_roc"]) > (best["result"]["f1"], best["result"]["auc_roc"]):
            best = {"result": result, "model": estimator}
        print(
            f"{name}: f1={result['f1']:.4f}, recall={result['recall']:.4f}, "
            f"precision={result['precision']:.4f}, auc={result['auc_roc']:.4f}, "
            f"threshold={result['threshold']:.3f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        ["f1", "auc_roc", "recall"], ascending=False
    )
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\n===== Model Comparison On Deduplicated Test Set =====")
    print(results_df.to_string(index=False))
    print(f"\nSaved comparison: {RESULTS_PATH}")
    print(f"Best by F1: {results_df.iloc[0]['model']}")
    print("Features used:")
    for feature in artifacts["feature_columns"]:
        print(f"- {feature}")


if __name__ == "__main__":
    main()
