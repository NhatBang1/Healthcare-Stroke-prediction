import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from Model import (
    ARTIFACTS_DIR,
    CALORIES_PATH,
    STROKE_PATH,
    build_calories_model,
    deduplicate_patients,
    preprocess_stroke_data,
    stratified_split,
    train_pipeline,
)


def get_positive_probability_from_onnx(outputs):
    probabilities = outputs[-1]
    if isinstance(probabilities, list):
        return np.array([row.get(1, row.get("1", 0.0)) for row in probabilities])
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 2 and probabilities.shape[1] > 1:
        return probabilities[:, 1]
    return probabilities.reshape(-1)


def evaluate_onnx_against_python(stroke_model, X_test):
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nSkip ONNX runtime check: install onnxruntime.")
        return

    onnx_path = ARTIFACTS_DIR / "stroke_model.onnx"
    if not onnx_path.exists():
        print("\nSkip ONNX runtime check: stroke_model.onnx was not created.")
        return

    sample = X_test.iloc[: min(100, len(X_test))].astype(np.float32)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: sample.to_numpy()})
    onnx_prob = get_positive_probability_from_onnx(outputs)
    python_prob = stroke_model.predict_proba(sample)[:, 1]
    max_diff = float(np.max(np.abs(onnx_prob - python_prob)))

    print("\n===== ONNX Runtime Check =====")
    print(f"Samples checked: {len(sample)}")
    print(f"Max probability diff vs Python model: {max_diff:.8f}")


def evaluate_saved_model():
    metadata_path = ARTIFACTS_DIR / "preprocess_metadata.json"
    stroke_path = ARTIFACTS_DIR / "stroke_model.joblib"
    calories_path = ARTIFACTS_DIR / "calories_model.joblib"

    if not stroke_path.exists() or not calories_path.exists() or not metadata_path.exists():
        print("Saved artifacts not found. Training model first...")
        result = train_pipeline(save_models=True, show_shap=False)
        return result["stroke_model"], result["calories_model"], result

    import json

    stroke_model = load(stroke_path)
    calories_model = load(calories_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    df_stroke = deduplicate_patients(pd.read_csv(STROKE_PATH))
    _, _, test_df = stratified_split(df_stroke)
    _, X_test, y_test, _ = preprocess_stroke_data(
        test_df,
        calories_model,
        artifacts=metadata,
    )
    return stroke_model, calories_model, {
        "X_test_pro": X_test,
        "y_test": y_test,
        "threshold": metadata["decision_threshold"],
    }


def main():
    stroke_model, _, result = evaluate_saved_model()
    X_test = result["X_test_pro"]
    y_test = result["y_test"]
    threshold = result["threshold"]

    y_prob = stroke_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print("\n===== Saved Model Evaluation =====")
    print("Threshold:", threshold)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_prob))
    print("Average precision:", average_precision_score(y_test, y_prob))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

    evaluate_onnx_against_python(stroke_model, X_test)


if __name__ == "__main__":
    main()
