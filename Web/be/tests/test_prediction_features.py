from app.prediction.preprocess import build_feature_frame


class NegativeCaloriesModel:
    def predict(self, features):
        return [-12.5]


def test_estimated_calories_is_non_negative_and_kept_in_feature_vector():
    metadata = {
        "numeric_fill": {},
        "categorical_fill": {
            "ever_married": "No",
            "work_type": "Private",
            "Residence_type": "Urban",
            "smoking_status": "unknown",
        },
        "selected_features": ["age", "gender", "bmi", "Estimated_calories"],
    }

    frame, _, _ = build_feature_frame(
        {"age": 20, "gender": "male", "bmi": 25},
        metadata,
        NegativeCaloriesModel(),
    )

    assert frame.iloc[0].to_dict()["Estimated_calories"] == 0.0
    assert "Estimated_calories" in frame.iloc[0].to_dict()


def test_feature_frame_is_compatible_with_sklearn_estimators():
    metadata = {
        "numeric_fill": {},
        "categorical_fill": {
            "ever_married": "No",
            "work_type": "Private",
            "Residence_type": "Urban",
            "smoking_status": "unknown",
        },
        "selected_features": ["age", "gender", "bmi", "Estimated_calories"],
    }

    class Estimator:
        def predict_proba(self, values):
            assert hasattr(values, "columns")
            return [[0.9, 0.1]]

    frame, _, _ = build_feature_frame(
        {"age": 20, "gender": "male", "bmi": 25},
        metadata,
        NegativeCaloriesModel(),
    )
    Estimator().predict_proba(frame)
