from __future__ import annotations


FIELD_LABELS = {
    "age": "tuổi",
    "gender": "giới tính",
    "hypertension": "huyết áp tâm thu/tâm trương (ví dụ 130/80 mmHg)",
    "heart_disease": "tiền sử bệnh tim (có/không)",
    "avg_glucose_level": "glucose trung bình",
    "bmi": "BMI",
}


def missing_information_reply(fields: list[str], profile: dict | None = None) -> str:
    labels = [FIELD_LABELS.get(field, field) for field in fields]
    profile = profile or {}
    category_label = {
        "normal": "bình thường",
        "elevated": "cao hơn bình thường",
        "stage_1": "tăng huyết áp độ 1",
        "stage_2": "tăng huyết áp độ 2",
        "severe": "rất cao",
    }.get(profile.get("blood_pressure_category"))
    bp_note = ""
    if category_label and profile.get("systolic_bp") is not None and profile.get("diastolic_bp") is not None:
        systolic = f"{float(profile['systolic_bp']):g}"
        diastolic = f"{float(profile['diastolic_bp']):g}"
        bp_note = (
            f" Huyết áp {systolic}/{diastolic} mmHg được xếp loại {category_label} theo ngưỡng người lớn; "
            "ngưỡng này không tách riêng nam/nữ."
        )
    return (
        "Mình đã ghi nhận các thông tin bạn vừa cung cấp. Để chạy model, "
        f"bạn vui lòng bổ sung: {', '.join(labels)}. "
        "Nếu không biết một chỉ số, hãy nói rõ “không biết” để mình không tự suy đoán."
        f"{bp_note}"
    )


def extraction_unavailable_reply() -> str:
    return (
        "Hệ thống tạm thời chưa thể đọc các chỉ số trong tin nhắn nên chưa chạy model. "
        "Thông tin của bạn chưa bị tự suy đoán; vui lòng thử lại sau."
    )


def emergency_reply() -> str:
    return (
        "Các dấu hiệu bạn mô tả có thể cần được đánh giá cấp cứu. Hãy gọi cấp cứu địa phương "
        "hoặc đến cơ sở y tế ngay, đặc biệt nếu triệu chứng xuất hiện đột ngột. Không tự lái xe."
    )


def prediction_reply(result: dict) -> str:
    probability = round(float(result.get("risk_probability", 0.0)) * 100, 2)
    score_10 = round(min(10.0, max(0.0, probability)), 2)
    label = {
        "low_risk": "Thấp",
        "medium_risk": "Trung bình",
        "high_risk": "Cao",
    }.get(result.get("risk_label"), str(result.get("risk_label", "Chưa xác định")))
    return (
        f"Tôi đã chạy model sàng lọc. Mức nguy cơ: {label} ({score_10:.2f}/10). "
        "Kết quả chỉ dùng để sàng lọc và không thay thế chẩn đoán của bác sĩ."
    )
