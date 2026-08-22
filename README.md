# StrokeGuard AI

Trợ lý AI hỗ trợ **sàng lọc nguy cơ đột quỵ** từ hồ sơ sức khỏe cơ bản. StrokeGuard kết hợp mô hình lâm sàng, trích xuất thông tin bằng DeepSeek, tìm kiếm nguồn y khoa qua Tavily và hội thoại an toàn trên PostgreSQL.

> Đây là công cụ sàng lọc và giáo dục sức khỏe, không phải hệ thống chẩn đoán hay thay thế bác sĩ.

## Điểm chính

- Nhận diện tuổi, giới tính, huyết áp, bệnh tim, glucose và BMI từ tiếng Việt tự nhiên.
- Chạy model khi đủ sáu trường bắt buộc; hiển thị result card và `Estimated_calories`.
- Tổng hợp thêm bối cảnh bệnh tim, hút thuốc và rượu bằng nguồn y khoa đáng tin cậy.
- Phân biệt câu hỏi kiến thức với dữ liệu sàng lọc, tránh chạy model ngoài ý muốn.
- Lưu lịch sử, prediction card và phiên chat trong PostgreSQL.
- Có cảnh báo an toàn cho các dấu hiệu đột quỵ cấp tính.

## Kiến trúc

```text
Web/fe  ->  Web/be (FastAPI + LangGraph)
                    |-> DeepSeek extraction
                    |-> clinical model + Estimated_calories
                    |-> Tavily evidence search
                    |-> PostgreSQL / Redis session

Origin_code/        training code, datasets and model artifacts
```

Các thư mục quan trọng:

| Thư mục | Nội dung |
| --- | --- |
| `Web/be/` | Backend, workflow LangGraph, API, migration và test |
| `Web/fe/` | Chat UI, lịch sử, tiến độ, auth và admin pages |
| `Origin_code/` | Code huấn luyện, dữ liệu và artifact mô hình |
| `output/doc/` | Báo cáo và tài liệu dự án |

Model production được nạp từ `Origin_code/artifacts/stroke_model_clinical.joblib`; schema và ngưỡng nằm trong `preprocess_metadata.json`.

## Chạy nhanh

### Backend

```bash
cd Web/be
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

### Frontend

```bash
cd Web/fe
npm install
npm run dev
```

Build production:

```bash
npm run build
```

API chính: `GET /health`, `GET /ready`, `POST /api/chat/message`, `POST /api/predict/stroke`.

## Kiểm thử

```bash
cd Web/be
PYTHONPATH=. .venv/bin/pytest -q

cd ../fe
npm test
npm run build
```

## Artifact và huấn luyện

Pipeline huấn luyện nằm trong `Origin_code/Code/Model.py`. Trước khi thay model production, cần kiểm tra artifact cùng `preprocess_metadata.json` và chạy lại backend tests.

## Nhóm thực hiện

1. **Nguyễn Nhật Bằng** - Lead Data Scientist: điều phối, Stacking, tối ưu ngưỡng, edge/ONNX/C++ và artifact.
2. **Trần Đức Thịnh** - Data Engineer: hợp nhất bệnh nhân, làm sạch BMI, preprocessing, hồi quy calo và feature schema.
3. **Phạm Văn Tuấn Ninh** - Data Analyst & Model Evaluator: EDA, baseline, metrics, trực quan hóa và review kết quả.
4. **Vũ Thế Diện** - Backend & Safety Engineer: FastAPI, emergency routing, Redis session, auth, rate limit và deployment.
