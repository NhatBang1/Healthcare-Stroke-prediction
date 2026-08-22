# StrokeGuard AI

StrokeGuard AI là hệ thống sàng lọc nguy cơ đột quỵ kết hợp mô hình lâm sàng, ước tính `Estimated_calories` và trợ lý hội thoại an toàn. Hệ thống chỉ hỗ trợ sàng lọc và giáo dục sức khỏe; không thay thế chẩn đoán hoặc chỉ định của bác sĩ.

## Thành phần hiện tại

- `Origin_code/`: mã nguồn huấn luyện, dữ liệu và artifact mô hình.
- `Origin_code/artifacts/stroke_model_clinical.joblib`: mô hình lâm sàng được backend nạp để suy luận.
- `Origin_code/artifacts/preprocess_metadata.json`: schema tiền xử lý và ngưỡng lâm sàng hiện hành.
- `Web/be/`: backend FastAPI, LangGraph, DeepSeek extraction, Tavily search và PostgreSQL persistence.
- `Web/fe/`: giao diện chatbot, lịch sử, tiến độ, xác thực và khu vực quản trị.
- `output/doc/`: báo cáo, action plan và script thuyết trình đã cập nhật.

## Luồng hoạt động

1. Người dùng cung cấp tuổi, giới tính, huyết áp, bệnh tim, glucose và BMI.
2. DeepSeek tool trích xuất các giá trị được nói rõ; backend kiểm tra khoảng hợp lệ và evidence trước khi merge profile.
3. LangGraph chạy mô hình khi đủ sáu trường bắt buộc. Card hiển thị điểm mức độ trên thang 10, độ tin cậy, các feature và calo ước tính.
4. Agent tổng hợp diễn giải kết quả. Khi có bệnh tim, hút thuốc hoặc rượu, hệ thống có thể truy vấn nguồn y khoa qua Tavily và đưa tóm tắt ngắn kèm liên kết.
5. Câu hỏi kiến thức về yếu tố nguy cơ đi vào general chat, không tạo prediction card. Gửi lại một bộ chỉ số đầy đủ trong cùng session sẽ yêu cầu xác nhận trước khi tạo prediction mới.
6. Lịch sử hội thoại, assistant card và prediction được lưu trong PostgreSQL; session tạm thời có thể dùng Redis theo cấu hình.

## Chạy backend

```bash
cd Web/be
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Các endpoint cơ bản:

- `GET /health`
- `GET /ready`
- `POST /api/chat/message`
- `POST /api/predict/stroke`
- `GET /api/sessions/{session_id}/messages`

Không commit `.env`, DeepSeek key, Tavily key hoặc credentials PostgreSQL. Migration PostgreSQL nằm trong `Web/be/migrations/`.

## Chạy frontend

```bash
cd Web/fe
npm install
npm run dev
```

Build production:

```bash
npm run build
```

## Kiểm thử

```bash
cd Web/be
PYTHONPATH=. .venv/bin/pytest -q

cd ../fe
npm test
npm run build
```

## Huấn luyện và artifact

Pipeline gốc nằm trong `Origin_code/Code/Model.py`. Các script so sánh mô hình, đánh giá, chẩn đoán feature và kiểm tra suy luận C++ nằm cùng thư mục. Artifact mới phải được kiểm tra cùng `preprocess_metadata.json` trước khi đưa vào backend.

## Nhóm thực hiện và vai trò

Vai trò dưới đây được đối chiếu theo `output/doc/SIC_AI_Capstone Project_Final Report_G4_Code-Aligned_33P_EN.docx`:

1. **Nguyễn Nhật Bằng** - Lead Data Scientist: điều phối, phát triển Stacking, tối ưu ngưỡng, tích hợp edge/ONNX/C++ và artifact.
2. **Trần Đức Thịnh** - Data Engineer: hợp nhất bệnh nhân, làm sạch BMI, tiền xử lý, hồi quy calo và schema feature.
3. **Phạm Văn Tuấn Ninh** - Data Analyst & Model Evaluator: trực quan hóa dữ liệu, EDA, baseline, metric, biểu đồ và review kết quả.
4. **Vũ Thế Diện** - Backend & Safety Engineer: FastAPI, emergency routing, Redis session, auth, rate limit và kiểm tra deployment.

## Giới hạn an toàn

Kết quả là ước tính từ dữ liệu đầu vào và mô hình hiện hành. Khi có dấu hiệu đột quỵ cấp tính như méo miệng, yếu hoặc tê một bên, nói khó, mất thăng bằng đột ngột hay đau đầu dữ dội đột ngột, hãy gọi cấp cứu và đến cơ sở y tế ngay.
