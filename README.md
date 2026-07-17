# Tích hợp Mô hình Học máy Đa nguồn Dự báo Nguy cơ Đột quỵ với Ước lượng Tiêu hao Năng lượng Nhân tạo
## SIC Capstone Project - Nhóm 4 (Healthcare Analytics)

Dự án này nghiên cứu xây dựng hệ thống học máy tích hợp đa nguồn dữ liệu nhằm dự báo nguy cơ đột quỵ sớm ở bệnh nhân, đồng thời tích hợp chỉ số hành vi ước lượng tiêu hao năng lượng nhân tạo (`Estimated_calories`) làm đặc trưng bổ trợ và chuyển đổi mô hình sang ngôn ngữ C++ phục vụ triển khai nhúng biên (edge computing) với độ trễ cực thấp.

---

## 📂 Cấu trúc thư mục (Directory Structure)

```text
Healthcare-Stroke-prediction/
├── Code/                               # Mã nguồn Python chính của dự án
│   ├── Model.py                        # Pipeline chính: huấn luyện mô hình Calo & Đột quỵ, lưu trữ mô hình và transpilation sang C++
│   ├── compare_models.py               # Chạy và so sánh hiệu năng của 6 thuật toán phân loại
│   ├── evaluate_model.py               # Đánh giá các mô hình đã lưu và kiểm chứng độ chính xác với ONNX Runtime
│   ├── feature_diagnostics.py          # Chẩn đoán đa cộng tuyến (VIF), rò rỉ dữ liệu (data leakage) & SHAP
│   ├── generate_figures.py             # Vẽ biểu đồ ma trận nhầm lẫn (Confusion Matrix) & đường cong ROC
│   └── generate_paper_icpu.py          # Tạo bài báo định dạng Word & PDF khớp chuẩn với template hội nghị
├── Calo_burn/                          # Tập dữ liệu hoạt động thể chất bổ trợ
│   └── calories.csv                    # Tập dữ liệu calo thô (15,000 dòng)
├── Stroke_dt/                          # Tập dữ liệu lâm sàng đột quỵ
│   ├── SSHP_Data Descriptipn.docx      # Mô tả chi tiết các thuộc tính dữ liệu lâm sàng
│   └── Stroke_dt.csv                   # Tập dữ liệu đột quỵ thô gốc (143,960 dòng)
├── Documents/                          # Các văn bản báo cáo & bài báo khoa học xuất bản
│   ├── Paper_ICPU.docx                 # Bài báo khoa học chuẩn hội nghị định dạng Word
│   └── Paper_ICPU.pdf                  # Bài báo khoa học xuất bản định dạng PDF (đã làm sạch cấu trúc thẻ)
├── artifacts/                          # Trọng số mô hình, siêu dữ liệu và biểu đồ đầu ra
│   ├── calories_model.joblib           # Trọng số mô hình hồi quy Calo (joblib)
│   ├── stroke_model.joblib             # Trọng số mô hình phân loại đột quỵ (joblib)
│   ├── stroke_predictor_model.h        # Mã nguồn header C++ chứa tham số mô hình nhúng
│   ├── confusion_matrix.png            # Biểu đồ ma trận nhầm lẫn tối ưu ngưỡng
│   ├── roc_curves.png                  # Đường cong ROC của 6 mô hình so sánh
│   ├── model_comparison.csv            # Kết quả số liệu so sánh chi tiết các mô hình
│   └── ...                             # Các đồ thị SHAP và metadata tiền xử lý
├── SIC_report/                         # Các báo cáo tiến độ và báo cáo cuối kỳ (Jupyter Notebooks)
│   ├── Final_Project_Report.ipynb      # Báo cáo dự án cuối kỳ (Final Capstone Report)
│   ├── Report1_Introduction.ipynb      # Báo cáo Tiến độ 1 (Project Proposal)
│   ├── Report2_EDA.ipynb               # Báo cáo Tiến độ 2 (EDA)
│   ├── Report3_Modeling.ipynb          # Báo cáo Tiến độ 3 (Model Training & Evaluation)
│   └── Report4.ipynb                   # Báo cáo Tiến độ 4 (Feature Diagnostics & SHAP)
├── requirements.txt                    # Danh sách các thư viện Python phụ thuộc
├── LICENSE                             # Bản quyền mã nguồn mở của dự án
└── README.md                           # Tài liệu hướng dẫn sử dụng này
```

---

## 🛠️ Hướng dẫn vận hành hệ thống (Usage Guide)

### 1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt các thư viện Python cần thiết được liệt kê trong `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Huấn luyện Pipeline chính (`Model.py`)
Tập lệnh [Model.py](file:///D:/Bang/Learning/FPT/Semester_8/DSP391/PRJ2/Healthcare-Stroke-prediction/Code/Model.py) thực hiện toàn bộ quy trình:
- Tiền xử lý dữ liệu lâm sàng đột quỵ thô và loại bỏ trùng lặp ở cấp độ bệnh nhân (Patient-level deduplication).
- Huấn luyện mô hình hồi quy XGBoost trên tập dữ liệu Calo phụ trợ để ước lượng calo cho bệnh nhân đột quỵ dựa trên tuổi, giới tính và BMI.
- Huấn luyện mô hình XGBoost phân loại đột quỵ trên tập dữ liệu đã khử trùng lặp và làm giàu đặc trưng.
- Thực hiện tối ưu hóa ngưỡng quyết định lâm sàng (clinical decision threshold) nhằm cực đại hóa chỉ số F1 và Recall (giảm thiểu chẩn đoán bỏ sót nguy cơ).
- Xuất bản các mô hình đã huấn luyện sang định dạng `joblib`, `json`, `onnx` và chuyển đổi trực tiếp thuật toán suy luận sang file header C++ tuần tự hóa (`stroke_predictor_model.h`) lưu vào thư mục `artifacts/`.

Chạy pipeline bằng lệnh:
```bash
python Code/Model.py
```

### 3. Chẩn đoán đặc trưng & Rò rỉ dữ liệu (`feature_diagnostics.py`)
Tập lệnh [feature_diagnostics.py](file:///D:/Bang/Learning/FPT/Semester_8/DSP391/PRJ2/Healthcare-Stroke-prediction/Code/feature_diagnostics.py) thực hiện:
- Kiểm tra tính rò rỉ thông tin (data leakage) giữa tập train và test khi phân chia theo hàng (Row split) so với phân chia theo bệnh nhân (Patient split).
- Tính toán hệ số đa cộng tuyến VIF (Variance Inflation Factor) cho các đặc trưng lâm sàng đột quỵ.
- Chạy phân tích SHAP giải thích mô hình và chẩn đoán hướng tác động của đặc trưng.

Chạy lệnh:
```bash
python Code/feature_diagnostics.py
```

### 4. So sánh hiệu năng các thuật toán (`compare_models.py`)
Tập lệnh [compare_models.py](file:///D:/Bang/Learning/FPT/Semester_8/DSP391/PRJ2/Healthcare-Stroke-prediction/Code/compare_models.py) chạy thử nghiệm huấn luyện và đánh giá chéo 6 mô hình phân loại:
1. Logistic Regression (LR)
2. Random Forest (RF)
3. Extra Trees (ET)
4. Gradient Boosting (GB)
5. Histogram Gradient Boosting (HGB)
6. XGBoost Classifier (XGB)

Kết quả so sánh các số liệu y tế (Accuracy, Precision, Recall, F1-score, AUC-ROC, AP) sẽ được lưu vào file [model_comparison.csv](file:///D:/Bang/Learning/FPT/Semester_8/DSP391/PRJ2/Healthcare-Stroke-prediction/artifacts/model_comparison.csv).

Chạy lệnh:
```bash
python Code/compare_models.py
```

### 5. Vẽ biểu đồ trực quan hóa (`generate_figures.py`)
Tập lệnh [generate_figures.py](file:///D:/Bang/Learning/FPT/Semester_8/DSP391/PRJ2/Healthcare-Stroke-prediction/Code/generate_figures.py) sinh ra các biểu đồ phục vụ cho báo cáo và bài báo khoa học:
- Đường cong ROC so sánh 6 thuật toán (`roc_curves.png`).
- Ma trận nhầm lẫn y tế tối ưu ngưỡng (`confusion_matrix.png`).

Chạy lệnh:
```bash
python Code/generate_figures.py
```

### 6. Đánh giá kiểm chứng suy luận (`evaluate_model.py`)
Tập lệnh [evaluate_model.py](file:///D:/Bang/Learning/FPT/Semester_8/DSP391/PRJ2/Healthcare-Stroke-prediction/Code/evaluate_model.py) thực hiện tải mô hình từ thư mục `artifacts/` và suy luận trên tập kiểm thử để so sánh đầu ra của mô hình Python gốc so với mô hình ONNX Runtime nhằm đảm bảo tính toàn vẹn và sai số tương đương bằng 0.

Chạy lệnh:
```bash
python Code/evaluate_model.py
```

### 7. Xuất bản bài báo khoa học (`generate_paper_icpu.py`)
Tập lệnh [generate_paper_icpu.py](file:///D:/Bang/Learning/FPT/Semester_8/DSP391/PRJ2/Healthcare-Stroke-prediction/Code/generate_paper_icpu.py) nạp mẫu bài báo Word từ desktop, tự động điền tiêu đề bài báo, danh sách 4 tác giả (không chứa email theo yêu cầu phản biện mù), nội dung khoa học, bảng biểu kết quả học máy và chèn các biểu đồ (ROC, CM, SHAP). Đồng thời, tập lệnh tự động gọi Microsoft Word thông qua COM automation để xuất bản ra tệp PDF đã làm sạch hoàn toàn các cấu trúc thẻ hỗ trợ tiếp cận (accessibility tags) và metadata, khắc phục hoàn toàn lỗi chuyển đổi định dạng (TET PDF IFilter) khi tải lên cổng thông tin hội nghị.

Chạy lệnh:
```bash
python Code/generate_paper_icpu.py
```

---

## 👥 Nhóm Thực Hiện & Vai trò (Team & Contributions)

* **Giảng viên hướng dẫn:** Cao Văn Mai
* **Đề tài Capstone:** SIC Capstone Project - Nhóm 4 (Healthcare Analytics)
* **Thành viên nhóm:**
  1. **Nguyễn Nhật Bằng** - Trưởng nhóm / Lead Data Scientist: Phụ trách thiết lập hệ thống, phát triển mô hình Stacking, tối ưu hóa ngưỡng lâm sàng và tích hợp suy luận C++.
  2. **Vũ Thế Diện** - Thành viên / Data Engineer: Phụ trách tiền xử lý, làm sạch dữ liệu BMI nhiễu, huấn luyện mô hình hồi quy Calo để sinh đặc trưng nhân tạo.
  3. **Phạm Tuấn Ninh** - Thành viên / Data Analyst: Phụ trách phân tích khám phá dữ liệu (EDA), phân tích đơn biến, đa biến, vẽ biểu đồ tương quan.
  4. **Đức Thịnh** - Thành viên / Researcher: Phụ trách khảo sát các nghiên cứu liên quan (Related Works), phát hiện nhiễu y tế (red-flag features) và khảo sát dịch tễ đột quỵ.
  5. **Nguyễn Văn Sơn** - Thành viên / Systems Engineer: Phụ trách kiểm thử suy luận, phân tích độ quan trọng đặc trưng (SHAP), kiểm chứng C++ và đánh giá ablation study.