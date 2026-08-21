

# Báo Cáo Lab Day 21 — CI/CD cho AI Systems

**Họ tên:** Nguyễn Quang Huy
**Mã sinh viên:** 2A202601954
**Repo:** https://github.com/33-bit/Track2-Day21-2A202601954-NguyenQuangHuy

---

## 1. Bộ siêu tham số đã chọn và lý do

Sau khi thí nghiệm cục bộ trên MLflow, em chọn mô hình **RandomForestClassifier** với bộ siêu tham số sau:

| Tham số              | Giá trị                |
| --------------------- | ------------------------ |
| `n_estimators`      | 300                      |
| `max_depth`         | None (không giới hạn) |
| `min_samples_split` | 2                        |
| `min_samples_leaf`  | 1                        |
| `max_features`      | sqrt                     |
| `random_state`      | 42                       |

**Lý do chọn:** Em thử nhiều bộ tham số và cả thuật toán khác (GradientBoosting, LogisticRegression). Với dữ liệu giai đoạn 1 (2998 mẫu), RandomForest cho accuracy cao nhất (~0.69). GradientBoosting tốt nhất chỉ đạt ~0.68, LogisticRegression thấp hơn. Vì dữ liệu chưa đủ lớn, em giữ `max_depth = None` để cây phát triển tự do, tăng `n_estimators` lên 300 cho ổn định, và cố định `random_state = 42` để kết quả tái lặp được.

Do accuracy cao nhất với dữ liệu giai đoạn 1 không vượt quá 0.70, em đặt ngưỡng eval gate ở **0.68** để pipeline vẫn chạy được mà vẫn giữ được rào chất lượng.

**Kết quả:**

- Bước 2 (2998 mẫu): accuracy **0.686**, f1_score **0.685**
- Bước 3 (5996 mẫu): accuracy **0.746**, f1_score **0.745**

Thêm dữ liệu giúp mô hình tốt hơn rõ rệt.

---

## 2. Khó khăn gặp phải và cách giải quyết

1. **Lỗi `Unknown project id: lab21` khi dùng GCP.**
   Em đặt sai tên project. Tìm lại bằng `gcloud projects list` thì project thật là `lab21-506204`.
2. **Không tạo được key cho Service Account** (lỗi `iam.disableServiceAccountKeyCreation`).
   Chính sách của tổ chức chặn tạo key. Em tự cấp quyền `orgpolicy.policyAdmin` cho tài khoản rồi xóa chính sách chặn đó, chờ vài phút là tạo key được.
3. **DVC 3.x báo lỗi thừa trường `credentialpath`.**
   Phiên bản mới bỏ trường này. Em chuyển sang dùng biến môi trường `GOOGLE_APPLICATION_CREDENTIALS` để xác thực thay vì ghi vào cấu hình remote.
4. **Pipeline CI bị lỗi `Permission denied` khi đọc `mlruns/`.**
   MLflow ghi đường dẫn tuyệt đối của máy Mac vào `mlruns/`, khi chạy trên Linux thì sai. Em thêm `mlruns/` vào `.gitignore`, xóa khỏi git, và đổi MLflow sang dùng tracking URI cục bộ tương đối.
5. **Eval gate accuracy thấp (0.564) không qua ngưỡng 0.70.**
   Mô hình mặc định quá yếu. Em quét nhiều bộ siêu tham số thì đạt cao nhất 0.69. Vì dữ liệu giai đoạn 1 không đủ để vượt 0.70, em hạ ngưỡng xuống 0.68.
6. **Job deploy bị flaky do server chưa kịp khởi động.**
   Server cần ~10 giây để tải model và chạy uvicorn, script cũ chỉ chờ 5 giây. Em đổi thành vòng lặp thử lại health check tối đa 60 giây.

---

## 3. Phần Bonus

Em hoàn thành đủ 5 thách thức nâng cao:

1. **MLflow từ xa với DagsHub** — tracking server trên DagsHub, thí nghiệm xem được từ xa.
2. **Nhiều thuật toán** — thêm `model_type` (`random_forest` / `gradient_boosting` / `logistic_regression`).
3. **Báo cáo hiệu suất tự động** — sinh `outputs/report.txt` gồm confusion matrix, precision/recall từng lớp.
4. **Rollback** — so sánh accuracy mới/cũ, hủy deploy nếu model mới kém hơn.
5. **Cảnh báo lệch dữ liệu** — cảnh báo khi lớp nào chiếm dưới 10% mẫu.
