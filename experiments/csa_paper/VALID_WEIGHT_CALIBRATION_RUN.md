# Chạy thử trọng số reranking trên VALID

Import `VALID_WEIGHT_CALIBRATION_CPU.ipynb` lên Kaggle, chọn **Accelerator None**, Internet ON, rồi Save & Run All. Không thêm Dataset, token hoặc checkpoint. Notebook tải lại đúng text/điểm đã lưu của CONTROL180/PROMPT60 từ GitHub theo commit và SHA.

Chỉ chạy một lần theo [protocol](VALID_WEIGHT_CALIBRATION_PROTOCOL.md). Cùng một bộ trọng số cho hai cấu hình; chọn trên 2000 ảnh, khóa lựa chọn, kiểm tra trên 3000 ảnh còn lại. Không thay câu, không train, không chạy TEST. Nếu không qua điều kiện CIDEr/CHAIR/recall thì giữ bộ cũ. Điểm CIDEr từng phần dùng corpus tham chiếu riêng, không so trực tiếp với số fullVALID5000.

Gửi lại `valid_weight_calibration_v1_review.zip`. Giữ folder `/kaggle/working/valid_weight_calibration_v1` để có log, điểm từng ứng viên, split và selection lock. Giới hạn một giờ CPU là STOP cap, không phải thời gian dự báo.

Đã kiểm tra local: 4 synthetic unit tests và 30002 metadata/original-selector replay assertions; source/notebook parse. Không chạy official scorer, calibration search hoặc model local. Chưa có kết quả cải thiện. VALID đã dùng trong phát triển: đây là kiểm tra nội bộ, không phải test độc lập hay bằng chứng ý nghĩa thống kê.
