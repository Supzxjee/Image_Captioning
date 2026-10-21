# UPT hidden-pair A/H pilot — kết quả

**Quyết định: NO-GO cho cấu hình H hiện tại; chưa chứng minh tăng chất lượng so với A.**

Train 10.000 ảnh, development 1.000 ảnh, cùng base/seed/1250 updates. 200 bước CE rồi CE + .05 SCST; không dùng test.

| Mô hình | CIDEr ↑ | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Caption ảo giác |
|---|---:|---:|---:|---:|---:|
| original | 1.189066 | 5.80% | 3.545% | 45.681% | 58/1000 |
| A | 1.205513 | 5.20% | 3.248% | 45.804% | 52/1000 |
| H | 1.201124 | 5.20% | 3.218% | 46.019% | 52/1000 |
| H_off | 1.202134 | 5.20% | 3.210% | 46.204% | 52/1000 |

CIDEr giữ thang của evaluator (không nhân 100); không có miền tối đa cố định.

| So sánh CIDEr | Δ | Paired bootstrap 95% CI |
|---|---:|---|
| H_minus_A | -0.004389 | [-0.023220, +0.014610] |
| H_minus_original | +0.012058 | [-0.010463, +0.034594] |
| A_minus_original | +0.016447 | [-0.003513, +0.036113] |
| H_minus_H_off | -0.001010 | [-0.007387, +0.005073] |

Bootstrap 20.000 lần, seed42, ghép theo image_id; không phản ánh phương sai seed hoặc việc đã xem dev nhiều lần.

H on/off đổi 48/1000 caption; đổi cặp UPT đổi 36/1000. Có 654 ảnh có cặp, 346 ảnh không có cặp; nhóm không cặp không đổi ở cả hai can thiệp.

H có gradient nhánh và can thiệp gây đổi caption, nên không phải nhánh hoàn toàn bị ngắt. Tuy nhiên, bật memory không làm CIDEr tốt hơn tắt memory; độ nhạy không chứng minh quan hệ đúng.

Phân nhóm chẩn đoán H−A: ảnh có cặp +.002564 (CI chứa0), không cặp −.017531 (CI chứa0). Không dùng nhóm thuận lợi để thay tiêu chí toàn bộ đã khóa. H và A có base weights cuối khác nhau; ảnh không cặp vẫn có thể đổi vì quá trình cập nhật base khác nhau. Đây là giải thích cơ chế có thể xảy ra, không phải bằng chứng rằng nhánh UPT gây lỗi ngữ nghĩa ở từng ảnh.

## Kiểm tra và khuyến nghị

Hai protocol chỉ khác arm/output/staging/conditions. Source, cohort, receipt, targets, caption gốc và evaluator reference khớp; hai rank hoàn tất, history hữu hạn và lịch CE/SCST đúng. Kaggle preflight trong log qua; pip có cảnh báo xung đột dependency nhưng không thấy lỗi kiểm tra hoặc thiếu bước huấn luyện.

Không mở rộng full train hoặc thêm seed chỉ để cứu cấu hình này. Giữ A làm mốc; báo kết quả cho Astra để quyết định bước chẩn đoán hoặc thiết kế tiếp. Không tự kết luận UPT vô ích hay mọi hướng căn chỉnh không khả thi. Chưa cần yêu cầu người dùng chấm thêm nhiều ảnh để xác nhận một gain chưa xuất hiện.

Chưa thực hiện human factual review; chưa xác minh ngữ nghĩa bằng xem ảnh. CHAIR không đánh giá đúng/sai quan hệ. Review ZIP không có checkpoint; chỉ kiểm tra hash worker báo cáo.
