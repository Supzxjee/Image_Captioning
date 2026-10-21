# VG region-caption G — kết quả sơ bộ so với A

**G hoàn tất đúng cấu hình, nhưng chưa đạt gate tăng chất lượng đã khóa. Chờ T để kiểm tra giá trị của vùng.**

| Mô hình | CIDEr ↑ | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Caption ảo giác |
|---|---:|---:|---:|---:|---:|
| original | 1.189066 | 5.80% | 3.545% | 45.681% | 58/1000 |
| A | 1.205513 | 5.20% | 3.248% | 45.804% | 52/1000 |
| G | 1.208401 | 5.30% | 3.307% | 45.619% | 53/1000 |

CIDEr giữ thang evaluator, không nhân100. A là baseline huấn luyện tiếp cùng1250updates; original là checkpoint trước pilot.

| So sánh | Δ CIDEr | Paired bootstrap95% CI |
|---|---:|---|
| G_minus_A | +0.002888 | [-0.017607, +0.023641] |
| G_minus_original | +0.019335 | [+0.000363, +0.038446] |
| A_minus_original | +0.016447 | [-0.003513, +0.036113] |

Bootstrap20000resamples,seed42,1000ảnh ghép theoID; không phải xác nhận trên test hoặc đo độ ổn định nhiều seed.

## Diễn giải

G−A chỉ khoảng+.002888, dưới ngưỡng+.01; CI chứa0. Không được kết luận G chắc chắn tốt hơn hoặc tương đương A. CHAIRs tăng52→53caption, unsupported mentions53→54; recall giảm1490→1484đối tượng duy nhất. CHAIRs/i không đạt điều kiện không tăng. Recall vẫn trong mức giảm tối đa.01 đã cho phép.

G tăng so original, nhưng A không có nhiệm vụ phụ cũng tăng; không gán toàn bộ G−original cho căn chỉnh vùng. Chưa có T nên chưa phân biệt tác dụng vùng với thêm văn bản/gradient. Không mở full train hoặc tune để cứu điểm. Giữ T đang chạy để hoàn tất đối chứng nếu ngân sách cho phép, không đổi cấu hình sau khi xem G.

## Kiểm tra

Cùng base initialization,cohort,stage hash,targets,legacy/cache,optimizer,runtime,evaluator reference và caption gốc với A. Common source chỉ khác run_pilot.py ở hook diagnostic2dòng; mỗi runtime source hash khớp protocol. Hai rank đủ1250updates, số tham số như A (33520400), không thêm branch; lịchCE/SCST/loss+.1phraseCE được đối chiếu từng step và hữu hạn. Synthetic model/mask/RNG tests vàDDP G/Tsmoke qua trong log Kaggle.

Tổng10000lượt phrase; trung bình32.26patch/vùng. Phrase CE trung bình100bước đầu4.1717,100bước cuối3.2025; đây là số mô tả trên minibatch khác nhau, không phải metric heldout. Giảm phrase loss không chứng minh grounding đúng.
Huấn luyện khoảng8.02phút (không gồm staging,khởi tạoCIDEr hay đánh giá). G/A khác392/1000caption, chưa chấm bằng ảnh.

Không chạy model/training/inference/metric mới trên local. Checkpoint không nằm trongZIP; chỉ có worker-reported hashes. T chưa nhận, human review chưa thực hiện.
