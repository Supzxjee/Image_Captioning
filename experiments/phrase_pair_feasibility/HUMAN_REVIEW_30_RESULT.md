# Hoàn tất audit30 nhóm human — 20/10/2026

**Đủ cơ sở để dừng đợt chấm này và chuyển sang chuẩn bị một phép thử kỹ thuật nhỏ; chưa có bằng chứng cải thiện mô hình.**

Đã nhận30/30 nhóm chấm đủ trường, đúng manifest hash/schema/IDs/enums. Nhãn10 nhóm đầu không đổi so với bản trước. Một người tự báo cáo chấm, chưa có đánh giá độc lập/adjudication. JSON gốc giữ local; bản kết quả public không ghi tên người chấm.

## Kết quả theo nhãn người dùng

| Tiêu chí | Số nhóm |
|---|---:|
| Đã chấm đủ trường | 30/30 |
| Ma trận2×2 đối lập rõ | 26/30 |
| Đối lập rõ và cả4box đúng | 26/30 |
| Thêm điều kiện cả hai event nhìn rõ ở224×224 | **21/30** |
| Trong21 nhóm: cùng ảnh, hai cặp khác nhau | 4 |
| Trong21 nhóm: cả hai cặp khớp selected UPT objects ởIoU≥.5 | 17 |

21 nhóm gồm42 event views/38 ảnh duy nhất: skateboard6, surfboard7, phone7, bicycle1. Không có chair/horse trong tập này. 38/42 event views có selected UPT object pair khớp; đây không phải raw detector recall hoặc predicate accuracy.

7/42 event views vẫn có ít nhất một đầu không chứa tâm patch14×14 dù human chấm bằng chứng còn nhìn được ở224. Cần báo/can thiệp nearest-patch fallback khi chuẩn bị model; không tự sửa nhãn visibility, không khẳng định CLIP có hoặc không có tín hiệu.

## Giữ riêng9 nhóm

| Nhóm | Lý do không đưa vào bộ21 |
|---|---|
| 1,13 | Cả hai event cùng hỗ trợ C1, không tạo đối chiếu hai chiều |
| 20 | Event1 bị chấm sai cho cả C1/C2, không có positive đối ứng |
| 28 | Một ô đối chiếu `unclear`; không biến thành negative |
| 4,5,14,15,22 | Ma trận ngữ nghĩa rõ nhưng bằng chứng224 ít nhất một event `unclear` |

Không bỏ JSON gốc, không sửa câu trả lời để khớp source action/proxy. Các event riêng lẻ có thể có giá trị khác, nhưng chưa tự đưa chúng vào loss hai chiều hoặc một split train.

## Đóng gói và kiểm tra

- `HUMAN_REVIEW_30_RESULT.json`: summary và từng lý do phân loại.
- `FEASIBLE_GROUPS_21.json`: subset21 với ma trận human và target index từng event **suy từ nhãn thật**, không từ hướng mining.
- `audit_human.py`: tái tạo hai artifact từ file review do người dùng cung cấp; không chạy model.
- 6data tests PASS, gồm kiểm tra targets của bộ21 theo ma trận human, IDs/hash, không chia ảnh qua các nhóm. Không có GPU/model inference/training.

Bộ21 có trạng thái `UNASSIGNED_FEASIBILITY`, `training_approved=false`. Đây là xác nhận khả thi của dữ liệu, không phải phê duyệt khoa học cho một benchmark hoặc một bộ train hoàn chỉnh.

## Bước kế tiếp và giới hạn

**Không yêu cầu chấm thêm lúc này.** Chuẩn bị protocol kỹ thuật nhỏ **CE so với CE+đối chứng hai chiều**, cùng encoder/adapter, cùng ảnh/nhãn/khởi tạo/budget. Phải khóa phân nhóm adaptation fit/holdout và lịch trước khi chạy; chưa tạo split trong audit này. Người dùng tự chạy notebook; assistant không tự launch.

Mục tiêu là kiểm tra đường loss→decoder→bằng chứng và sự khác nhau ở sinh cụm, không chỉ giảm loss. Bộ21 quá nhỏ để khẳng định hiệu quả tổng quát; nếu chia một phần làm holdout, độ bất định sẽ lớn. Các mẫu đã được người dùng chấm và dùng quyết định thiết kế; gọi đúng là holdout của adaptation nếu có, không phải test độc lập mới.

Tập dùng một số mẫu câu cố định, bốn họ cấu hình quan sát được, phần lớn khác ảnh. Vì vậy language/geometry/background shortcuts và khái quát sang động từ/paraphrase mới chưa được giải quyết. Không đồng nhất việc phân biệt posture/phone position với hiểu mọi hành động. Các kiểm tra cùng ảnh có ích nhưng chỉ có4 nhóm; không đủ tuyên bố binding tổng quát.

Nếu scoring tăng mà greedy không tốt hơn, hoặc model chỉ học các template, chưa scale. Ngay cả sinh cụm tốt hơn vẫn cần milestone caption matched tiếp theo; chưa có CIDEr/CHAIR hay lợi ích caption mới từ đợt audit này.
