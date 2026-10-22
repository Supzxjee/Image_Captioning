# Sau smoke PASS: chính sách dữ liệu trước pilot B/R

**PLAN_LOCKED_BEFORE_FULL_ANNOTATION_ANALYSIS.** Astra đã đọc summary và xác nhận technical gate theo audit; không đọc lại ZIP hoặc chạy model. Chính sách này chưa được áp dụng lên annotation đầy đủ. Chưa tải file mới, chưa tạo manifest hoặc notebook pilot.

## Phạm vi

Lấy official COCO 2014 instance annotations train+val, lưu URL/hash, chỉ giữ đúng train10k đã khóa theo Karpathy. Ảnh từ file val2014 có thể thuộc restval TRAIN; không dùng tên file để quyết split. Không lấy held-out làm supervision. Thiếu annotation/box là thiếu supervision, không phải negative.

Không thêm teacher, VG, parser/LLM mới, attention rollout hoặc vòng smoke riêng.

## Mapping trước khi xem số đếm mới

- Giữ longest alias matching theo whitespace-token positions và các ambiguity exclusions đã có. Không thêm lại train/orange/remote/table để tăng coverage.
- Loại các compound đổi đối tượng, ví dụ pizza box không tự thành pizza; danh sách loại trừ hữu hạn phải được ghi rõ trong manifest/policy trước khi chạy counts. Chưa mặc định danh sách đã hoàn chỉnh.
- Chỉ giữ lớp có **đúng một instance annotation trong ảnh**, instance không crowd. Đếm mọi instance trước khi loại crowd để tránh tưởng lớp là single-instance.
- Loại span truncate, UNK hoặc sai vocabulary/offset. Notebook sẽ kiểm lại tokenizer thật.
- Bbox hợp lệ; fractional bbox–patch intersection, không segmentation/nearest-patch fallback/nới box.
- Diện tích bbox tương đương **>=4 patch và <=50% ảnh**.
- Lấy mẫu xác định trước để **person <=25% supervision spans**. Caption chính không bị loại khỏi train.

Đây là mapping yếu ở mức lớp, không human grounding gold; depiction, compound/context và nghĩa từ vẫn có rủi ro.

## Gate đầu tư, không điều chỉnh sau counts

Sau lọc và cap person, cần đồng thời:

1. Ít nhất **2.000 ảnh** có supervision.
2. Ít nhất **5.000 unique spans**.
3. Ít nhất **15 lớp**, mỗi lớp >=50 spans.

Đây là mốc thực dụng cho pilot, không bảo đảm thống kê/gain. Kiểm thêm **16 bản ghi chọn xác định trước, phủ lớp**, về context/ID/bbox/token, không chọn theo model/CIDEr. Nếu lỗi hệ thống, sửa quy tắc chung trước manifest cuối, ghi rõ thay đổi; không nới gate để bù counts.

Nếu không đạt, dừng trước pilot và báo attrition theo từng rule; không tự mở nguồn/vòng screen khác. Nếu đạt, khóa manifest span–class–bbox–mask và supervision schedule dùng chung B/R, rồi chốt recipe trước training.

## Một pilot, hai nhánh mới

Giữ raw last-layer head-mean attention từ smoke: `a[t,p]=sum_q D[t,q]*C[q,p]`, không normalize lại Q/spatial slices. Với `S=sum_p a[t,p]`, `R=sum_p mask[p]*a[t,p]`, epsilon như smoke:

- **B**: `L_B=-log(S+epsilon)` — chỉ khuyến khích đường Q→spatial patch, không chỉ định vị trí.
- **R**: `L_R=-log(R+epsilon)` — thêm vị trí đúng; tương đương `L_B-log((R+epsilon)/(S+epsilon))` khi cùng epsilon, không detach S.

Cùng data/span/schedule/aux weight và baseline recipe. Không dùng random/incorrect mask làm nhãn âm. R>B mới hỗ trợ giá trị thành phần vị trí trong recipe; chưa phải causal grounding. Hai loss có gradient magnitude khác, phải ghi norm và global clipping; “cùng aux weight” không làm optimizer dynamics tự động bằng nhau.

So R với A chỉ khi khớp recipe. Tiếp tục cần R vượt gate caption đối với A, giữ factuality, và tốt hơn B. Nếu R≈B nhưng cả hai hơn A, chỉ có bằng chứng routing vào ảnh, chưa có bằng chứng vị trí. Nếu chỉ routing loss giảm mà caption không tăng, dừng cấu hình.

Trọng số, lịch, checkpoint và tiêu chí so sánh cuối chưa khóa; sẽ chốt sau manifest, trước run, không dò dev. Chưa xác minh novelty hoặc hứa gain.
