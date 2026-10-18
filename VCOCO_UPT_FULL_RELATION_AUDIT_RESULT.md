# UPT full native-proposal audit: raw verb score không đạt điều kiện chọn prompt

User-run GPU audit đã hoàn tất: 507 Calibration + 2.867 Development, hai shard
1.687 ảnh. Local replay **PASS**. Chưa train caption, chưa thay nguồn prompt YOLO,
không có CIDEr/CHAIR mới.

Nguồn `vcoco_upt_full_relation_audit_review.zip`, 4.143.589 bytes, SHA256
`181e47b9db1e1ec87217d10a70f191cb8bc0ff69a61d7947e9f47ef701f4291f`.
Code run `5b397038a24fb17264588c328da9e069ea821507`. Official UPT HICO weights giữ
hash đã pin. Không fit model, không sửa threshold theo Development.

## Định vị endpoint trên Development

| Quan hệ | Liên kết chuẩn | UPT định vị đủ | Tỷ lệ |
|---|---:|---:|---:|
| Holding | 1.608 | 1.313 | 81,65% |
| Riding | 223 | 204 | 91,48% |
| Tổng | 1.831 | 1.517 | 82,85% |

Trên cùng danh sách ảnh/link và matching policy, cache YOLO cap10 đạt 1.323/1.831
(72,26%), cap100 đạt 1.391/1.831 (75,97%). Đây là so **cấu hình**: UPT native
DETR/proposal threshold/min/max khác YOLO, không phải benchmark detector cùng
ngân sách và không chứng minh DETR luôn tốt hơn YOLO. UPT cũng có no-interaction
image behavior khác vì native min-instance retention. Không ảnh Development nào
không có native pairs, nhưng điều đó không chứng minh mọi ảnh đều có quan hệ thật.

Localization là điều kiện cần, không phải độ đúng của hành động. Các số IoU là số
GPU journal báo: ZIP không giữ bbox nên local audit không replay độc lập IoU.
Đã replay các ID/label/flag consistency và denominator với annotation chuẩn.

## Bộ lọc raw score không dùng được dưới policy đã khóa

Nhánh đang đánh giá dùng `sigmoid(raw verb logit)` cho các cặp UPT hỗ trợ theo HICO
ontology. Calibration tìm ngưỡng thấp nhất đạt >=90% precision trên known labels,
ít nhất 5 accepted known, cần cả dương và âm; nếu không có thì ngưỡng1.01 (abstain).

| Calibration | Positive/negative known có score | Ngưỡng hợp lệ | Quan hệ được nhận trên Development |
|---|---:|---|---:|
| Holding | 222 / 3.614 | Không có | 0 |
| Riding | 33 / 1.353 | Không có | 0 |

Precision của chính sách rỗng là **undefined/null**, không phải0% hay100%; recall0%.
Ngưỡng1.01 không phải xác suất mà model tạo ra: đó là mã abstain, cao hơn mọi
probability [0,1]. Không dùng threshold mới trên Development để cứu kết quả.

Phân tích failure trên Calibration, không tạo policy mới: precision cao nhất trong
các high-score tails có ít nhất5 accepts chỉ là holding96/162=59,26%; riding26/255
=10,20%. Ngay các score gần1 vẫn có nhiều nhãn action-absent. Không hạ mục tiêu90%
để công bố một bộ prompt mới. 131.618 candidate cells unknown trong toàn run được
giữ unknown, không tự biến thành false. 35.702 known cells được đối chiếu label gốc.

## Vì sao chưa kết luận UPT không tốt?

Đây là full native proposal pool, khác các diagnostic cũ chỉ trên GT-matched/candidate
subsets được tuyển trước. Raw action logits có thể cao ở các proposal mà chính
detector cho confidence thấp. Official UPT dùng thêm prior:

`native_score = sigmoid(verb_logit) * subject_detection_score**2.8 * object_detection_score**2.8`

trên những object–verb được model hỗ trợ. Công thức có trong pinned upstream
`interaction_head.compute_prior_scores` và `UPT.postprocessing`. Vì vậy kết quả
raw-only hiện tại chưa phải phép chấm native score. Đây là giới hạn của lựa chọn
score trong thí nghiệm của mình, không nên quy thành việc mọi mô hình HOI thất bại.

ZIP này chỉ giữ số lượng detections, candidate score và matched IDs; không giữ box,
object confidence hay native score nên không thể tính lại prior chỉ từ ZIP này.
Không bịa giá trị thiếu và không yêu cầu chạy lại 3.374 ảnh ngay: full cache notebook
đã xuất cả native score, raw logits, bbox và detection confidence để tái dùng.

## Quyết định tiếp theo

1. Giữ nguyên kết quả failed acceptance này; chưa dùng raw score làm prompt source.
2. Khi có full cache, replay native-score policy trên chính Calibration trước,
   cố định cách chọn trước Development; giữ denominator bỏ sót và unknown riêng.
3. Bước đó là follow-up sau khi đã xem failure, không gọi là xác nhận độc lập.
   Không chọn công thức/trọng số theo caption test hay tự chọn whitelist bằng Dev.
4. Chỉ khi nguồn quan hệ có precision/coverage đủ hữu ích mới thử caption adaptation.
   Full cache là dữ liệu tái dùng, không tự chứng minh chất lượng prompt/caption.

## Đã kiểm tra gì?

`audit_vcoco_upt_review.py` kiểm CRC/hash protocol/policy; hai shard đủ, đúng thứ tự;
protocol images khớp toàn bộ metadata/nhãn nguồn; mapped IDs đúng ảnh và một-một
nhất quán; known/unknown labels được tính lại; full positive denominators giữ nguyên;
fit threshold từ Calibration và replay metrics khớp cả hai nhóm. Không chạy inference,
không thay mô hình, không đo lại độ đúng IoU hoặc GPU checkpoint tensors từ ZIP.

Public aggregate: `audit/vcoco_upt_full_relation_audit_result.json`.
Local: `C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/upt_full_relation_audit_result`.

```powershell
python audit_vcoco_upt_review.py --archive "PATH/vcoco_upt_full_relation_audit_review.zip" --grounded-data "PATH/grounded_links_v1" --instances "PATH/instances_vcoco_trainval2014.json" --output "PATH/new_audit"
```
