# Kết quả UPT official score: holding có tín hiệu, riding lệch định nghĩa nhãn

Run eb00d6c đã hoàn tất 507 Calibration + 2.867 Development. ZIP SHA256
`d8a96db3c093cce6eb8e49be5d0e9b5c5a7f697a5a8c54b0092173d816583ff5`,
8.289.151 bytes. Local audit PASS: đủ shard, đúng nguồn/ID/nhãn, CRC/hash đúng;
replay bbox matching IoU, positive denominators và công thức native score khớp.
Không chạy model GPU lại.

## Kết quả với policy đã định trước

Giữ nguyên policy >=90% known precision, >=5 known accepts trên Calibration,
chọn ngưỡng thấp nhất đạt điều kiện; nếu không đạt thì abstain. Không sửa ngưỡng
theo Development.

| Nhóm / holding | Đúng | Sai known | Unknown được nhận | Known precision | Full positive recall |
|---|---:|---:|---:|---:|---:|
| Calibration | 90 | 10 | 40 | 90,00% | 90/275 = 32,73% |
| Development | 559 | 89 | 170 | 86,27% | 559/1.608 = 34,76% |

Ngưỡng holding 0.503703236579895. Trên Development có 818 claims được nhận thuộc
728 ảnh; 559 đúng + 89 sai known + 170 chưa rõ. Không gọi 86,27% là precision trên
toàn bộ 818 claims: unknown bị loại khỏi mẫu số precision. Mục tiêu 90% trên
Calibration không bảo đảm 90% ở Development; kết quả này chưa đạt mục tiêu đó
ngoài tập chọn ngưỡng.

Riding vẫn ngưỡng sentinel 1.01 (abstain), accepted = 0, precision undefined,
recall 0. UPT định vị đủ 204/223 riding và 1.313/1.608 holding ở Development;
tổng 1.517/1.831 = 82,85%. Không nhầm localization với relation precision.

So raw-score run trước: cả hai predicate đều abstain. Official score cải thiện
khả năng chọn holding. Chưa có kết quả caption/CIDEr/CHAIR mới.

## Riding: cần đối chiếu lại ontology trước khi kết luận model nhận sai

Phân tích Calibration, KHÔNG tạo policy mới: score tail riding có precision cao
nhất trong các tail >=5 known accepts là threshold 0.4716057777404785, 23 TP/118
known = 19,49%, gồm 95 known false. Đây không phải ngưỡng đã được chấp nhận.

Toàn bộ 95 known false nhắm predicted objects là surfboard (41), skateboard (24),
snowboard (17), skis (13). Đối chiếu chính actor đó với nhãn gốc V-COCO train/val:
41 có action surf, 24 skateboard, 16 snowboard, 13 ski, 1 không có bốn action này.
Tức 94/95 có nhãn thể thao được V-COCO tách khỏi ride. Source hashes được kiểm.

Điều đó cho thấy dùng cùng chữ ride để chuyển trực tiếp giữa hai ontology đang
phạt cả những hoạt động có định nghĩa khác. Không đủ chứng minh 94 claims đều đúng
về instance/target; nhãn actor hành động không tự xác nhận target được UPT đề xuất.
Không sửa 95 nhãn thành positive, không loại các lớp vật này sau khi xem Dev để
gọi là cải thiện độc lập, và không kết luận mọi ride false đều do ontology.

## Quyết định tiếp theo

Chưa đưa nguồn quan hệ này vào full caption training hoặc chuyển ngay sang test.
Trước hết lập bảng đối chiếu ngữ nghĩa HICO/V-COCO theo action và object từ tài
liệu/annotation, xử lý unsupported hoặc không tương đương rõ ràng, rồi mới khóa
protocol đánh giá tiếp theo. Không chọn mapping theo score hay caption test.
Bbox/confidence/raw/native claims đã có: các phép đối chiếu giữ nguyên inventory
hold/ride có thể tính lại local, không cần chạy UPT lại. Muốn đo các verb khác phải
có full cache tương ứng. Đây là follow-up phát triển phương pháp, không phải
confirmation độc lập trên dữ liệu chưa xem.

12 CPU tests PASS gồm 4 test mới: replay bbox/prior đúng, tráo raw score bị từ chối,
sửa box bị từ chối, thiếu raw claim bị từ chối. Thực tế replay đủ 3.374 ảnh PASS,
35.702 known + 131.618 unknown candidate labels. Chưa tái chạy weights hoặc xác minh
thủ công từng ảnh; annotation chưa bao phủ mọi target, pretraining overlap chưa rõ.

Aggregate: `audit/vcoco_upt_native_score_audit_result.json`.

```powershell
python audit_vcoco_upt_review.py --archive "PATH/vcoco_upt_native_score_audit_review.zip" --grounded-data "PATH/grounded_links_v1" --instances "PATH/instances_vcoco_trainval2014.json" --vcoco-sources "PATH/sources" --output "PATH/new_audit"
```
