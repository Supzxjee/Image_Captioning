# Detector có tìm được đúng hai endpoint của quan hệ không?

Ngày 18/10/2026. Đã kiểm tra cache YOLO cũ trên toàn bộ 5.400 ảnh V-COCO train/val
đã khóa. **Không chạy lại detector, không huấn luyện, không chạy notebook Kaggle.**
Đây là phép đo định vị, chưa phải độ chính xác quan hệ hay kết quả caption.

## Điều kiện kiểm tra cố định

- Cache giống input SCST/caption trước đây, SHA256
  `498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c`.
- Confidence >=0.5; same-class NMS IoU0.8; pixel xyxy đổi sang normalized xyxy.
- Hai giới hạn được khai báo trước khi tính: cap10 là ngân sách tham chiếu triển khai
  cũ; cap100 chỉ kiểm tra tác động cắt số box. Không chọn cap bằng caption score.
- Candidate gồm mọi người được giữ -> mọi instance khác được giữ, kể cả người khác.
  Không đọc GT role, predicate hoặc lớp vật “có thể cưỡi/cầm” để lọc candidate.
- Chỉ sau khi tạo candidate mới match với **toàn bộ** box GT hợp lệ, noncrowd
  trong ảnh, không chỉ box endpoint đang được quan tâm. Matching cùng lớp,
  IoU>=0.5, một-một greedy theo IoU giảm dần, confidence dùng phá hòa.
- Một liên kết chỉ được coi là định vị đủ khi cả subject và object match hai detection
  riêng. GT mapping là output audit, không được dùng làm bộ chọn cặp lúc inference.
- Mọi liên kết GT đều ở mẫu số, kể cả bỏ sót endpoint, ảnh không có candidate hoặc
  thiếu cache. Cache thiếu khác với cache có danh sách detection rỗng.

## Kết quả chính: cap10

| Nhóm | Liên kết chuẩn | Định vị đủ người và vật | Tỷ lệ |
|---|---:|---:|---:|
| Fit | 1.330 | 957 | 71,95% |
| Calibration | 308 | 225 | 73,05% |
| Development | 1.831 | 1.323 | 72,26% |
| Tổng mô tả dữ liệu | 3.469 | 2.505 | 72,21% |

Tổng là thống kê localization trên các nhóm, không gộp để huấn luyện hoặc chỉnh ngưỡng.
Development holding: 1.155/1.608 (71,83%); riding: 168/223 (75,34%).
Trong 508 liên kết development mất định vị: 444 thiếu vật đích nhưng có người,
45 thiếu người nhưng có vật, 19 thiếu cả hai. Các số tính theo liên kết nên có thể
cùng một vật/người bị tính ở nhiều liên kết; không phải 508 ảnh độc lập.

Không thiếu bản ghi cache trong 5.400 ảnh. Sau lọc, fit có 1 ảnh và development
có 1 ảnh không còn detection; calibration không có. Số ảnh không sinh được cặp
lần lượt 80,27,114. Ảnh không có cặp không được gọi là ảnh thật sự không có vật.

## Ảnh hưởng của giới hạn box

| Nhóm | Cap10 định vị đủ | Cap100 định vị đủ |
|---|---:|---:|
| Fit | 957/1.330 (71,95%) | 1.003/1.330 (75,41%) |
| Calibration | 225/308 (73,05%) | 237/308 (76,95%) |
| Development | 1.323/1.831 (72,26%) | 1.391/1.831 (75,97%) |

Cap100 vẫn chưa bao phủ hết, nên chỉ tăng số box không giải quyết toàn bộ định vị.
Không suy nguyên nhân tất cả phần còn lại là detector không phát hiện: threshold,
class mismatch, NMS, annotation và thuật toán matching cũng có thể ảnh hưởng.
Tăng cap làm tổng candidate từ 89.770 lên 136.702; chưa có số đo chi phí model
hoặc caption để chọn một phương án. Cả hai cap dùng cùng confidence và NMS.

## Không dùng “không có liên kết” làm nhãn âm

Với cap10, 39.767 candidate xuất hiện trên ảnh không có liên kết dương holding/riding
đủ role trong nguồn. Không gán tất cả chúng âm: có thể action vắng mặt đã annotate,
action chưa annotate, action dương thiếu role, hoặc hoạt động thuộc lớp khác.
Những trường hợp này phải xét nhãn actor/action cụ thể khi tạo supervision.

Định vị được 72% liên kết không có nghĩa 72% quan hệ dự đoán đúng. Chưa có bộ
quan hệ nào được chấm trong lần này. Matching GT không đo false relation trên
mọi detector proposal và không phải standard V-COCO role AP.

## Kiểm tra và artifact

23 tests PASS cho inventory, ghép box và coverage: missing-cache/empty tách biệt,
không match sai lớp, GT không liên kết cũng cạnh tranh detection, cap/NMS không
dùng role label, không tạo nhãn âm từ ảnh không có focus link, nhiều predicate giữ
riêng. Kiểm lại hash cả sáu report, số link, candidate và mapping đầu-cuối.

Code: `audit_vcoco_detector_coverage.py`.
Aggregate: `audit/vcoco_grounded_caption_detector_coverage.json`.
Local output: `C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/detector_coverage_v1`.
Reports giữ đầy đủ candidate detections và trạng thái từng link; không chỉ export
những link match được. Không đưa per-image detections đầy đủ vào public repo.

```powershell
python audit_vcoco_detector_coverage.py --grounded-data "PATH/grounded_links_v1" --instances-json "PATH/instances_vcoco_trainval2014.json" --detections "PATH/objectdetectioncache.json" --output "PATH/new_detector_coverage"
python -m unittest tests.test_vcoco_detector_coverage tests.test_vcoco_grounded_caption_data tests.test_vcoco_alignment_data -v
```

## Bước tiếp theo

Đã có kiểm tra endpoint cho toàn bộ nguồn, nên không chạy lại cùng diagnostic
GT-matched nhỏ. Cần đo nguồn điểm quan hệ trên candidate thực, giữ unknown và
unavailable; ưu tiên tái dùng HOI pretrained đã có trước khi huấn luyện lại MLP
từng thất bại. Phần decoder alignment phải có mục tiêu và control rõ ràng; chuẩn bị
notebook để người dùng tự chạy sau khi chốt protocol. Chưa có cơ sở hứa tăng CIDEr.
