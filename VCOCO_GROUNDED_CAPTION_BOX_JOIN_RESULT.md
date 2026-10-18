# Ghép nhãn quan hệ với box: giữ nhiều liên kết trong một ảnh

Ngày 18/10/2026. Đã hoàn thành chuẩn bị dữ liệu trên CPU; không huấn luyện,
không chạy Kaggle, không có kết quả caption mới.

## Đã làm gì

`prepare_vcoco_grounded_caption_data.py` ghép ID người và ID đối tượng trong
V-COCO với annotation box COCO chính thức, trên đúng các nhóm ID đã khóa ở
`audit/vcoco_grounded_caption_week1_groups.json`. Giữ **tất cả** liên kết holding
và riding có endpoint hợp lệ, không giới hạn một cặp hoặc ba cặp mỗi ảnh.

| Nhóm | Ảnh | Liên kết có box | Ảnh có ít nhất một liên kết | Ảnh có ít nhất hai liên kết |
|---|---:|---:|---:|---:|
| Fit | 2.026 | 1.330 | 1.014 | 208 |
| Calibration | 507 | 308 | 238 | 51 |
| Development | 2.867 | 1.831 | 1.415 | 281 |

Tổng 3.469 liên kết hợp lệ trên 5.400 ảnh; giữ cả ảnh không có liên kết dương của
hai quan hệ đang xét, không diễn giải là ảnh không có quan hệ hoặc không có đối tượng.
Liên kết fit gồm holding 1.141 trên 981 ảnh, riding 189 trên 91 ảnh.

Không mất liên kết có ID đích vì missing annotation/crowd/box lỗi trong lần ghép này.
Tuy nhiên có **1.020 nhãn hành động dương thiếu ID đích**: holding 977, riding 43,
chia fit 332/25, calibration 90/5, development 555/13. Chúng vẫn nằm trong danh sách
nhãn hành động của người, nhưng không tự tạo box đích hay liên kết để điền vào.
Một annotation hợp lệ hình học vẫn có thể sai ngữ nghĩa; kiểm tra box không thay
thế việc đánh giá chất lượng nhãn và không bảo đảm mọi action label là đúng.

## Nhiều người và caption nhóm

Fit có 158 ảnh, calibration 37 ảnh và development 215 ảnh có nhiều chủ thể khác
nhau cùng thực hiện một predicate với đối tượng cùng lớp trong annotation. Các số
theo predicate/object class không cộng trực tiếp thành tổng ảnh vì có thể giao nhau.

Ví dụ COCO 45089 có hai liên kết được ghi rõ:

- Người ID 204714 → riding → ngựa ID 57887.
- Người ID 2151556 → riding → ngựa ID 60387.

Giữ cả hai liên kết. Việc trong ảnh còn người/ngựa khác không có đầy đủ role link
không biến chúng thành âm. Số liên kết không phải số người thật trong ảnh. Không
tự sinh câu “two people” hoặc “a group” rồi dùng làm caption ground truth từ số này.
Decoder cần giữ đường đặc trưng toàn ảnh cùng nhiều bằng chứng cục bộ để có thể
mô tả hoạt động nhóm; chưa có thí nghiệm chứng minh cơ chế đó ở bước chuẩn bị này.

## Dữ liệu xuất ra

Mỗi ảnh có kích thước, danh sách instance với ID/lớp/box normalized xyxy, danh sách
quan hệ dương và nhãn hành động riêng theo người. Unknown giữ là -1. Box lấy từ
annotation GT, **không phải box detector**. Những instance được xuất chỉ là actor
được annotate và các endpoint đã liên kết, không phải toàn bộ proposal trong ảnh.
Chúng dùng cho giám sát; không được dùng GT role để chọn candidate khi inference.

Local: `C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/grounded_links_v1/`.
Ba file `relation_fit_images.json`, `relation_calibration_images.json`,
`relation_dev_images.json`; chỉ aggregate/hashes được đưa vào Git.

## Nguồn và tái lập

`fetch_vcoco_coco_instances.py` lấy đúng member `annotations/instances_train2014.json`
trong ZIP COCO chính thức bằng HTTP Range, kiểm CRC khi đọc hết, ghi ETag và SHA256.
Sau đó lọc metadata cho 5.400 ID V-COCO; kết quả gồm 47.423 object annotations,
khoảng 4,98 MB. Không tải ảnh, model hoặc captions COCO từ ZIP này. Caption metadata
Karpathy và nhãn V-COCO là nguồn đã kiểm hash ở bước trước.

```powershell
python -m pip install ijson==3.4.0
python fetch_vcoco_coco_instances.py --sources "PATH/sources" --output "PATH/sources/instances_vcoco_trainval2014.json"
python prepare_vcoco_grounded_caption_data.py --sources "PATH/sources" --instances-json "PATH/sources/instances_vcoco_trainval2014.json" --groups "audit/vcoco_grounded_caption_week1_groups.json" --output "PATH/new_grounded_links"
python -m unittest tests.test_vcoco_alignment_data tests.test_vcoco_grounded_caption_data -v
```

16 tests PASS: giữ nhiều actor/predicate, không tạo link từ role thiếu, bỏ endpoint
lỗi nhưng giữ action actor còn dùng được, kiểm split/identity, unknown và nguồn.
Script không ghi đè output cũ. Đã đối chiếu SHA256 và toàn bộ 3.469 links với nhãn
gốc; chạy lại nguồn thật cho ba file export giống byte và summary trùng khớp.

Aggregate: `audit/vcoco_grounded_caption_box_join_result.json`.
Provenance: `audit/vcoco_grounded_caption_coco_subset_provenance.json`.

## Việc còn lại

Đánh giá khả năng định vị đúng endpoint bằng detector và nguồn dự đoán quan hệ,
không đánh đồng số GT links với recall triển khai. Chốt loss tác động decoder trước
khi viết notebook train; các thất bại reader/attention/memory cũ vẫn là đối chứng.
Không có tuyên bố cải thiện CIDEr/CHAIR hoặc tính mới từ bước ghép annotation này.
