# K/V CPU preparation — dữ liệu sẵn sàng cho model preflight

Ngày kiểm tra: 30/10/2026. User tự chạy notebook core `bfc54da3078ce241282643a4018698942f67f253`; root không chạy mô hình/Kaggle. [Protocol khóa trước](RELATION_KV_CPU_PREPARATION_PROTOCOL.md), [audit JSON](KV_CPU_PREPARATION_RESULT.json), [audit code](audit_kv_preparation.py).

## Kết quả đã xác minh

Receipt báo `READY_FOR_MODEL_PREFLIGHT`, hoàn tất sau **3689,47 giây (~61,5 phút)** ở bước CPU. 5472 ảnh có cache đã stage: 4000 caption TRAIN, 1000 caption dev và 472 ảnh auxiliary bổ sung. Thời gian này là CPU preparation, không phải GPU training hoặc ETA D/E.

| Nhóm | Fact độc lập sau dedup | Ảnh | Touching đúng/sai | Containment đúng/sai | Nhóm query có observed true + false |
|---|---:|---:|---:|---:|---:|
| Fit | 401 | 393 | 138 / 146 | 63 / 54 | 58 |
| Held | 99 | 98 | 39 / 31 | 16 / 13 | 16 |

Containment gồm `contains↔inside`; `touching` đối xứng. Một fact gồm hai equivalent views vẫn chỉ là một event, không phải hai nhãn độc lập. Tổng500facts này là nguồn auxiliary nhỏ, không phải500caption images độc lập hoặc một benchmark quan hệ đầy đủ.

673 observed assertions đủ policy nguồn; 534 ghép được với hai predicted boxes phân biệt, 139 không ghép được bị skip (unknown, không phải negative). Một orbit chứa hai annotation gán GT entities khác nhau bị quarantine cả2observations. 532observations còn lại gộp thành500facts. Không sửa gate để qua kiểm tra.

Fit/held không trùng image. Held không trùng4000captionTRAIN hay1000captiondev; 19fitimages nằm trong captionTRAIN như policy đã chốt. Auxiliary chỉ lấy officialVSRTRAIN, loại mọi image xuất hiện trong officialVSRdev/test và128image held cũ. Held là giữ khỏi adaptation mới, không khẳng định ảnh chưa từng được C0 nhìn thấy.

## Kiểm tra và giới hạn

- ZIP chỉ đọc một lần, text cache ignored tại `review/kv_preparation_loaded.json`; ZIP SHA256 `5c133973195263ba2e34514379875a79597f1bdfcac03f7cf44d74d0961482fa`.
- Audit metadata **69990assertions PASS**: sáu review payload SHA, hai executed source files khớp raw Git, nhãn/validator/source-row chính thức, inverse views/dedup, split và MP groups, bank bounds/class-free fields, đúng lịch native4000TRAIN/1000dev/vocabulary. Đây là số assertions, không phải số thực nghiệm hay neural tests.
- Log user-run ghi16metadata/NumPy/mock tests PASS; copy/readback5472/5472rows hoàn tất; ba rawvisual shards có metadata tương thích. Không tạo encoder/captioner; C0 checkpoint chỉ hash.
- Review ZIP không chứa `inputs.h5`. H5 SHA do user-run receipt báo là `960e1995c6e7695597d95f9535f536576a5c74a03c6c26386da4f7c5b2a4da89`; root chưa kiểm binary H5 độc lập, chưa GT-IoU rematch local, chưa CUDA/model/gradient/parity/preflight. Notebook D/E cần verify payload hash khi mount dataset.
- Nguồn có consensus và literal ontology, chưa có đánh giá thủ công mới trên500facts; không coi gate/count là chứng minh semantic correctness. Chỉ một cặp inverse có hướng và một quan hệ đối xứng nên không nhận khả năng bao phủ nhiều quan hệ/direction hoặc novelty/hiệu quả caption.

## Quyết định

**GO sang chuẩn bị model preflight D/E trong phạm vi pilot đã duyệt**, không GO full training hoặc claim alignment thành công. Giữ nguyên thư mục `/kaggle/working/relation_kv_preparation_v1` thành Dataset để tránh stage lại. Root đang lấy version/path từ user và đã gửi Astra số liệu đã kiểm để khóa shared objective/schedule/checks; không yêu cầu đọc ZIP lần nữa. D/E chỉ khác value projection; không thêm nhánh hoặc loss để cứu kết quả. Quota gần nhất20giờ/tài khoản, chưa đo tốc độ model.


## Lưu output115

User xác nhận version115 của `ducanh2403/2-vi-cai-kia-qua-nhieu`. Notebook `SAVE_RELATION_KV_V115_DATASET.ipynb` tải đúng lượt này, khóa receipt SHA `bcfeb4f263f3860374e3d255f66b0ed8912816413706f3444b3091aa4d8af1a8` và cả7payloads, sao chép8file gốc nguyên byte rồi tạo Dataset Private mới qua token workflow đã dùng. Không cần Add Input; Accelerator=None, Internet ON; Secret mặc định KAGGLE_API_TOKEN (đổi SECRET_NAME nếu tên codex). Path dataset mặc định dự kiến `/kaggle/input/datasets/ducanh2403/relation-kv-preparation-v115`; user gửi mount thực tế sau khi lưu/chia sẻ với Ngân. Chưa gọi Kaggle API hoặc upload từ root.4synthetic/static transport checks PASS; chưa xác nhận Dataset tồn tại hoặc upload thành công.
