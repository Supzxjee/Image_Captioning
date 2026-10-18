# Đánh giá nguồn quan hệ UPT trên Calibration và Development đầy đủ

Trạng thái: **đã chuẩn bị code/notebook, chưa chạy suy luận GPU**. Người dùng tự chạy
Kaggle. Đây là bước kiểm tra chất lượng nguồn quan hệ, không phải mô hình caption
mới hoặc đóng góp đã chứng minh. Không yêu cầu chấm thêm thủ công ở bước này.

## Mục đích

Mô hình tương tác pretrained UPT-R50 HICO-DET nhận ảnh và tự tạo người/vật/cặp ứng
viên; không dùng annotation role để chọn cặp đầu vào. Chấm hai quan hệ holding,
riding với các nhãn V-COCO đã ghép box. Dùng toàn bộ 507 ảnh Calibration và 2.867
ảnh Development đã khóa; không lọc còn những ảnh/cặp mà model đoán đúng.

**UPT dùng detector DETR riêng.** Số 72,21% của audit YOLO trước là độ bao phủ của
cache YOLO, không phải của UPT và không được mang sang làm mẫu số có điều kiện.
Notebook đo lại localization, khả năng có điểm quan hệ và quyết định nhận quan hệ.

## Đánh giá thế nào?

1. Frozen UPT suy luận từ ảnh, giữ native proposals và sigmoid verb logits cho các
   tổ hợp object–verb mà checkpoint HICO hỗ trợ. Không fine-tune, không thêm lớp mới.
2. Sau suy luận mới match với toàn bộ GT hợp lệ trong ảnh, cùng lớp và IoU>=0.5,
   một-một greedy như audit YOLO. Không dùng role label để tạo proposal.
3. Cặp match đúng người/vật và có nhãn dương là positive. Nếu người có nhãn action
   vắng mặt thì claim tương ứng là known negative. Cặp sai target hoặc actor không
   có nhãn vẫn UNKNOWN, không mặc nhiên coi mọi proposal không match là false.
4. Calibration chọn riêng ngưỡng holding/riding: ngưỡng thấp nhất đạt >=90% precision
   trên nhãn known, ít nhất 5 accepted known và phải có cả outcome dương/âm. Không có
   ngưỡng đạt thì abstain toàn bộ predicate đó. Không fit temperature hoặc quét seed.
5. Ghi `locked_calibration_policy.json` trước khi tính báo cáo Development. Không
   thay ngưỡng sau khi xem Development; inference các ảnh không dùng label.

Báo `accepted_true`, `accepted_false_known`, `accepted_unknown`, precision trên
known, số ảnh có claim được giữ, và recall với mẫu số **mọi liên kết dương chuẩn**.
Thiếu detection/unsupported ontology/missing pair không bị xóa khỏi mẫu số recall.
Precision known không phải precision trên mọi proposal: phần unknown được báo riêng.
Đây chưa phải official V-COCO role AP và không phải bài kiểm tra hoàn chỉnh sai-target.

Development đã được xem trong nghiên cứu trước, không gọi là benchmark mới độc lập.
HICO/COCO pretraining có thể chứa ảnh liên quan; chưa kiểm tra đầy đủ overlap web.

## Chạy trên Kaggle

Tạo notebook mới hoặc thay nội dung notebook bạn muốn dùng bằng
`VCOCO_UPT_FULL_RELATION_AUDIT.ipynb` (hoặc dán `kaggle_vcoco_upt_full_audit.py`).
Chọn **Internet ON + GPU T4 x2**. Assistant không submit hoặc chạy notebook thay bạn.

Chỉ cần Input MS COCO hiện có:

```python
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
```

Không cần checkpoint Q-Former, prompt cache, visual cache hoặc cache YOLO.
Đường dẫn ghi tường minh ở đầu cell, không quét Input. Hai worker dùng hai GPU riêng,
không DDP. Mỗi worker log sau 10 ảnh, runner báo hoạt động mỗi 15 giây.

Bootstrap tự lấy nhãn V-COCO chính thức và đúng member annotation COCO (~102 MiB
nén), dựng lại đúng dữ liệu đã kiểm chứng; không tải lại ảnh COCO. Sau đó tải official
UPT HICO checkpoint khoảng 220 MB nếu chưa có. Có thể điền `LOCAL_UPT_CHECKPOINT`
để dùng file model đã attach; cần đúng hash, không dùng checkpoint Q-Former vào đây.

Code inference pin commit `5396125bcefe0c091944fcd15a5e65e4b5d47a83`, upstream và
checkpoint giữ nguyên identity từ các kiểm tra UPT trước. Bootstrap kiểm hash sources,
split, box join và chạy 30 tests trước suy luận. Chưa đo thời gian cả notebook trên T4,
không hứa ETA hoặc kết quả. GPU/torchvision/upstream compatibility cần xác nhận lúc chạy.

## Khi xong gửi gì?

Gửi **`vcoco_upt_full_relation_audit_review.zip`**. Không cần gửi weights. ZIP gồm
protocol, decisions/labels theo native proposal, ngưỡng đã khóa, logs và summary.
Nếu runner lỗi, ZIP kèm `failure.json`; lỗi trước khi bắt đầu runner có thể chưa có ZIP.
Không gọi failure/incomplete là kết quả thí nghiệm. Dừng/killed session cũng có thể
chưa tạo ZIP. Runner dùng output mới, không ghi đè hoặc tự resume một run bị thay cấu hình.

## Kiểm tra chuẩn bị

30 CPU tests PASS cho inventory, box join, detector coverage và evaluator mới:
unknown target không thành âm; missing detection còn trong denominator; GT khác
cũng cạnh tranh mapping; threshold không tách score ties; không đủ support phải
abstain; unknown accepted được đếm riêng, không nâng precision giả. Syntax PASS.
Đây là test evaluator và protocol, chưa phải GPU inference trên 3.374 ảnh.

Khi nhận ZIP mới quyết định nguồn relation có đủ precision/coverage để thử tác động
decoder hay không. Nếu không đủ, không đưa prompt nhiễu vào full caption training
và không chọn whitelist sau khi xem kết quả rồi gọi đó là cải thiện độc lập.
