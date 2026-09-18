# Chỉ đối tượng: H1.2 + Gate + visual cache

Import `objects_only_kaggle.ipynb` vào notebook Kaggle mới. Giữ nguyên 3 visual
cache .h5, MSCOCO ảnh và Karpathy split. Gắn thêm kết quả YOLO gốc chứa danh sách
đối tượng. Chưa chạy thực nghiệm GPU trong môi trường phát triển này.

## Input đối tượng

Source là JSON ánh xạ đường dẫn/tên ảnh tới danh sách lớp hoặc mục có `objects`:

Notebook đã đặt sẵn source bạn cung cấp:
`/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json`,
`OBJECTS_FIELD='objects'`, `NAME_KEY='label'`. Với detection mẫu có nhiều bowl và
orange, text tạo ra là `bowl, broccoli, orange`.

```json
{
  "COCO_val2014_000000391895.jpg": {
    "objects": ["person", "bicycle"],
    "prompt": "person next to bicycle"
  }
}
```

Danh sách detection dictionaries cũng được hỗ trợ:

```json
{"COCO_val2014_000000391895.jpg": {"detections": [{"class_name": "person", "confidence": 0.9}]}}
```

Ví dụ thứ hai cần `OBJECTS_FIELD='detections'`. Class name mặc định lấy từ
`name`/`label`/`class_name`; nếu dùng field khác đặt NAME_KEY. Không đoán tên lớp
từ class ID số, không lấy từ caption, không cố bỏ quan hệ khỏi embedding.
JSON chỉ có trường `prompt` chưa đủ: cần danh sách đối tượng gốc hoặc code tạo prompt
cũ để khôi phục chính xác cách chọn đối tượng. Cache .pt có raw objects cũng dùng
được qua CLI, nhưng notebook mặc định kiểm tra schema JSON.

Đặt SOURCE đúng đường dẫn Kaggle trước khi chạy. Không chạy YOLO hoặc VLM lại.
Giữ thứ tự source, loại tên lớp trùng, chuyển lowercase, nối bằng dấu phẩy.
Không lọc confidence thêm; danh sách phải đã được chọn đúng như thí nghiệm cũ.
Ảnh không detection giữ text rỗng (CLIP BOS/EOS), metadata báo số trường hợp đó.
Kiểm tra text mẫu và số prompt vượt 20 tokens trước khi đánh giá kết quả.

## Train và test

Notebook tạo `prompt_yolo_objects.pt` mới bằng CLIP text last_hidden_state,
FP32 extraction/FP16 storage, shape (20,512), mask giữ padding. Sau đó verify visual
cache, train từ đầu 10 epoch, seed 42, batch 32, lr 1e-4, workers 0 và test đủ 5000
ảnh với beam 5. Output experiment riêng: `/kaggle/working/h1_2_gated_yolo_objects_only`.
Lưu cache mới để các lần train sau không cần encode lại. File cache đã tồn tại
sẽ không bị ghi đè; lần chạy lại train trỏ --prompt-cache-path tới file có sẵn.

So sánh với Gate có prompt quan hệ và cache hiện tại (CIDEr 1.182, khoảng 5h train).
Chưa cam kết metric hoặc thời gian cho bản chỉ đối tượng. Cần kiểm tra names,
thứ tự, lựa chọn đối tượng, template và encoding metadata của source cũ: nếu khác,
kết quả là so sánh nội dung prompt, chưa cô lập tuyệt đối chỉ riêng quan hệ.
Không dùng test để chọn hyperparameter; khác biệt nhỏ cần nhiều seed.

## Gate hiện tại

Gọi P là prompt features sau projection, V là visual features sau projection.

```
A = CrossAttention(query=P, key=V, value=V)
g = sigmoid(Linear(concat(P, A)))
P_grounded = LayerNorm(P + Dropout(g * A))
memory = concat(P_grounded, V)
```

Gate có shape (batch,20,512): mỗi token và mỗi channel có một giá trị giữa 0 và 1.
Nó điều chỉnh phần visual update A, không loại bỏ prompt gốc P. Weight của linear
khởi tạo 0, bias -2, g ban đầu xấp xỉ 0.119. Projection, attention, gate và decoder
được học bằng caption cross-entropy; CLIP backbone cố định. g không phải xác suất
prompt đúng hoặc điểm tin cậy của detector. Chưa có loss căn chỉnh riêng.
