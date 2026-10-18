# Cache UPT cho toàn bộ 123.287 ảnh COCO

**Đã chuẩn bị, chưa chạy suy luận GPU.** Người dùng hoặc Ngân tự chạy notebook
`UPT_COCO_FULL_CACHE.ipynb` trên tài khoản của mình. Không ảnh hưởng notebook
V-COCO audit đang chạy; không yêu cầu quyền truy cập tài khoản của người khác.

## Chạy lần đầu

1. Tạo notebook riêng, upload `UPT_COCO_FULL_CACHE.ipynb` hoặc dán launcher
   `kaggle_upt_coco_full_cache.py`.
2. Bật **Internet + GPU T4 x2**. Chỉ attach MS COCO 2014 hiện dùng.
3. Kiểm tra hai đường dẫn đầu cell:

```python
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
PREVIOUS_CACHE = None
MAX_HOURS = 8
```

Không cần checkpoint Q-Former, visual cache, prompt cache, YOLO cache hay nhãn V-COCO.
Bootstrap tải official UPT HICO weights khoảng 220 MB, kiểm hash và pin code
`e70347809a8e896562f312392a4d4c1df49c9b80`. Không tải lại ảnh COCO.

Hai worker chạy độc lập mỗi worker một GPU, không DDP. Log sau mỗi 10 ảnh mới,
runner báo hoạt động mỗi 15 giây. Khi hết ngân sách 8 giờ, worker lưu tiến độ rồi
dừng. Đây là ngân sách của một lượt, **không phải ETA toàn bộ 123.287 ảnh**.
Chưa đo tốc độ/dung lượng cache thực tế trên T4; có thể cần nhiều phiên.

## Toàn bộ ảnh và tách split

| Nhóm cache | Ảnh | Shard tối đa 500 ảnh |
|---|---:|---:|
| Train + restval | 113.287 | 227 |
| Validation | 5.000 | 10 |
| Test | 5.000 | 10 |
| Tổng | 123.287 | 247 |

Mỗi record vẫn ghi `source_split` train/restval/val/test, không trộn shard giữa các
nhóm. Dữ liệu tham chiếu caption không được đọc để tạo dự đoán: chỉ dùng ID,
filename, filepath và split từ metadata. Cache test không được dùng để chỉnh
ngưỡng hoặc chọn phương pháp. Không có metrics/test evaluation trong notebook này.

## Cache lưu gì?

- ID ảnh, filename, source split và SHA256 bytes ảnh.
- Kích thước ảnh gốc và kích thước sau resize của UPT.
- Mọi bbox, nhãn lớp và điểm đối tượng **mà UPT giữ lại**.
- Cặp chỉ số người–đối tượng theo native UPT proposal inventory.
- Raw FP32 logits và native interaction scores cho **mọi verb được HICO ontology
  hỗ trợ với lớp vật đích**, không chỉ holding/riding. Các số không lượng tử hóa FP16.
- Ảnh không có human-object pair vẫn giữ detections và danh sách cặp rỗng; không
  làm mất thông tin đối tượng hoặc loại ảnh khỏi cache.

`cache_identity.json` chứa object/verb vocabulary và `verbs_by_object`. Với cặp thứ
`i = pairs[i] = [s,o]`, cột `pair_logits[i]`/`native_scores[i]` lần lượt theo
`verbs_by_object[labels[o]]`. Verb không hỗ trợ được bỏ khỏi cột, **không phải nhãn âm**.
Sigmoid raw logit và native score là hai loại điểm khác nhau; không dùng lẫn.
Native score đã chứa prior detector của UPT; lưu riêng để tái lập nhánh cũ.

Bbox `boxes_resized_xyxy` là pixel xyxy trong `resized_size_hw`, không phải ảnh gốc.
Để lấy normalized xyxy, chia x cho width và y cho height của resized size;
để lấy pixel trên ảnh gốc, nhân normalized x/y với original width/height.
`original_size_wh` dùng thứ tự width,height; `resized_size_hw` dùng height,width.

“Trước lọc ngưỡng” nghĩa là chưa áp ngưỡng chọn prompt của mình. Detector/proposal
selection **nội bộ UPT vẫn có**: score threshold0.2, min3/max15 instances theo nhóm
người/vật trong implementation đã pin. Cache không chứa mọi DETR query hoặc mọi
đối tượng thật trong ảnh. Muốn thay preprocessing/proposal policy phải chạy cache
mới; không thể lấy lại bằng việc đổi ngưỡng prompt từ cache này.

Cache không chứa CLIP features, attention maps hay caption mới. Những thứ này
không cần cho yêu cầu lưu object–relation predictions hiện tại.

## Lưu output và tiếp tục ở phiên sau

Output chính **`/kaggle/working/upt_coco_full_cache/`** gồm các shard `.jsonl.gz`,
identity, manifest, summary, logs và tối đa hai journal đang dở. Repo/model đặt
trong `/tmp`, không làm Output tăng lên hàng nghìn file. Hoàn tất có khoảng 260
file chính, thấp hơn giới hạn 500 file mà người dùng gặp trước đây.

**Giữ toàn bộ thư mục cache**, không chỉ ZIP review. Save Version để lưu output
sau mỗi phiên. Sang phiên mới, attach output đã lưu hoặc dataset chứa cache,
đặt trực tiếp đường dẫn tới thư mục có `cache_identity.json`:

```python
PREVIOUS_CACHE = Path('/kaggle/input/.../upt_coco_full_cache')
```

Các shard hoàn tất được kiểm signature, thứ tự ID, gzip CRC và hash manifest nếu
có, rồi copy sang Output mới để kết quả mới tự chứa cả phần cũ lẫn phần mới.
Ngân sách giờ có thể đổi; identity model/data/schema/runtime phải giữ nguyên.
Copy cache trước đó cần dung lượng đĩa ngoài phần mới được tạo.

Journal đang dở được lưu từng ảnh. Nếu process bị ngắt ở dòng cuối, chỉ dòng ghi
dở đó được bỏ; không chạy lại cả 500 ảnh. Gzip shard chỉ xuất hiện khi ghi xong
và được đọc lại đầy đủ. Trường hợp session bị kill trước khi ghi manifest, vẫn
có thể phục hồi shard hoàn tất từ header signature và đúng thứ tự ID. Cần Save
Version/giữ output; dữ liệu của một session không được lưu thì không thể phục hồi.

Không đổi nguồn hay code giữa phiên rồi nối vào cùng cache. Nếu identity khác,
script từ chối; không tự chấp nhận số liệu từ hai model/runtime khác nhau.

## Biết đã xong bằng cách nào?

`run_summary.json` phải ghi `CACHE_COMPLETE`, 247/247 shards, và số ảnh hoàn tất:
train113287/val5000/test5000. `CACHE_PARTIAL` nghĩa là còn phải chạy tiếp, không
phải cache đủ. Các ảnh journal đang dở được báo riêng khỏi ảnh shard hoàn tất.

Gửi trợ lý **`upt_coco_full_cache_review.zip`** để kiểm tra tiến độ/hashes. ZIP này
chỉ có metadata/logs, **không chứa các shard cache**. Không tải/gửi lại weights.
Giữ hoặc attach cả thư mục cache cho notebook tạo prompt sau này.

## Kiểm tra đã làm

- 8 tests mới PASS: split separation, bỏ caption text, sparse supported logits,
  ảnh không cặp/không detection, invalid prediction, incomplete gzip, phục hồi
  journal và cache qua phiên kể cả thiếu manifest sau kill; không sửa Input.
- Cộng 30 regression tests inventory/box/coverage/relation audit: **38 PASS**.
- Đọc lại 5 prediction cache UPT thật đã có, kiểm hash; giữ chính xác 891 logits
  được hỗ trợ qua đóng gói và nén/đọc lại. Không chạy model inference local.
- Lập kế hoạch từ dataset thật: đúng 123.287 ảnh, 247 shard; chưa chạy GPU và chưa
  xác nhận tốc độ, tổng dung lượng hoặc end-to-end runtime trên máy Ngân.

Cache là kết quả của công cụ pretrained, không phải ground truth và chưa chứng
minh cải thiện prompt/caption. Chất lượng nguồn quan hệ do notebook audit đang
chạy kiểm tra riêng. Không cần dừng lượt audit đó để chạy cache trên notebook khác.
