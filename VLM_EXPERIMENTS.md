# Thực nghiệm tiếp theo: VLM thay YOLO và spatial heuristic

Mốc hiện tại: H1.2 + Gate + visual cache, train loss 1.915,
BLEU-1 0.766, BLEU-4 0.366, METEOR 0.283, ROUGE-L 0.571, CIDEr 1.182,
thời gian train khoảng 5 giờ. Những kết quả dưới đây chưa được chạy trên GPU.

## Thiết kế

| Thực nghiệm | Prompt | Mô hình caption |
|---|---|---|
| Mốc hiện tại | YOLO + spatial heuristic | Gate + visual cache |
| VLM objects | Đối tượng + thuộc tính nhìn thấy | Giữ nguyên |
| VLM spatial | Cùng đối tượng + thuộc tính, thêm quan hệ không gian | Giữ nguyên |

Một lần VLM đọc ảnh tạo JSON dùng chung cho hai loại prompt. Không đưa caption
tham chiếu vào VLM. Dùng Qwen/Qwen2.5-VL-3B-Instruct, greedy decoding, FP16,
giới hạn độ phân giải 512 visual tokens theo cấu hình processor. Đây là lựa chọn
để thử trên GPU Kaggle, không phải cam kết tốc độ hoặc khả năng vừa VRAM cho mọi ảnh.
Model card và API:
https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

CLIP text_model.last_hidden_state tạo prompt tokens (20, 512), trích xuất FP32,
lưu FP16. Giữ nguyên giới hạn prompt 20 tokens, visual cache, kiến trúc, seed 42,
batch size 32, lr 1e-4, 10 epoch, beam 5 và Karpathy split. Train từ đầu,
không warm-start checkpoint Gate cũ. VLM không chạy trong từng epoch.

Quan hệ ở đầu prompt để tránh bị cắt hết bởi giới hạn 20 tokens. Với cùng ngân sách,
prompt spatial có thể giữ ít thuộc tính/đối tượng hơn prompt objects. Metadata báo
số prompt bị cắt và số tiền tố quan hệ vượt ngân sách; cần kiểm tra trước train.
So sánh này đo việc thêm quan hệ trong ngân sách cố định, chưa cô lập hoàn toàn
ảnh hưởng của độ dài và thứ tự token. Nếu nhiều prompt bị cắt, thiết kế một lượt
ablation ngân sách khác và chạy lại cả hai loại, không chỉ tăng một nhánh.

## Thứ tự notebook Kaggle

1. Import `vlm_extract_kaggle.ipynb`, gắn MSCOCO ảnh và split JSON, bật GPU/Internet.
   Chạy mặc định SMOKE=True: 100 ảnh, xem 10 scene đầu và metadata truncation.
   Kiểm tra trực tiếp một số ảnh với JSON. Ghi seconds/image và ETA để chọn số shard;
   mặc định 8 chỉ là cấu hình ban đầu, không phải số bắt buộc.
2. Copy `resolved_revision` trong file `.meta.json` của pilot vào REVISION.
   Chọn PARTS dựa trên pilot, giữ cố định cho mọi shard. Đổi SMOKE=False,
   chạy PART lần lượt 1..PARTS ở các phiên/version riêng. Save & Run All lưu các file
   trong `/kaggle/working/vlm_outputs`. Giữ `.jsonl` và `.jsonl.meta.json` của mọi part.
   Không cần gắn visual cache hoặc prompt cache YOLO trong bước tạo scene.
3. Upload/gắn tất cả full scene shards thành Input. Import `vlm_embed_kaggle.ipynb`,
   sửa SCENE_DIR. Bước này kiểm tra đúng split, profile VLM, không trùng ảnh và đủ
   train/val/test trước khi tạo hai file `prompt_vlm_objects.pt`, `prompt_vlm_spatial.pt`.
   Giữ cả hai `.pt` và metadata `.json` thành Dataset; không dùng cache smoke để train.
4. Import `vlm_train_kaggle.ipynb`, gắn MSCOCO/split, 3 visual cache .h5 hiện tại
   và Dataset prompt mới. Sửa PROMPT_DIR. Chạy VARIANT='objects' rồi 'spatial'
   trong hai version riêng. Có verify-cache trước train và test đầy đủ tự động.
   Nếu metric lỗi, caption và ground truth vẫn được lưu để tính lại bằng mode metrics.

Extraction lưu mỗi scene hợp lệ ngay và cho chạy tiếp cùng file, cùng cấu hình.
JSON không hợp lệ được thử sửa tối đa 2 lần với phản hồi cụ thể từ validator
(`--retries 2`), vẫn dùng cùng ảnh và không dùng caption gốc. Mỗi lỗi được lưu
vào `.errors.jsonl`; nếu hết lượt vẫn sai thì dừng, không thay bằng prompt rỗng.
Scene lưu số lần retry để kiểm tra chất lượng dữ liệu. Trong phiên
mới, copy scene và metadata cũ từ Input sang một đường dẫn output ghi được rồi
trỏ --output đến đó để resume. Save & Run All không tự mang file phiên tương tác
sang phiên mới. File bị ghi dở một dòng phải được kiểm tra/sửa trước khi resume.

## CLI tương đương

```bash
python -u build_vlm_prompts.py generate --part 1 --parts 8 --limit 100 --output /kaggle/working/vlm_outputs/scenes_part_01_of_08_smoke.jsonl
python -u build_vlm_prompts.py generate --part 1 --parts 8 --limit 0 --revision <resolved_revision> --output /kaggle/working/vlm_outputs/scenes_part_01_of_08.jsonl
python -u build_vlm_prompts.py embed --scenes /kaggle/input/your-vlm-scenes/scenes_part_*.jsonl --output-dir /kaggle/working/vlm_prompt_cache
```

Không chọn prompt, hyperparameter hoặc checkpoint dựa trên test metric.
Dùng val để quyết định thay đổi; test dùng để báo cáo cấu hình đã chốt.
Giữ JSON metric, caption dự đoán, checkpoint, metadata prompt, mã commit và thời gian
train thực đo của mỗi thực nghiệm. Một seed chỉ cho kết quả sơ bộ; nếu có cải thiện,
chạy thêm seed cho mốc và biến thể tốt nhất trước khi khẳng định đóng góp.
