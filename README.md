# Image Captioning: H1.2-G

Biến thể có gate của H1.2 prompt-to-visual cross-attention, dùng MS COCO 2014 và CLIP prompt token cache. Giữ notebook H1.2 gốc để đối chiếu.

## Cơ chế

- Prompt là Q; visual tokens là K và V trong cross-attention encoder.
- Gate theo từng token và chiều đặc trưng: `sigmoid(Linear(concat(prompt, attended_visual)))`.
- Prompt cập nhật: `LayerNorm(prompt + Dropout(gate * attended_visual))`.
- Gate khởi tạo với weight 0, bias -2 (khoảng 0.119), giúp bắt đầu với cập nhật visual nhỏ. Đây không phải khởi tạo tương đương hoàn toàn H0 vì vẫn có LayerNorm của H1.2.
- Nối prompt cập nhật và visual thành memory cho decoder. Không thêm attention chiều ngược, không thêm loss căn chỉnh trong thí nghiệm này.
- Chưa có kết quả thực nghiệm; không bảo đảm tăng điểm.

## Cấu trúc mã nguồn

```text
captioning/
  config.py       # Hằng số kiến trúc, cấu hình và đường dẫn
  cli.py          # Tham số dòng lệnh
  pipeline.py     # Ghép các bước và chọn train/evaluate
  visual_cache.py # Đọc HDF5 theo COCO ID, an toàn cho worker
  cache_verification.py # Đối chiếu cache với CLIP trực tiếp
  data.py         # Chia tập COCO, cache prompt, Dataset và DataLoader
  tokenizer.py    # Vocabulary và mã hóa caption
  models.py       # CLIP encoder, gated cross-attention và decoder
  checkpoints.py # Nạp checkpoint: warm-start hoặc đánh giá
  training.py     # Train, loss và lưu checkpoint
  inference.py    # Sinh caption bằng beam search
  metrics.py      # Ground truth và BLEU/METEOR/ROUGE-L/CIDEr
  evaluation.py   # Chạy val/test, log và lưu kết quả
  __main__.py     # Cho phép chạy python -m captioning
train_h1_2_gated.py           # Entry point giữ tương thích lệnh cũ
kltn_mscoco_h1_2_gated.ipynb   # Notebook launcher, không sao chép implementation
kltn_mscoco_h1_2_cross_attention.ipynb # H1.2 gốc để đối chiếu
```

Sửa kiến trúc trong `captioning/models.py`, training trong `captioning/training.py`, và decoding trong `captioning/inference.py`. Import package không nạp dữ liệu, tải model hoặc chạy training. Thư viện tính metrics chỉ được import khi tính điểm, nên train không phụ thuộc việc khởi tạo METEOR.

Lệnh cũ vẫn hoạt động; có thể dùng tương đương:

```python
!python -u -m captioning --mode train --epochs 10
```

Các tùy chọn bổ sung: `--seed`, `--batch-size`, `--num-workers`, `--base-path`, `--dataset-json-path`, `--prompt-cache-path`, `--work-dir`, `--experiment-name`. Xem toàn bộ bằng `!python train_h1_2_gated.py --help`. Các tùy chọn CLI thay thế biến môi trường `CAPTION_*` của notebook cũ.

## Chạy trên Kaggle

Bật Internet và GPU, gắn các dataset cung cấp đúng ba đường dẫn:

```text
/kaggle/input/datasets/vuthetam/mscoco-2014/images
/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json
/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt
```

Chạy trong một cell Kaggle:

```python
!git clone https://github.com/Supzxjee/Image_Captioning.git /kaggle/working/Image_Captioning
%cd /kaggle/working/Image_Captioning
!pip install -q -r requirements.txt
!python -u train_h1_2_gated.py --mode train --epochs 10
```

Nếu đã clone, dùng `!git -C /kaggle/working/Image_Captioning pull --ff-only` để cập nhật. Để tái lập thí nghiệm, ghi lại `!git rev-parse HEAD` và dùng cùng commit cho lần chạy sau. Nếu repo private, cần token qua Kaggle Secrets; không ghi token trong notebook hay repo.

Script dùng một GPU. Việc cấp T4 x2 không tự động làm script sử dụng cả hai GPU.

### Chạy thử trước

```python
!python -u train_h1_2_gated.py --mode train --epochs 1
!python -u train_h1_2_gated.py --mode evaluate --epochs 1 --split val --limit 50
```

`--epochs 1` vẫn train toàn bộ tập train trong một epoch. Notebook H1.2-G là launcher gọi cùng các module Python; cell đánh giá được comment mặc định.

### Đánh giá riêng sau train

```python
!python -u train_h1_2_gated.py --mode evaluate --checkpoint /kaggle/working/h1_2_gated_prompt_to_visual_crossattn/checkpoints/model_h1_2_crossattn_epoch_10.pth --split val --limit 500
```

`--limit 0` (mặc định) đánh giá toàn bộ split. Sàng lọc checkpoint bằng cùng 500 ảnh validation, xác nhận trên toàn bộ validation rồi mới chạy `--split test`. Không chọn cấu hình dựa trên điểm test.

Inference giữ beam size 5, tối đa 29 bước token như H1.2. Chưa batch hóa beam search, nên full evaluation vẫn có thể lâu. Mỗi 50 ảnh có log thời gian/ETA và lưu predictions tạm; file tạm không tự động resume và không phải kết quả đầy đủ.

Tính METEOR cần Java; môi trường Kaggle phải có `java` trên PATH.

### Fine-tune từ H1.2

```python
!python -u train_h1_2_gated.py --mode train --epochs 3 --lr 1e-5 --checkpoint /kaggle/input/YOUR_CHECKPOINT_DATASET/model_h1_2_crossattn_epoch_10.pth
```

Thay đường dẫn checkpoint thực tế. Lệnh nạp model weights của H1.2 hoặc H1.2-G, khởi tạo optimizer mới và đánh số epoch lại từ 1; không phải resume đầy đủ. Không dùng checkpoint H0 vì thiếu attention của H1.2. Với checkpoint cũ không lưu vocabulary, cần đảm bảo cùng dataset/tokenizer. Khi so sánh, H1.2 đối chứng cũng phải có cùng ngân sách fine-tune để tránh nhầm hiệu quả của gate với hiệu quả train thêm.

## Dùng visual cache HDF5 có sẵn

Gắn dataset [CLIP TrainVal Features MSCOCO 2014](https://www.kaggle.com/datasets/vuthetam/clip-trainval-features-mscoco-2014) bằng Add Input. Copy đúng đường dẫn thư mục từ bảng Input; ví dụ dưới đây dùng đường dẫn ngắn thường gặp, môi trường của bạn có thể là `/kaggle/input/datasets/vuthetam/clip-trainval-features-mscoco-2014`.

Cache cần có hai keys `features` (N, 197, 768), float16/float32 và `imgids` (N,), COCO image IDs. Reader ghép theo ID lấy từ filename, không dùng `eval_id` hoặc thứ tự dòng. Gộp tất cả `.h5` trong thư mục; kiểm tra thiếu ảnh, ID trùng, shape và giá trị không hữu hạn. Không âm thầm chuyển sang ảnh trực tiếp khi cache thiếu mẫu.

### Xác nhận cache trước khi train

```python
CACHE = '/kaggle/input/clip-trainval-features-mscoco-2014'
!python -u train_h1_2_gated.py --mode verify-cache --visual-cache "$CACHE" --split val --limit 10
```

So sánh các features cache với `last_hidden_state` của CLIP và preprocessing hiện tại. Cho phép sai số FP16 (`atol=0.01`, `rtol=0.005`); nếu không đạt, dừng để kiểm tra notebook tạo cache. PASS trên mẫu nhỏ không đảm bảo mọi dòng cache đúng; nên thử thêm ảnh và đối chiếu cách tạo cache. Lệnh verification cần cả ảnh gốc và prompt cache/dataset JSON như pipeline bình thường.

### Train và evaluate dùng cache

```python
!python -u train_h1_2_gated.py --mode train --epochs 10 --visual-cache "$CACHE" --experiment-name h1_2_gated_visualcache
!python -u train_h1_2_gated.py --mode evaluate --epochs 10 --split test --visual-cache "$CACHE" --experiment-name h1_2_gated_visualcache
```

Dùng tên experiment riêng để tránh ghi đè mốc đã train. Khi evaluate checkpoint cũ, truyền đường dẫn bằng `--checkpoint`; cache path không bắt buộc trùng đường dẫn lúc train nhưng phải chứa đúng features. Có thể truyền từng file bằng cách lặp `--visual-cache /path/train.h5 --visual-cache /path/val.h5`.

Mỗi worker DataLoader mở HDF5 riêng và chỉ đọc features từng ảnh; chỉ bảng IDs nằm trong RAM. Tensors FP16 trên disk được chuyển thành FP32 để dùng với projection hiện tại. Cache nằm trước projection, nên projection, attention, gate và decoder vẫn được train. Trong chế độ cache, CLIP backbone vẫn được nạp để giữ tương thích checkpoint nhưng không chạy forward trong train/inference. Không dùng cách này nếu bạn muốn fine-tune backbone hoặc dùng augmentation ảnh ngẫu nhiên.

History mới có `train_seconds` từng epoch và log giây/batch để đo tốc độ. Chưa đo tốc độ với dataset thật trên Kaggle, chưa kiểm chứng preprocessing của cache bạn cung cấp; cần chạy verification trước. Đánh giá vẫn chạy beam search từng ảnh như cũ.

### Kiểm tra cục bộ

```bash
python -m unittest discover -s tests -v
```

Tests dùng HDF5 tổng hợp để kiểm tra ID/shards, coverage, dữ liệu lỗi và worker spawn; encoder tests dùng backbone giả để kiểm tra đường cache bỏ qua backbone, giữ gradients và khớp đầu ra với đường pixels. Không tải CLIP weights trong tests, nên tests không xác nhận cache thật khớp preprocessing.

## Đầu ra

Trong `/kaggle/working/h1_2_gated_prompt_to_visual_crossattn/`:

- `checkpoints/`: checkpoint mỗi epoch, gồm weights, optimizer và vocabulary.
- `train_history_h1_2.json`: loss train.
- `evaluation/`: predictions, ground truth, metrics theo split và số ảnh; predictions tạm mỗi 50 ảnh.

CLI mặc định không tự chạy full test sau train; bật `--test-after-train` nếu cần. Thư mục `/kaggle/working` nằm ngoài thư mục clone nên kết quả không bị lẫn với mã nguồn repo. Tải checkpoint/kết quả từ phiên Kaggle sau khi hoàn tất.

## Kiểm chứng

Đã kiểm tra cú pháp các module và notebook, CLI, cấu hình, việc import không khởi chạy pipeline. Cả 16 regression tests CPU cho HDF5 reader và cached encoder/decoder đã qua (backbone giả, không tải CLIP). Chưa chạy training/inference trên GPU tại máy phát triển; chưa xác nhận tương thích toàn bộ dependencies Kaggle. Notebook gốc được giữ nguyên. So sánh các mô hình với cùng dữ liệu, seed, ngân sách train, checkpoint selection và decoding.

## Train xong tự chạy full test

```python
!python -u train_h1_2_gated.py --mode train --epochs 10 --visual-cache "$CACHE" --experiment-name h1_2_gated_visualcache --test-after-train
```

Tùy chọn này dùng checkpoint epoch cuối vừa train, cùng visual cache, toàn bộ split test (không áp dụng `--limit`). Nếu warm-start, không dùng nhầm checkpoint đầu vào. Coverage test được kiểm tra trước training khi có cache. Mặc định train vẫn không chạy test; chỉ bật khi đã chốt thí nghiệm. Notebook launcher có biến `TEST_AFTER_TRAIN = True` để chạy cả train và test qua Save & Run All.

### Chỉ sinh caption hoặc tính metric từ file

```python
!python -u train_h1_2_gated.py --mode predict --checkpoint /path/checkpoint.pth --split test --visual-cache "$CACHE" --experiment-name h1_2_gated_visualcache
!python -u train_h1_2_gated.py --mode metrics --predictions /kaggle/working/h1_2_gated_visualcache/evaluation/test_5000_captions_h1_2_gated.json --ground-truth /kaggle/working/h1_2_gated_visualcache/evaluation/test_5000_gt_h1_2_gated.json --metrics-output /kaggle/working/h1_2_gated_visualcache/evaluation/test_5000_metrics_h1_2_gated.json
```

`predict` lưu captions và ground truth, không tính metric. `evaluate` sinh caption rồi tính metric. Captions và ground truth luôn được lưu trước scoring; nếu thiếu pycocoevalcap/Java hoặc scoring lỗi, cài dependencies và chạy lại `metrics`, không chạy lại `evaluate`. `metrics` không nạp checkpoint, model, dữ liệu ảnh hay visual cache. Nó kiểm tra ID khớp giữa hai file trước khi chấm.

### Cache dùng số thứ tự JSON thay vì COCO ID

Nếu `imgids` là chỉ số zero-based của `images` trong cùng `dataset_coco.json`, thêm `--visual-cache-id-key eval_id` vào cả verify-cache, train và evaluate. Không tự động đoán loại ID. Chạy verify-cache để đối chiếu với ảnh thực; số lượng và khoảng ID chưa đủ chứng minh cùng thứ tự JSON. Mặc định vẫn dùng COCO ID từ filename.

Cache được kiểm tra có 113287 train + 5000 val, tổng 118287 mẫu; không thể chứa toàn bộ 123287 ảnh. Không bật `--test-after-train` nếu thiếu cache test. Sau train, evaluate checkpoint trong lệnh riêng, bỏ `--visual-cache` để chạy CLIP trực tiếp cho tập test. Có thể bổ sung cache test sau.

### Profile cache RAG_Captioning

Đối chiếu repo https://github.com/vuthetam/RAG_Captioning: extract_visual_features.py lưu raw CLIP ViT-B/16 last_hidden_state với resize bicubic, antialias=True và CUDA autocast FP16. imgids lấy từ trường imgid trong sentences của Karpathy JSON, không phải COCO filename ID; không cần giả định nó luôn bằng thứ tự dòng JSON.

Thêm cả ba tùy chọn vào verification, train và test trực tiếp:

```text
--visual-cache-id-key karpathy_id --visual-preprocessing bicubic --visual-precision amp-fp16
```

Giữ mặc định bilinear/fp32 cho checkpoint baseline cũ. Profile bicubic/AMP khác preprocessing/precision các thí nghiệm trước, nên cần báo cáo riêng và dùng profile nhất quán khi test checkpoint được train bằng cache này. Repository reference cho thấy cấu hình tạo cache; cache dataset thực tế vẫn phải verify trên Kaggle. Không tăng dung sai để ép cache không khớp vượt qua kiểm tra. Nếu vẫn mismatch, đối chiếu version Transformers, ảnh/JSON và GPU/kernel/batch dùng khi tạo cache.

Cache train/val không có test: bật `--test-after-train --test-visual-source images` để tự train bằng cache rồi full test từ ảnh gốc; giữ cùng bicubic/AMP cho test. Chế độ same (mặc định) yêu cầu cache chứa đủ ảnh test. Notebook launcher đã chọn images cho test và profile RAG_Captioning khi VISUAL_CACHE không rỗng.
