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

## Đầu ra

Trong `/kaggle/working/h1_2_gated_prompt_to_visual_crossattn/`:

- `checkpoints/`: checkpoint mỗi epoch, gồm weights, optimizer và vocabulary.
- `train_history_h1_2.json`: loss train.
- `evaluation/`: predictions, ground truth, metrics theo split và số ảnh; predictions tạm mỗi 50 ảnh.

Không tự chạy full test sau train. Thư mục `/kaggle/working` nằm ngoài thư mục clone nên kết quả không bị lẫn với mã nguồn repo. Tải checkpoint/kết quả từ phiên Kaggle sau khi hoàn tất.

## Kiểm chứng

Đã kiểm tra cú pháp các module và notebook, CLI, cấu hình, việc import không khởi chạy pipeline, và đối chiếu AST kiến trúc với phiên bản trước khi tách module. Chưa chạy training/inference trên GPU tại máy phát triển; chưa xác nhận tương thích toàn bộ dependencies Kaggle. Notebook gốc được giữ nguyên. So sánh các mô hình với cùng dữ liệu, seed, ngân sách train, checkpoint selection và decoding.
