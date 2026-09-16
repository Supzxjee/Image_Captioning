# Image Captioning: H1.2-G

Biến thể có gate của H1.2 prompt-to-visual cross-attention, dùng MS COCO 2014 và CLIP prompt token cache. Giữ notebook H1.2 gốc để đối chiếu.

## Cơ chế

- Prompt là Q; visual tokens là K và V trong cross-attention encoder.
- Gate theo từng token và chiều đặc trưng: `sigmoid(Linear(concat(prompt, attended_visual)))`.
- Prompt cập nhật: `LayerNorm(prompt + Dropout(gate * attended_visual))`.
- Gate khởi tạo với weight 0, bias -2 (khoảng 0.119), giúp bắt đầu với cập nhật visual nhỏ. Đây không phải khởi tạo tương đương hoàn toàn H0 vì vẫn có LayerNorm của H1.2.
- Nối prompt cập nhật và visual thành memory cho decoder. Không thêm attention chiều ngược, không thêm loss căn chỉnh trong thí nghiệm này.
- Chưa có kết quả thực nghiệm; không bảo đảm tăng điểm.

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

`--epochs 1` vẫn train toàn bộ tập train trong một epoch. Có thể kiểm tra shape bằng cell shape test trong notebook trước.

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

Đã kiểm tra cú pháp script và các cell Python. Chưa chạy training/inference trên GPU tại máy phát triển; chưa xác nhận tương thích toàn bộ dependencies Kaggle. Notebook gốc được giữ nguyên. So sánh các mô hình với cùng dữ liệu, seed, ngân sách train, checkpoint selection và decoding.
