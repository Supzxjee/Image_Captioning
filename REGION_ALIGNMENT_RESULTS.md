# Kết quả ablation: Object–Region Alignment Loss

## Mục tiêu

Thí nghiệm kiểm tra liệu tín hiệu phụ căn chỉnh trực tiếp object label YOLO với vùng
ảnh CLIP có giúp mô hình caption hay không. Với mỗi bounding box, các patch CLIP
14×14 nằm trong box được mean-pooling; vector vùng được phân loại về một trong 80
nhãn object bằng CLIP text prototype. Hàm mất mát dùng khi train:

```text
L_total = L_caption + λ × L_region
```

Cấu hình caption model, prompt, visual cache, seed 42, beam size 5 và Karpathy split
được giữ nguyên. `L_region` chỉ được dùng khi train; kiến trúc inference không thêm
YOLO hay region target.

## Kết quả test 5.000 ảnh

| Mô hình | λ | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|---:|---:|
| Cross-Attention có Gate | 0 | 0.7660 | 0.3660 | 0.2830 | 0.5710 | 1.1820 |
| Gate + Object–Region Alignment | 0.10 | 0.7604 | 0.3584 | 0.2790 | 0.5642 | 1.1532 |
| Gate + Object–Region Alignment | 0.02 | 0.7641 | 0.3624 | 0.2834 | 0.5687 | 1.1705 |

Số đầy đủ của λ=0.02:

```json
{
  "Bleu_1": 0.7640647949083471,
  "Bleu_2": 0.6033879291808011,
  "Bleu_3": 0.46744240263669795,
  "Bleu_4": 0.36242529775932375,
  "METEOR": 0.28335493980543064,
  "ROUGE_L": 0.5687178209611304,
  "CIDEr": 1.1704900868487857
}
```

## Chênh lệch

λ=0.02 so với λ=0.10:

| Metric | Chênh lệch |
|---|---:|
| BLEU-1 | +0.0037 |
| BLEU-4 | +0.0040 |
| METEOR | +0.0044 |
| ROUGE-L | +0.0045 |
| CIDEr | +0.0173 |

λ=0.02 so với Gate không có region loss:

| Metric | Chênh lệch |
|---|---:|
| BLEU-1 | −0.0019 |
| BLEU-4 | −0.0036 |
| METEOR | +0.0004 |
| ROUGE-L | −0.0023 |
| CIDEr | −0.0115 |

## Kết luận

Giảm λ từ 0.10 xuống 0.02 làm giảm negative transfer và cải thiện toàn bộ metric
so với λ=0.10. Tuy vậy, λ=0.02 vẫn không vượt mốc Gate trên BLEU-1, BLEU-4,
ROUGE-L và CIDEr; mức tăng METEOR +0.0004 quá nhỏ để xem là cải thiện có ý nghĩa.
Vì vậy object–region classification loss hiện tại được giữ như một kết quả ablation,
không chọn làm mô hình cuối.

Kết quả cho thấy tín hiệu phân loại object theo bbox chưa chuyển thành caption tốt hơn.
Các nguyên nhân hợp lý gồm lưới CLIP 14×14 còn thô cho box nhỏ, mean-pooling làm mất
cấu trúc trong vùng, nhãn YOLO trùng thông tin đã có trong prompt/CLIP và caption
tham chiếu không bắt buộc nhắc đến mọi object. Loss này cũng chưa biểu diễn quan hệ
giữa các vùng.

Không so sánh trực tiếp `L_total` của hai λ với caption loss của Gate, vì `L_total`
đã cộng thêm thành phần `λ × L_region`. Khi phân tích history cần báo cáo riêng
`caption_loss`, `alignment_loss` và `train_loss`.

Các số hiện tại đến từ một seed. Không tuyên bố khác biệt thống kê nếu chưa chạy
thêm seed hoặc bootstrap trên caption-level scores. Bước kiến trúc tiếp theo là thử
learnable visual queries/Q-Former thay vì tiếp tục dò thêm λ cho cùng region loss.

## Tệp bằng chứng cần giữ

```text
test_5000_captions_h1_2_gated.json
test_5000_gt_h1_2_gated.json
test_5000_metrics_h1_2_gated.json
train_history_h1_2.json
model_h1_2_crossattn_epoch_10.pth
```

Workflow tái lập test: [EVALUATE_L002_TEST.md](EVALUATE_L002_TEST.md).
