# BÁO CÁO TIẾN ĐỘ TUẦN

**Đề tài:** Căn chỉnh ngữ nghĩa thị giác cho bài toán sinh mô tả ảnh  
**Dữ liệu:** MS COCO 2014, Karpathy split  
**Ngày tổng hợp:** 24/09/2026

## 1. Mục tiêu trong tuần

Tuần này tập trung kiểm tra chất lượng dữ liệu YOLO, thử căn chỉnh trực tiếp object
với vùng ảnh, tối ưu thời gian huấn luyện và xây dựng một hướng căn chỉnh mới bằng
learnable visual queries lấy cảm hứng từ Q-Former.

Các thay đổi được thực hiện tuần tự để xác định đóng góp của từng thành phần, thay
vì thêm nhiều kỹ thuật vào cùng một lần chạy.

## 2. Công việc đã hoàn thành

### 2.1. Kiểm tra thống kê kết quả YOLO

Đã xây dựng công cụ audit toàn bộ object detection cache thay vì kiểm tra thủ công
từng ảnh. Kết quả chính:

| Nội dung | Kết quả |
|---|---:|
| Tổng số ảnh | 123.287 |
| Ảnh có dữ liệu detection | 123.287/123.287 |
| Tổng số detection | 867.045 |
| Số detection trung bình mỗi ảnh | 7,03 |
| Số nhãn object | 80 |
| Ảnh không có detection | 1.052 (0,853%) |
| Detection có confidence dưới 0,5 | 281.462 (32,462%) |
| Bounding box rất nhỏ | 52.703 (6,078%) |
| Bounding box gần phủ toàn ảnh | 8.748 (1,009%) |
| Bounding box sai tọa độ hoặc vượt biên | 0 |
| Tỷ lệ nhãn `person` | 31,26% tổng detection |

Từ kết quả audit, region targets được lọc theo cấu hình:

```text
confidence >= 0.5
0.001 <= bbox area ratio <= 0.9
tối đa 10 regions mỗi ảnh
```

Region target cache hoàn chỉnh gồm 123.287 ảnh, 80 nhãn và có dung lượng khoảng
37,76 MB. File này được lưu riêng để các phiên Kaggle sau không phải mở lại toàn bộ
ảnh COCO.

### 2.2. Thử nghiệm Object–Region Alignment Loss

Đã bổ sung một auxiliary loss căn chỉnh vùng ảnh với object label:

```text
CLIP visual patches trong YOLO bbox
                ↓ mean pooling
          region representation
                ↓
phân loại về 80 CLIP object-label prototypes
```

Hàm mất mát:

```text
L_total = L_caption + λ × L_region
```

Đã thử hai trọng số `λ=0.10` và `λ=0.02`. Phần region pooling ban đầu chạy chậm do
lặp từng region trên GPU; sau đó đã được vector hóa để xử lý tất cả region trong
batch cùng lúc.

### 2.3. Chuyển huấn luyện sang hai GPU

Đã tích hợp Hugging Face Accelerate với DistributedDataParallel:

- Hai GPU Tesla T4 cùng tham gia huấn luyện.
- Global batch size giữ nguyên 32, mỗi GPU nhận 16 ảnh.
- Gradient được đồng bộ trước khi cập nhật mô hình.
- Chỉ rank 0 ghi checkpoint, history và log.
- Loss hiển thị là trung bình của hai GPU.
- Có heartbeat log sau mỗi 100 batch.
- Process group NCCL được đóng đúng sau khi train.

Thời gian bản region alignment λ=0.02 giảm xuống khoảng 2 giờ cho train và validation,
nhanh hơn đáng kể so với các lượt một GPU trước đó. Thời gian này phản ánh thay đổi
hạ tầng huấn luyện, không phải chất lượng kiến trúc.

### 2.4. Xây dựng Lightweight Q-Former

Đã xây dựng một module learnable visual queries bằng PyTorch, không tải Q-Former
pretrained hoặc weights BLIP-2 bên ngoài.

Cơ chế:

```text
197 CLIP visual tokens
          ↓
32 learnable visual queries
          ↓
2 tầng query self-attention + query-to-visual cross-attention + FFN
          ↓
32 visual query features
          ↓
Prompt-to-query cross-attention + Gate
          ↓
[20 grounded prompt tokens; 32 visual query tokens]
          ↓
Caption decoder
```

So với pipeline Gate trước, decoder memory giảm từ 217 tokens xuống 52 tokens. Bản
Q-Former đầu tiên chỉ dùng caption loss, không dùng YOLO bbox và không dùng region
alignment loss để đo riêng đóng góp của learnable queries.

### 2.5. Bổ sung image–text contrastive alignment cho Q-Former

Đã bổ sung ITC loss để căn chỉnh 32 visual queries với embedding CLIP của 5 caption
tham chiếu thuộc train split. Điểm ảnh–văn bản lấy query có cosine similarity cao
nhất; các mẫu còn lại trong global batch 32 là negative. Loss tổng là
`L_caption + 0.1 × L_ITC`. Thí nghiệm này không dùng region loss để đo riêng đóng
góp của căn chỉnh ảnh–caption. Caption embedding được cache thành HDF5 để không
chạy lại CLIP Text Encoder trong từng batch.

## 3. Kết quả trên test split 5.000 ảnh

Các mô hình trong bảng này đều được đánh giá trên test split. Các giá trị của mô
hình cũ được ghi theo kết quả đã lưu trước đó.

| Mô hình | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | Thời gian huấn luyện |
|---|---:|---:|---:|---:|---:|---:|
| Không Prompt | 0.7320 | 0.3250 | 0.2640 | 0.5530 | 1.0970 | 3 giờ 30 phút |
| Chỉ có prompt | 0.7650 | 0.3650 | 0.2860 | 0.5690 | 1.1720 | 8 giờ 30 phút |
| Cross-Attention Prompt–Ảnh | 0.7660 | 0.3640 | 0.2810 | 0.5680 | 1.1690 | 7 giờ 16 phút |
| Cross-Attention có Gate | **0.7660** | **0.3660** | 0.2830 | **0.5710** | **1.1820** | khoảng 5 giờ |
| Chỉ có object label | 0.7620 | 0.3640 | 0.2820 | 0.5690 | 1.1590 | 5 giờ 30 phút |
| Gate + Region Alignment λ=0.10 | 0.7604 | 0.3584 | 0.2790 | 0.5642 | 1.1532 | khoảng 5 giờ 14 phút |
| Gate + Region Alignment λ=0.02 | 0.7641 | 0.3624 | **0.2834** | 0.5687 | 1.1705 | dùng dual GPU |

Kết luận từ test:

- Gate hiện vẫn là mô hình có CIDEr, BLEU-4 và ROUGE-L tốt nhất.
- λ=0.10 làm giảm toàn bộ metric, cho thấy auxiliary task lấn át caption task.
- Giảm xuống λ=0.02 giúp phục hồi kết quả nhưng vẫn chưa vượt Gate.
- Object–region classification loss hiện tại được giữ như một ablation, không chọn
  làm mô hình cuối.

## 4. Kết quả trên validation split 5.000 ảnh

Không trộn các số validation dưới đây vào bảng test phía trên.

| Mô hình | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gate + Region Alignment λ=0.02 | 0.7666 | 0.6084 | 0.4718 | 0.3654 | 0.2828 | 0.5686 | 1.1679 |
| Lightweight Q-Former, 32 queries, 2 layers | 0.7672 | 0.6089 | 0.4744 | **0.3703** | **0.2830** | **0.5712** | **1.1740** |
| Q-Former + ITC α=0.1 | **0.7684** | **0.6105** | **0.4747** | 0.3687 | 0.2828 | 0.5705 | 1.1705 |

Q-Former so với region alignment λ=0.02 trên cùng validation split:

| Metric | Chênh lệch |
|---|---:|
| BLEU-1 | +0.0006 |
| BLEU-2 | +0.0005 |
| BLEU-3 | +0.0026 |
| BLEU-4 | +0.0049 |
| METEOR | +0.0002 |
| ROUGE-L | +0.0026 |
| CIDEr | +0.0061 |

Kết quả Q-Former tích cực nhất ở BLEU-3 và BLEU-4. Điều này cho thấy visual memory
được cô đọng có thể hỗ trợ decoder tạo các cụm từ dài tốt hơn. Tuy nhiên, mức tăng
vẫn nhỏ và mới được kiểm tra với một seed.

Chưa thể kết luận Q-Former tốt hơn Gate baseline vì Q-Former hiện có metric
validation, còn Gate trong bảng hiện có metric test. Cần đánh giá Q-Former trên test
sau khi đã chọn bằng validation để có phép so sánh cùng split.

ITC α=0.1 so với Q-Former không ITC tăng BLEU-1 `0.0012`, BLEU-2 `0.0016` và
BLEU-3 `0.0003`, nhưng giảm BLEU-4 `0.0016`, METEOR `0.0002`, ROUGE-L `0.0007`
và CIDEr `0.0035`. Kết quả trái chiều và rất nhỏ, nên chưa chọn checkpoint ITC để
đánh giá test. Điều này cho thấy căn chỉnh toàn ảnh–caption chưa trực tiếp cải thiện
khả năng tạo caption dài và chính xác hơn.

## 5. Kết luận trong tuần

1. Visual feature cache và dual-GPU Accelerate giúp giảm đáng kể thời gian thực nghiệm.
2. YOLO cache có coverage đầy đủ nhưng chứa nhiều detection confidence thấp và các
   nhãn mất cân bằng, đặc biệt là `person`.
3. Căn chỉnh vùng ảnh bằng auxiliary object classification loss chưa cải thiện caption.
4. λ=0.02 phù hợp hơn λ=0.10 nhưng vẫn thấp hơn Gate trên test.
5. Learnable visual queries/Q-Former cho kết quả validation tốt hơn region-loss
   candidate ở toàn bộ metric.
6. Q-Former là hướng có triển vọng hơn việc tiếp tục dò trọng số λ cho region loss.
7. ITC α=0.1 chỉ tăng BLEU-1/2/3 nhưng làm giảm BLEU-4, METEOR, ROUGE-L và CIDEr;
   giữ Q-Former không ITC làm candidate hiện tại.
8. Các chênh lệch hiện tại mới có một seed, chưa phải bằng chứng về ý nghĩa thống kê.

## 6. Các vấn đề đã xử lý

| Vấn đề | Cách xử lý |
|---|---|
| Visual cache khác ID Karpathy/COCO | Dùng COCO ID từ filename và kiểm tra coverage |
| Sai số cache FP16 | Kiểm tra cosine similarity và relative error |
| Tạo region targets mất khoảng một giờ | Lưu `region_targets_yolo.pt` thành Kaggle Input |
| Region loss chạy rất chậm | Vector hóa pooling bbox trên toàn batch |
| Kaggle không hiện tiến độ `tqdm` | Thêm log dòng thường mỗi 100 batch |
| Chỉ sử dụng GPU 0 trong T4 x2 | Chuyển sang Accelerate DDP hai process |
| Log và checkpoint bị ghi hai lần | Chỉ cho rank 0 ghi output |
| Cảnh báo NCCL khi thoát | Gọi `accelerator.end_training()` |
| Visual memory 197 tokens lớn hơn prompt 20 tokens | Thử nén thành 32 learnable visual queries |
| Căn chỉnh mới chỉ dựa vào caption loss | Thêm ITC giữa visual queries và CLIP caption embeddings |

## 7. Công việc tiếp theo

1. Giữ Q-Former không ITC làm candidate và đánh giá trên test 5.000 ảnh.
2. So sánh Q-Former và Gate trên cùng test split để chọn backbone.
3. Với backbone tốt hơn, sinh nhiều caption ứng viên thay vì chỉ lấy beam tốt nhất.
4. Xây dựng Object Consistency Checker đối chiếu object trong caption với YOLO cache.
5. Re-rank ứng viên bằng điểm ngôn ngữ, điểm nhất quán object và độ dài.
6. Báo cáo thêm hallucination/CHAIR nếu dữ liệu annotation đáp ứng.
7. Nếu Q-Former cải thiện trên test, chạy thêm seed cho Gate và Q-Former trước khi
   khẳng định đóng góp cuối cùng.

## 8. Tệp và hướng dẫn liên quan

- [YOLO_AUDIT_RESULTS.md](YOLO_AUDIT_RESULTS.md): thống kê detection cache.
- [REGION_ALIGNMENT_RESULTS.md](REGION_ALIGNMENT_RESULTS.md): kết quả ablation region loss.
- [EVALUATE_L002_TEST.md](EVALUATE_L002_TEST.md): workflow test λ=0.02.
- [QFORMER_EXPERIMENT.md](QFORMER_EXPERIMENT.md): cơ chế, code Kaggle và kết quả Q-Former.
- [QFORMER_ITC_EXPERIMENT.md](QFORMER_ITC_EXPERIMENT.md): cơ chế, code Kaggle và kết quả ITC.
- [REGION_ALIGNMENT_L002_DUAL_GPU.md](REGION_ALIGNMENT_L002_DUAL_GPU.md): workflow Accelerate hai GPU.

## 9. Artifact cần lưu

Đối với mỗi thí nghiệm cần giữ:

```text
checkpoint epoch 10
train_history_h1_2.json
val/test captions JSON
ground-truth JSON
metrics JSON
commit GitHub
seed, split và cấu hình decoding
```

Các file này giúp tái lập kết quả, tính lại metric và kiểm tra caption mà không phải
train hoặc sinh caption lại.
