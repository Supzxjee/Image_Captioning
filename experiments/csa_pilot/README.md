# Pilot CSA hai pha trên Q-Former

Chạy [CSA_TWO_PHASE_PILOT_2GPU.ipynb](CSA_TWO_PHASE_PILOT_2GPU.ipynb), **T4×2, Internet ON, Save & Run All**. Người dùng tự chạy, một notebook dùng cả hai GPU. Không cần chạy lại A hoặc smoke.

## Dataset

Như CSA smoke: `qformer-scst-object-w05-legacy`, `prompt-cache`, `visual-cache`, `vuthetam/mscoco-2014` (JSON), `objectdetectionecache`. Path có sẵn trong cell đầu. Không cần UPT, VG hoặc ảnh gốc. Dùng checkpoint ban đầu của A, không lấy checkpoint đã học G/H/GV hoặc smoke.

## Recipe

Giữ nguyên core/banks/readout đã [smoke thành công](../csa_core_smoke/SMOKE_RESULT.md). Globalbatch8, seed42, train10k/dev1000 cũ. Core2object+1relation, input32latentqueries, memory `[baseline;Vo;Vr]`, bank luôn frozen.

| | Pha1 | Pha2 |
|---|---|---|
| Updates |1250 |1250 |
| Cập nhật |Chỉ core CSA |Baseline như A + core |
| LRbase |Frozen |5e-6 |
| LRcore |Step1=1e-6, tăng tuyến tính đạt1e-4 ởstep100, giữ tới1250 |2e-5 |
| Loss |CE |CE200steps rồi CE+.05SCST |
| SCST reward |Không |CIDErD+.5object_count_v2 |

AdamW options như A, clipglobalnorm1, FP32 strict numeric profile, dropoutoff với gradient bật. Hai pha chạy trong **hai torchrun process groups riêng**, không tái dùng optimizer state hoặc DDP hooks. Pha2 seed42+rank trước optimizer/DDP/trainiterator, sampler seed42 epoch0, đúng lịch A. Tham số decoder frozen ởpha1 nhưng phép tính decoder vẫn truyền gradient về core. Không thêm gate/loss mới.

Nguồn/bank được tải từ tác giả tại commit0cc570b, kiểmSHA trước load. Root runtime pin dfa1d4e kèm hash-checked overlays. Mỗi nguồn được giữ trong ZIPreview. Stage chỉCLIP/legacy, bỏ thông tin UPT không dùng; giữ handle HDF5 nguồn/đích mở theo lượt đọc. Thay đổi I/O không thay dtype hoặc giá trị tensor; chưa đo tốc độ toàn bộ trên Kaggle.

Cuối pha1 kiểm digest baseline/bank bất biến, DDP states giống nhau, lưu `phase1_checkpoint.bin`. Pha2 phải nạp đúng checkpoint có contract input/source/cohort/recipe và statehash hợp lệ, baseline vẫn đúng checkpoint ban đầu. Contract không phụ thuộc mount/output paths để có thể tiếp tục trên notebook mới.

## Kiểm tra và tiếp tục

Trước stage/train: tests recipe, synthetic CPU freeze/gradient/optimizer, tests DDP2GPU cả hai pha, dependency smoke PTB/CIDEr/CHAIR. Đây là tests tích hợp mới; kiểm toán output/gradient core gốc đã chạy ở smoke trước, không cần chạy lại.

Nếu phiên dừng sau pha1: lưu `phase1_checkpoint.bin` thành dataset, mount notebook mới rồi sửa `PHASE1_CHECKPOINT` trong cell đầu. Notebook mới stage lại cùng dữ liệu, bỏ pha1 và chỉ chạy pha2. Checkpoint cuối pha2/smoke sai format/phase sẽ bị từ chối. Đây là tiếp tục tại ranh giới pha, không resume giữa1250updates.

## Đánh giá và đầu ra

Chỉ cuối pha2 rawgreedy dev1000, original/on/off. Original là đường baseline trước cập nhật pha2 (baseline không đổi ởpha1). Off saupha2 không phải A. Cùng PTB/CIDEr full devIDF và CHAIR, ghi per-image scores/count denominators để root so matched A + paired bootstrap. Không dùng test, không chọn checkpoint theo dev, không rerank.

Giữ `checkpoint.bin` cuối và `phase1_checkpoint.bin`; gửi **`csa_two_phase_pilot_review.zip`**. ZIPkhông chứa checkpoint/bank/stage tensor; có source, protocol, source reference, train/devIDs, samplerorders, traininghistory, runtime/optimizer/transition/completion, captions, per-imageCIDEr, CHAIR và notebooklog. Save & Run All để giữ output trên Kaggle.

Gate soA:ΔCIDEr≥.01,paired95%lower>0; vượtoriginal; CHAIRs/i không tăng, recalldrop≤.01. Chỉ nếu đạt mới blindreview nhỏ trước xác nhận. Chạy hợp lệ không đạt thì dừng recipe, chưa fulltrain/tự kéo dài.

**Giới hạn:** CSA có1250core-onlyupdates nhiều hơn A,23Mtham số mới; gain hệ thống chưa cô lập thuật toán/capacity/compute. Mask/query không tự xác nhận grounding. Globalclip làm hai nhóm tham số ảnh hưởng chung scale update. Dev đã dùng nhiều lần, một seed.2500updates là ngân sách sàng lọc, không bảo đảm hội tụ. Smoke8ảnh không bảo đảm peak/time mọi batch; ghi riêng preparation/training/eval.

Local chỉ chạy `test_policy_static.py`, AST/portable artifact verification. Torch/CPUgradient/DDP/metric tests và pilot được notebook chạy trên Kaggle; chưa có kết quả mới khi phát hành.
