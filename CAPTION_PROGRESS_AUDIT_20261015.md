# Chốt lại kết quả caption và mốc so sánh — 15/10/2026

**Đính chính: nhận định “chưa có kết quả nào tốt hơn baseline” là sai nếu áp dụng cho toàn bộ đề tài.** E2 và re-ranking cố định đã có mức tăng số học được ghi nhận từ trước. Các prototype quan hệ gần đây chưa vượt pipeline tốt nhất đó. Không được trộn hai kết luận này.

## Bảng đối chiếu đã có

Số dưới đây lấy từ Experiment Registry và tài liệu thí nghiệm đã lưu. Đây là đối chiếu lịch sử, không phải một lần chạy caption mới.

| Mô hình / cấu hình | CIDEr valid | CIDEr test | BLEU-4 test | CHAIRs test | Phạm vi kết luận |
|---|---:|---:|---:|---:|---|
| ViT-direct + Gate | Chưa xác minh | ~1.1820 | ~0.3660 | 4.90% | Mốc ViT; metric caption trong tài liệu chỉ có số làm tròn |
| Q-Former epoch 10 | 1.174012 | 1.188245 | 0.370666 | 4.32% | Baseline kiến trúc |
| Q-Former continuation thêm 3 epoch | 1.169574 | Chưa có | Chưa có | Chưa có | Đối chứng ngân sách train của E2 trên valid |
| E1 Object Semantic Alignment | 1.168861 | Không mở rộng | — | — | Không qua tiêu chí valid |
| E2 Cascade Alignment, prompt V1 | 1.186754 | 1.194745 | 0.371145 | 3.88% | Tăng nhỏ so với Q-Former; một seed |
| E2 không Soft Selector | 1.192112 | 1.195871 | 0.370954 | 3.88% | Ablation; CIDEr nhỉnh hơn E2 full, không thắng mọi độ đo |
| E2 Relation V2.1 | 1.189464 | 1.193298 | 0.372165 | 3.62% | Giảm CHAIR so với E2 V1 nhưng CIDEr thấp hơn |
| E3 Pair-Grounded, 3 epoch | 1.182629 | Chưa có | Chưa có | Chưa có | Valid thấp hơn V2.1; không gán kết quả pilot khác vào đây |
| Q-Former + fixed re-ranking 0.3/0.4/0.3 | Chưa xác minh | **1.221815** | 0.369356 | **3.48%** | Mốc pipeline CIDEr mạnh nhất đã ghi; là cải thiện hậu xử lý |

E2 V1 cao hơn continuation control trên valid **0.017180 CIDEr**. Trên test, E2 V1 cao hơn Q-Former epoch 10 **0.006501 CIDEr**. Tuy nhiên chưa có test của continuation control trong các bằng chứng đã kiểm tra, nên không xem test đó là đối chứng hoàn chỉnh cùng ngân sách train.

Re-ranking tăng **0.033570 CIDEr** so với Q-Former rank 0, đồng thời BLEU-4 giảm **0.001310**. Trọng số cố định được tài liệu ghi là lấy từ validation của Prompt-Conditioned Q-Former rồi áp dụng sang Q-Former cũ, không chọn lại trên test. Đây là mốc hậu xử lý cần giữ, không phải đóng góp kiến trúc mới.

E3 thấp hơn E2 V2.1 trên valid **0.006834 CIDEr**. Không so trực tiếp CIDEr valid của E3 với CIDEr test của E2 để tuyên bố tăng/giảm.

## Kiểm tra lại artifact 5.000 ảnh

Đã đọc `results.zip` người dùng cung cấp, không thực thi mã bên trong. Hai bộ CHAIR của Q-Former và E2 V1 có đúng cùng **5.000 ID**, caption đầu vào khớp caption được chấm, và tập nhãn đối tượng GT theo ảnh giống nhau. Tính lại các tổng từ các bản ghi từng caption cho kết quả:

| Tổng kiểm tra | Q-Former | E2 V1 |
|---|---:|---:|
| Caption có hallucination | 216 | 194 |
| Object mentions được nhận diện | 7.327 | 7.396 |
| Object mentions hallucinated | 222 | 197 |
| Object GT được nhắc đúng | 6.889 | 6.990 |
| Tổng object GT dùng tính recall | 15.746 | 15.746 |

Có **101** ảnh chuyển từ hallucinated sang clean và **79** ảnh chuyển theo chiều ngược lại, giảm ròng **22** caption hallucinated. CHAIRi và Recall tái tổng hợp cũng khớp kết quả đã lưu.

Bootstrap ghép cặp theo ảnh, 20.000 lần, seed 42, cho khoảng percentile 95% của `CHAIRs(E2) − CHAIRs(Q-Former)` là **[-0.0096, 0.0010]**. Khoảng này chứa 0: có cải thiện số học, chưa đủ kết luận chắc về khác biệt thống kê từ phép kiểm tra này. Không có bootstrap CIDEr vì chưa có đầy đủ reference/candidate bundle tương ứng tại máy để tính lại.

Đây là tái tổng hợp output CHAIR đã có, không phải chạy lại toàn bộ scorer với annotation gốc. Các CIDEr/BLEU/METEOR trong bảng vẫn được đánh dấu là **đã báo cáo**, chưa tái tính trong lần kiểm tra này. Với fixed re-ranking, tài liệu và sheet khớp nhau nhưng bundle caption sau re-ranking chưa có trong các artifact local đã kiểm tra. Không nâng trạng thái đó thành “đã tái lập”.

## Mốc làm việc từ đây

- So sánh kiến trúc: giữ Q-Former, continuation cùng ngân sách và E2; tách từng split, checkpoint, seed và decoding.
- So sánh pipeline cuối: phải có **Q-Former + fixed re-ranking**, cùng ngân sách candidates/decoding và cùng hậu xử lý cho các kiến trúc được so sánh. Không chỉ thắng Q-Former rank 0 rồi kết luận thắng pipeline tốt nhất.
- Giữ E2 V2.1 như một mốc trade-off caption/faithfulness; không gọi nó là tốt nhất ở mọi metric.
- Dừng chuỗi head nhỏ đã không qua đối chứng. Mô hình HOI huấn luyện sẵn được đánh giá riêng ở [PRETRAINED_HOI_TRANSFER_RESULT.md](PRETRAINED_HOI_TRANSFER_RESULT.md).
- Không tuyên bố đã có một mô hình caption mới tốt hơn, hay đóng góp bài báo đã được xác nhận, từ một audit quan hệ nhỏ.

## Bằng chứng và tái lập

Nguồn: `QFORMER_OBJECT_ALIGNMENT.md`, `QFORMER_CASCADE_ALIGNMENT.md`, `EVALUATE_QFORMER_TEST.md`, `EVALUATE_CASCADE_CHAIR.md`, `QFORMER_FIXED_RERANKING_ABLATION.md`, log V2.1/E3 người dùng đã cung cấp và snapshot registry `audit/caption_metrics_registry_20261015.json`.

Kết quả có hash nguồn: `audit/caption_progress_reconciliation_20261015.json`. Script tái kiểm tra:

```powershell
python audit_caption_progress.py --archive C:/Users/leduc/Downloads/results.zip --output C:/KhoaLuanTotNghiep/Code/output/caption_progress_audit_20261015
```

Không chỉnh sửa báo cáo tuần DOCX/PDF. Không chạy lại caption model trong audit này.

## Bổ sung: full validation UPT đã hoàn tất

[Artifact validation mới đã audit](HOI_CAPTION_VALIDATION_RESULT.md) tái lập Q-Former rank 0 CIDEr 1.174011504; fixed re-ranking 1.210722882; thêm UPT λ=.1 đạt 1.212580140. CHAIRs tương ứng 4.28% / 3.56% / 3.58%. UPT chỉ đổi 43 caption so với fixed; tăng CIDEr nhỏ và thêm ròng một caption hallucinated. Đây là validation chọn λ, không thay thế mốc test lịch sử và chưa có test mới. Các hạn chế tái kiểm tra nêu ở phần lịch sử phía trên được giữ nguyên cho artifact lịch sử; bundle validation mới đã tính lại CIDEr/BLEU/ROUGE.
