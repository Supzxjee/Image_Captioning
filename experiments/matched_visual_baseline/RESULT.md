# Baseline cùng decoder: hoàn tất CE10, VALID5000

22/11/2026. Người dùng xác nhận nguồn `ducanh3/notebook7bc83c4c76/188`. [Số liệu và audit](RESULT.json), [quyết định trước chạy](PROTOCOL.md). Không chạy mô hình hoặc scorer local; không sửa báo cáo Word.

## Kết quả chính

Baseline hoàn tất113287TRAIN/10epochs/177110updates/5667470caption events. Seed42,global32,FP32,Accelerate/DDP haiGPU, không sharing. Best và minVALID NLL đều ở epoch9; giữ best.bin, không dùng checkpoint cuối epoch10 thay cho best. Không TEST/SCST/rerank trong lượt này.

So với full model seed42 đã có, cả hai cùng decoder ba lớp/dropout0.2, lịch CE10 và chọn best9. Mỗi cột dưới đây là beam top1, không rerank. Điểm chất lượng và CHAIR/recall nhân100; chênh lệch CHAIR tính bằng điểm phần trăm.

| Thang đo | Baseline thị giác | Bank + prompt | Full − baseline |
|---|---:|---:|---:|
| BLEU-1 | 77,87 | 77,92 | +0,05 |
| BLEU-4 | 37,89 | 37,94 | +0,05 |
| ROUGE-L | 57,90 | 57,83 | −0,07 |
| CIDEr | 119,14 | 120,30 | +1,16 |
| CHAIRs (%) | 5,70 | 4,00 | −1,70 |
| CHAIRi (%) | 3,99 | 2,79 | −1,20 |
| Recall (%) | 42,64 | 42,70 | +0,06 |
| NLL | 2,19418 | 2,16421 | −0,02997 |
| Trainable parameters | 29.553.424 | 53.183.736 | +23.630.312 |

METEOR/SPICE chưa được chấm ở lượt baseline này. Không điền kết quả TEST của full model vào bảng VALID.

Full model giảm số caption có đối tượng hallucination285→200 (giảm ròng85caption, tương đối29,8% CHAIRs), lượt nhắc đối tượng hallucination293→203. Recall6791/15928→6801/15928, độ dài trung bình9,183→9,2932 từ. Vì thế lợi ích CHAIR quan sát không đi kèm caption ngắn hơn hoặc sụt recall tổng. Không suy ra mọi caption đều tốt hơn hoặc mọi quan hệ đều đúng; CHAIR chỉ chấm đối tượng.

## Kiểm tra thực chạy

- Hai rank xác nhận77tensor chung khởi tạo bằng constructor full cùng seed; đúng decoder, hai memory cùngV, prompt không ảnh hưởng và vẫn phụ thuộc ảnh. Baseline29.553.424/full53.183.736tham số được runtime đếm. Constructor full dùng dummy bank cho khởi tạo, không nạp checkpoint đã học.
- CUDA/NCCL haiGPU,Accelerate1.15.0,FP32/accum1; causal/beam rescoring và gradient/DDP first32/tail27 PASS.64benchmarkupdates bỏ đi, main worker khởi tạo lại.
- ZIP CRC/SHA,134files,41source payload SHA khớp notebook đã giao.10VALIDepochs/50.000captionrows, cohort5000IDs cố định, per-image CIDEr means, NLL theo token, mean words và mọi evaluation hash kiểm tra đạt.10.000caption CHAIR best/end khớp câu đã sinh; các số đếm tái tổng hợp khớp. Training log SHA khớp receipt,1780dòng ghi tiến độ đến177110.
- Worker19.496,71giây≈5giờ25phút, gồm huấn luyện/VALID/checkpoint trong worker; chưa gồm setup và CPU quality. Metadata preparation14,02giây, không chép full patch cache. Không phải phép đo giờ GPU tính quota hoặc so tốc độ có kiểm soát với lượt cũ.

## Diễn giải và giới hạn

Đối chứng mới giảm được confound decoder của baseline bốn lớp trước đây. Kết quả một seed trên VALID ủng hộ **toàn bộ nhánh bank + prompt**, với tăng CIDEr nhỏ và giảm ảo giác đối tượng rõ hơn về số lượng quan sát. Tuy nhiên full model thêm23,63triệu tham số; chưa tách bank/M2/M3 hoặc giá trị riêng embedding chữ. Không tuyên bố significance khi chưa kiểm định, không thay biến thiên theo ảnh cho biến thiên huấn luyện.

Lượt full seed42 lịch sử dùng localFP16 staging, baseline mới đọc trực tiếp cùngrawH5; cùngpatch/token convention và intendedrecipe nhưng không phải bitwise replay toàn bộ training. Khởi tạo chung được kiểm với constructor full tái dựng, chưa đối chiếu một binary khởi tạo lịch sử. VALID đã tái sử dụng để phát triển và chọn checkpoint, chưa phải kết luận TEST độc lập.

Review ZIP không chứa trọng số nên chỉ kiểm receipt, chưa rehash/load binary thực tế. Best epoch9 SHA `1eade1fca3b872091b2f2a345a04d8141a887d3fcd1e5772e4952a495c5f4e6b`; cuối epoch10 SHA `ab38aaf76c854fe84242925ddd262c1c7f2d034decf295ed4e76b03262aee2ce`. Giữ cả hai trong outputv188.

## Trạng thái và bước kế tiếp

Sheet `Kết quả mới!A273:D289` đã ghi và đọc lại đúng số liệu, metadata định dạng kiểm tra; chưa kiểm hình native. Nguồn full seed42 là checkpoint best9 đã dùng trong kết quả cũ, không sinh lại câu.

Khuyến nghị kế tiếp: chấm TEST5000 cho chính best baselinev188 rồi so với full seed42 trên cùngcohort/beam/scorers. Chưa triển khai hoặc chạy TEST trong lần nhận ZIP này; chưa tự mở rộng seed/GPT-2/SCST.
