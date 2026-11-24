# M2-only run1 v243 — chưa hoàn tất CE10

Nguồn: `ducanh2403/2-vi-cai-kia-qua-nhieu/243`, người dùng xác nhận ngày24/11/2026. [Số liệu và kiểm tra](RUN1_RESULT.json). Log cho thấy `TimeoutError: Saved before parent/worker cap` ở cả hai rank, không có bằng chứng lỗi CUDA/mô hình. Trọng số đã lưu trước khi dừng.

Đã hoàn tất8epoch, đang epoch9/cursor5616, update147304/177110 (83,17%), 4.713.688 caption events. Còn29.806update để đủ lịch10epoch. Đây không phải lượt10epoch hoàn tất; chưa so sánh ablation cuối hoặc chấm TEST từ best tạm.

| Epoch đã hoàn tất | CIDEr VALID | NLL VALID |
|---|---:|---:|
| 1 | 94,69 | 2,70418 |
| 2 | 107,70 | 2,43920 |
| 3 | 111,43 | 2,33858 |
| 4 | 114,38 | 2,27820 |
| 5 | 117,05 | 2,23708 |
| 6 | 117,68 | 2,20492 |
| 7 | 118,31 | 2,18535 |
| 8 | **119,34** | **2,17500** |

Guards haiGPU xác nhận49.239.288tham số, chỉ bỏ3.944.448tham số M3; bank/M2, decoder và tất cả166statevalues còn lại khởi tạo đúng constructorfull. Hai memory bằng đầu ra M2, prompt/ROI không ảnh hưởng, zero bank trên reference làm đổi memory (positivecontrol). Causal/beam/DDP first32 và tail27 đềuPASS. Đây là kiểm tra cơ chế, chưa chứng minh chất lượng caption hoặc hallucination.

ZIP115files, CRC và42sourceSHA khớp notebook đã giao; kiểm40.000caption ở8VALIDepoch, tất cả evaluationseals, trung bìnhCIDEr/NLL/meanwords, 1.480dòngtraininglog và hash log khớp checkpointreceipt. Bankstage chỉ haiNPY đúng seal, không promptstage. Không có CHAIR/METEOR/SPICE/TEST trong lượt này. Checkpoint lớn không có trong ZIP; SHA trọng số là receipt, sẽ hash binary khi tải resume trên Kaggle. Không chạy Torch/model/scorer local.

Latest SHA `16b76195ae52ebe150d87b81843906a862fc13903a005b031407388753475bab`; best8 SHA `a5d9dbab46e0f188d3db28b7caa66d2687abfbd625a17ddf4dc04d90642a4510`. Tiếp tục bằng latest v243, giữ optimizer/RNG/cursor/LR và lịch10epoch; không restart. [Protocol resume](RESUME_V243_PROTOCOL.md). Giữ toàn bộOutput; notebook resume dùng API exactv243, không cần Dataset checkpoint mới.
