# GV — kiểm tra đường gradient phrase phụ

**Theo tiêu chí khóa trước: dừng VG region-CE với cache và quy mô hiện tại. GV chạy hợp lệ nhưng chưa vượt A hoặc G.**

| Nhánh | CIDEr ↑ | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Caption ảo giác |
|---|---:|---:|---:|---:|---:|
| original | 1.189066 | 5.80% | 3.545% | 45.681% | 58/1000 |
| A | 1.205513 | 5.20% | 3.248% | 45.804% | 52/1000 |
| G | 1.208401 | 5.30% | 3.307% | 45.619% | 53/1000 |
| T | 1.185747 | 5.20% | 3.264% | 45.527% | 52/1000 |
| GV | 1.193220 | 5.30% | 3.303% | 45.865% | 53/1000 |

CIDEr giữ thang evaluator, không nhân100. A là caption CE/SCST đối chứng; GV chỉ bỏ gradient phụ trực tiếp vào tham số decoder.

| So sánh | Δ CIDEr | Paired95% CI |
|---|---:|---|
| GV_minus_A | -0.012292 | [-0.032404, +0.007695] |
| GV_minus_G | -0.015181 | [-0.035035, +0.003657] |
| GV_minus_T | +0.007473 | [-0.014702, +0.029766] |

Bootstrap20000lần,seed42,1000ảnh ghép theoID; dev đã dùng nhiều lần, một seed, không phải xác nhận test độc lập.

GV−A −.012292, GV−G −.015181, cả hai CI chứa0. Không có gain đủ tiêu chí, nhưng cũng chưa chứng minh GV chắc chắn kém hơn A/G. Recall nhỉnh hơn A (1492vs1490đối tượng unique), trong khi CIDEr thấp hơn và caption ảo giác53vs52, không đủ coi cấu hình thành công.

Giả thuyết gradient phrase vào decoder là nguyên nhân chính làm mất lợi ích chưa được ủng hộ bởi phép thử này. Không suy ra mọi giám sát vùng hoặc mọi hướng căn chỉnh bất khả thi. Theo thỏa thuận trước: không full train, không tự thử freeze/trọng số/seed khác để cứu chuỗi này; báo Astra chọn hướng tiếp từ tóm tắt.

## Tính hợp lệ và giới hạn cơ chế

Hai rank đủ1250steps, lịchCE/SCST+.1phrase đúng và loss hữu hạn. Cùng init/cohort/stage hash/payload/schedule/targets/optimizer/runtime/original predictions/evaluator với G; mọi source gốc G giữ nguyên, chỉ thêm4sourceGV. Mọi rank-step ghi aux_decoder_gradient_enabled=false; synthetic tests thực sự qua trong log: aux loss/visual gradient khớpG và decoder.grad=None; decoder gradient combinedCE/SCST khớpcaption-only; current weights/parameter restoration;2GPU DDP smoke. Đây là test synthetic và source audit, không đo tách gradient trên từng minibatch thật.

Caveat: global gradient clipping norm1 được giữ như G, có 2500/2500rank-steps vượt ngưỡng. Dù decoder không nhận gradient phrase trực tiếp, loss phụ vẫn có thể đổi hệ số clipping của cập nhật decoder; visual memory đổi cũng ảnh hưởng nhiệm vụ caption. Do đó không diễn giải đây là decoder được cách ly hoàn toàn khỏi VG.

Aux10000draws,mean32.258patch; phraseCE mean100step đầu4.6605→100step cuối4.3441, trên minibatch khác nhau, không heldoutmetric. Training≈9.23phút, chưa gồm staging/IDF/eval.

Không chạy model, inference hay training mới trên local. CHAIR không chấm quan hệ; chưa human visual review. Checkpoint không có trong reviewZIP, hashes do worker báo cáo.
