# VG region-caption — đối chứng A/G/T hoàn tất

**Quyết định: chưa mở rộng cấu hình G. Chỉ định vùng tốt hơn đối chứng cùng phrase toàn ảnh T, nhưng chưa cải thiện đáng tin cậy so với A.**

| Nhánh | CIDEr ↑ | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Caption ảo giác |
|---|---:|---:|---:|---:|---:|
| original | 1.189066 | 5.80% | 3.545% | 45.681% | 58/1000 |
| A | 1.205513 | 5.20% | 3.248% | 45.804% | 52/1000 |
| G | 1.208401 | 5.30% | 3.307% | 45.619% | 53/1000 |
| T | 1.185747 | 5.20% | 3.264% | 45.527% | 52/1000 |

CIDEr giữ thang evaluator, không nhân100. Original là checkpoint đầu; A cùng lịch caption CE/SCST không có phrase phụ, G thêm phrase phụ theo vùng, T thêm đúng phrase đó nhưng đọc toàn ảnh.

| So sánh | Δ CIDEr | Paired95% CI |
|---|---:|---|
| G_minus_A | +0.002888 | [-0.017607, +0.023641] |
| G_minus_T | +0.022654 | [+0.001460, +0.044563] |
| T_minus_A | -0.019766 | [-0.039012, -0.001215] |
| G_minus_original | +0.019335 | [+0.000363, +0.038446] |
| T_minus_original | -0.003319 | [-0.025459, +0.018263] |

Bootstrap20000lần,seed42,ghép theo1000imageIDs development đã dùng nhiều lần; không phải test độc lập hay nhiều seed.

## Kết luận có thể rút ra

G−T +.022654 có CI dương trong phép resampling ảnh này, nên chỉ định vùng có tín hiệu hữu ích so với dùng cùng phrase với toàn ảnh. Nhưng G−A chỉ+.002888, CI chứa0 và CHAIRs/i tăng nhẹ, nên chưa đạt gate nâng chất lượng baseline. T−A −.019766 với CI âm: một phần khoảng cách G/T đến từ đối chứng T giảm chất lượng. Không diễn giải G>T thành G tốt hơn baseline hoặc đã chứng minh grounding đúng.

Giả thuyết cần trao đổi với Astra: chỉ định vùng có thể giảm sự không khớp giữa phrase cục bộ và ảnh toàn cảnh; nguồn VG yếu hoặc mục tiêu phụ cũng có thể gây nhiễu. Đây là giả thuyết cơ chế, chưa kiểm chứng bằng visual review hoặc ablation khác. Không tự tăng dữ liệu/tune/full train để cứu kết quả.

## Audit

G/T protocol chỉ khác arm,output,stage path; source runtime hoàn toàn giống nhau, cùng payload/lịch phrase,stage hash,caption cohort,targets,optimizer,runtime và original predictions. Hai rank T đủ1250updates,tham số như A/G,loss hữu hạn đúng công thức; T dùng196patch ở mọi rank-step, G trung bình32.258patch; mean số patch vùng gốc khớp từng step G/T. Không thêm tham số hoặc dùng vùng VG khi inference caption. Synthetic CPU/model/DDP preflight qua trong log.

Training G≈8.02phút, T≈8.31phút; không gồm staging,khởi tạoIDF/evaluation. G/T khác349/1000caption, chưa chấm đúng/sai bằng ảnh.

Phrase loss G: 4.1717→3.2025; T: 4.5791→3.4437 (mean100step đầu/cuối trên minibatch khác nhau, không heldoutmetric). Loss thấp hơn không thay thế gain caption hoặc bằng chứng grounding.

Đã chỉ xử lý saved artifacts trên CPU; không chạy model, sinh caption, hay huấn luyện mới. CHAIR không chấm quan hệ. Checkpoint không nằm trong reviewZIP; hash do worker báo cáo. Chờ Astra quyết định hướng tiếp theo từ tóm tắt, không yêu cầu đọc lại rawfiles.
