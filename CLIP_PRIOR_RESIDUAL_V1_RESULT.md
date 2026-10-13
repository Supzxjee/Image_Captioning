# CLIP prior + residual V1: chạy CPU trên feature V2 đã lưu

Nguồn: instance_contrast_alignment_v2_diagnostic_1024_review.zip. Không tải COCO,
không trích CLIP lại, không dùng GPU.1024train/256dev,seed42,head20epochs,scale0.1
đã cố định trước khi đọc kết quả. Mẫu này là development set đã được xem nhiều lần.

## Phương pháp đã triển khai

score = frozen CLIP union/text cosine +0.1*tanh(learned correction).
MLP nhận ordered A,B,union; text targets frozen. Lớp output khởi tạo zero nên
score khởi đầu bằng CLIP chính xác, correction mỗi claim giới hạn trong[-0.1,0.1].
Giới hạn correction không bảo đảm giữ ranking CLIP sau training.

BCE dùng một hằng số centering từ train labels/scores để raw cosine không bị
coi là logit quanh0 một cách bất hợp lý. Center dùng chung mọi lớp/instance,
không đổi ranking hoặc AP. Ranking type-balanced và masked unknown giữ như V2.
Không fit center/scale/epochs bằng dev và không chọn best epoch bằng dev.
Center cũng là thay đổi so với headV2; chưa có factorial ablation để quy mọi
cải thiện chỉ cho residual architecture.

## Kết quả trên đúng cùng mẫu V2

| Phương pháp | Known-label macro AP ×100 | Actor wins/40 | Predicate wins/604 |
|---|---:|---:|---:|
| CLIP union cosine | 25.76 | 34 | 461 |
| Supervised BCE head | 54.44 | 29 | 549 |
| Head+edge-mean contrast | 55.44 | 27 | 555 |
| Head+type-balanced contrast | 54.61 | 29 | 553 |
| CLIP prior+bounded residual | 73.22 | 36 | 588 |

So với CLIP trên actor edges:34cùng đúng,2được sửa,4cùng sai,0bị làm sai.
Actor edges chỉ thuộc32ảnh và không có riding actor edge;2wins thêm không chứng minh
hiệu quả ổn định hay significance. Predicate:453cùng đúng,135được sửa,8cùng sai,
8bị làm sai. Residual không giữ nguyên mọi quyết định đúng của CLIP.
Mean actor gap0.03474 (CLIP0.02153); margin0.2 vẫn không actor edge nào đạt.

AP per class×100: holding69.19,riding100.00,sitting_on72.81,carrying52.34,
eating69.12,talking_on75.83. Dev positive support98,11,21,18,11,13.
Holding AP hơi thấp hơn CLIP69.65, không cải thiện đồng đều mọi lớp.
Macro AP73.22 không phải precision73.22, không official role AP/detection mAP.

Threshold train90% empirical precision vẫn dẫn đến abstain mọi known dev claim
carrying/talking_on. Holding nhận2claims/98positives; sitting_on precision75%.
Chưa sẵn sàng tạo fullprompt cache. Correction abs trên dev trung bình gần bound
(xem residual_summary), cần kiểm tra saturation/generalization trong thiết kế sau.

## Kết luận trong phạm vi dữ liệu

Giữ residual làm ứng viên kiến trúc vì một lượt diagnostic có tín hiệu tốt hơn head
học lại vector, cả action AP và actor ranking. Không chứng minh novelty, không caption
quality improvement. Chưa nối Decoder hoặc reranking. Đánh giá tiếp cần ảnh mới chưa
dùng để thiết kế phương pháp và lỗi theo instance, trước khi quyết định huấn luyện lớn.

## Tái lập trên CPU

Trong repository với torch/numpy:

```python
import subprocess, sys
subprocess.run([sys.executable, 'run_residual_alignment_cached.py',
    '--source', 'PATH_TO/instance_contrast_alignment_v2_diagnostic_1024_review.zip',
    '--output', 'PATH_TO/clip_prior_residual_v1', '--epochs', '20', '--scale', '0.1'], check=True)
```

ZIP đầu vào được đọc trực tiếp, không extract; tensor dùng weights_only=True.
Kiểm tra SHA256 splits, instance IDs, feature coverage và image overlap trước train.
13 tests PASS, gồm exact CLIP initialization, bound và gradient của residual output.
Các baselines dùng kết quả đã lưu từ đúng mẫu V2; không train lại baselines trên CPU.
