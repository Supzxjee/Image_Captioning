# Residual và bước căn giữa: ablation2×2,3seed,CPU

Chạy feature V2 cố định1024train/256dev,seed42/7/123,20epochs,scale0.1.
Thiết kế4biến thể đã ghi run_plan trước train; không chọn seed/epoch/hyperparameter
theo dev trong lượt này. Mẫu development này đã được xem nhiều lần, không heldout.

## Kết quả trên cùng ảnh/nhãn/features

| Kiến trúc | Căn giữa loss | Mean macro AP×100 | SD qua seed (điểm AP) | Actor wins theo seed /40 |
|---|---|---:|---:|---|
| Head học lại vector | Không | 54.28 | 0.62 | 29,30,31 |
| Head học lại vector | Có | 56.38 | 3.46 | 30,31,31 |
| CLIP+bounded residual | Không | 25.76 | 0.00 | 34,34,34 |
| CLIP+bounded residual | Có | 72.80 | 0.66 | 36,36,36 |

Frozen CLIP baseline AP25.76,actor34/40. SD là biến động qua3seed trên cùng mẫu,
không uncertainty do lấy ảnh và không confidence interval.40actor edges chỉ32ảnh.
Seed42 residual+center tái lập đúng AP của R50(73.215555%). Head raw seed42
tái lập AP R49(54.606991%). Cả3seed residual+center giữ34actor edges CLIP đã đúng,
sửa2và vẫn sai4. Không chứng minh giữ mọi case CLIP đúng trên ảnh mới.

## Diễn giải

Shared train-only center lấy từ train cosine/positive prior và chỉ dùng trong BCE.
Nó không đổi ranking nếu chỉ trừ cùng hằng số khi inference; lợi ích quan sát ở đây
là thay đổi quá trình học. Unknown labels bỏ khỏi loss, dev labels không fit center.
Ranking vẫn BCE+0.2type-balanced contrast, cùng margin/optimizer/head epochs.

Residual không center đẩy correction gần-0.1 cho hầu hết claim và AP/ranking cuối
không cải thiện so với CLIP. Raw cosine phần lớn dương, BCE quanh mốc0 gây áp lực
negative khó thỏa với correction bị chặn. Đây là lời giải thích phù hợp với diễn biến
đã thấy, không phép chứng minh causal đầy đủ.

Chỉ center cho head cũ không đạt mức residual+center. Kết quả 2×2 hỗ trợ việc giữ
CLIP prior kết hợp objective đúng mốc hơn là gán toàn bộ cải thiện cho một thành phần.
Không coi residual hoặc shared centering tự nó là novelty đã xác nhận.

## Giới hạn và quyết định

Giữ residual+center làm ứng viên kiến trúc. Không tiếp tục tối ưu trên cùng dev để
đẩy AP lên. Cần ảnh development/evaluation chưa dùng để thiết kế phương pháp, kiểm tra
negative/positive coverage và precision-recall theo lớp trước khi nối caption model.
Wearing,wrong-object labels,detector boxes,candidate caption/reranking chưa triển khai.
Carrying/talking_on vẫn có vấn đề abstention ởR50; AP không bảo đảm prompt đủ coverage.

14tests PASS, gồm residual khởi tạo/bound/gradient và center đổi BCE nhưng không
đổi contrast ranking. Đã chạy12lượt CPU; không tải dữ liệu hoặc CLIP lại.
Artifacts gồm run_plan,summary,predictions từngseed/biến thể và head checkpoints.
Review ZIP gọn chỉ chứa JSON results/predictions; checkpoints ở thư mục output riêng.

## Tái lập

```python
import subprocess, sys
subprocess.run([sys.executable, 'run_residual_center_ablation.py',
    '--source', 'PATH_TO/instance_contrast_alignment_v2_diagnostic_1024_review.zip',
    '--output', 'PATH_TO/residual_center_ablation_3seed',
    '--seeds', '42', '7', '123', '--epochs', '20', '--scale', '0.1'], check=True)
```
