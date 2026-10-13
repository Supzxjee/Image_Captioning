# Kết quả diagnostic V2: loss cân bằng chưa cải thiện tổng thể

Nguồn: instance_contrast_alignment_v2_diagnostic_1024_review.zip; code c8c303f,
workflow c23d3aa; seed42; CLIP frozen; GT boxes/classes; head20epochs.
Kaggle version không có trong manifest. Các kết quả là development diagnostic,
không official V-COCO role AP, không caption/detector performance.

## Phân bố và kiểm tra dữ liệu

1024 train pairs/685 ảnh;256 dev pairs/212 ảnh. Positive-containing pairs:
515/1024 (50.29%) và128/256 (50%). Lỗi positive-first fill V1 đã hết;
mẫu vẫn curated/enriched, không phản ánh prevalence tự nhiên.
Train151 actor edges,2385 predicate edges; dev40 actor edges/32 ảnh và604 predicate
edges/126 ảnh. Train/dev image overlap0; tất cả caption train/restval. File hashes
khớp pair_summary và predictions khớp thứ tự instance IDs của dev.

## So sánh trên cùng mẫu V2

| Phương pháp | Macro AP ×100 | Actor wins | Predicate wins | Mean actor gap |
|---|---:|---:|---:|---:|
| Frozen CLIP union cosine | 25.76 | 34/40 | 461/604 | 0.02153 |
| Supervised BCE | 54.44 | 29/40 | 549/604 | 0.00813 |
| BCE + edge-mean contrast | 55.44 | 27/40 | 555/604 | 0.01034 |
| BCE + type-balanced contrast | 54.61 | 29/40 | 553/604 | 0.01239 |

Type-balanced tăng2 actor wins so với edge-mean, nhưng không tăng so với BCE và
vẫn thấp hơn cosine5wins. AP giảm0.83điểm so với edge-mean. Chưa có lợi ích tổng thể
hoặc bằng chứng ổn định/ý nghĩa thống kê. Không so AP54.61 với V1 69.53 như cùng
benchmark: prevalence, support và sample đã đổi.

So sánh từng actor edge của balanced với cosine:26cùng đúng,3cùng sai,3được sửa,
8bị làm sai. So với edge-mean:27cùng đúng,11cùng sai,2được sửa,0bị làm sai.
Actor edges chia theo lớp: holding23,sitting_on4,carrying7,eating1,talking_on5;
không có riding actor edge.40edges không phải40ảnh độc lập và chưa chứng minh đủ
các loại relation. Không actor edge nào đạt margin0.2 trong cả4mốc.

## Chất lượng từng loại quan hệ và abstention

AP balanced ×100: holding73.01,riding93.79,sitting_on64.71,carrying11.99,
eating60.95,talking_on23.20. Positive dev support:98,11,21,18,11,13.
Carrying còn kém cosine27.26. Carrying/talking_on đều abstain toàn bộ known dev.
Holding precision87.5% nhưng recall7.14%; riding100%/81.82%; sitting_on75%/28.57%;
eating100%/27.27%. Đây là threshold fit từ train, không calibrated confidence.

Loss train0.70135→0.26631, ranking0.19989→0.11957; loss giảm không chứng minh
instance binding đạt yêu cầu. Main score vẫn dựa trên MLP của global crop embeddings.

## Quyết định

Giữ kết quả supervised BCE làm bằng chứng khả năng học action semantics từ nhãn thật.
Không tuyên bố type-balanced tốt hơn và không nối scorer hiện tại vào Decoder/full
prompt cache. Không chạy thêm chỉ để tăng epoch hoặc weight của cùng loss.

Giả thuyết kiến trúc cần xem xét: head học lại toàn bộ vector căn chỉnh có thể làm mất
thứ tự instance vốn có trong CLIP. Baseline mạnh hơn ở actor cho động lực giữ score
CLIP gốc và học residual hiệu chỉnh, hoặc dùng tương tác đặc trưng vùng chi tiết hơn.
Đây là giả thuyết, không kết luận causal. Cần thiết kế/duyệt bản tiếp rồi mới chạy.
Chưa triển khai residual, patch attention, detector boxes hoặc caption fusion.
