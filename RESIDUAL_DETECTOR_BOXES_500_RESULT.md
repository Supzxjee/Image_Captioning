# Frozen residual với box detector: đánh giá conditional matched pairs

Feature source residual_detector_boxes_500_features_review.zip; extraction26de41a,
workflow117deac. CPU dùng đúng checkpoint R50seed42 và frozen thresholds; không train/refit.
GT dùng để match/labels, crop đầu vào dùng box detector từ cache.Confidence0.5,
sameclassIoU0.5,greedyone-to-one,NMS0.8; không đo đầy đủ detector-proposed HOI candidates.

## Coverage

364/500pairs(72.8%) trên286ảnh có cảA/B match.136pairs không match:
subject7,object127,both2. Cache issues rỗng: không thiếu filename records hoặc invalid box
được report. Không diễn giải mọi cặp mất là detector không thấy đồ vật: threshold,class,
IoU và matchinggreedy cùng ảnh hưởng. Extra detections không tự được tính falsepositive.

Known positive claims được giữ243/329(73.86%). Theo relation:
holding142/185,riding18/21,sitting_on28/46,carrying27/32,eating23/30,talking_on5/15.
Các lớp mất cặp không đồng đều; talking_on còn5positives nên AP/precision rất hạn chế.

## So sánh trên cùng364matchedpairs

| Feature boxes / scorer | Known-label macro AP×100 | Actor wins/60 | Predicate wins/842 |
|---|---:|---:|---:|
| GT / CLIP |25.50|45|645|
| Detector / CLIP |25.62|38|646|
| GT / residual |70.72|50|819|
| Detector / residual |72.14|48|818|

Actor60edges thuộc49ảnh,predicate842edges thuộc178ảnh. Chênh AP GT/detector nhỏ
không chứng minh detector tốt hơn GT. Không so72.14 với70.01 của toàn500 như cùngmẫu.

Trên detector boxes, residual giữ38actor claims CLIP đúng,sửa10,vẫn sai12,
làm sai0(bao gồm một CLIP tie trong nhóm chưa đúng). Mean actor gap0.01350→0.02744;
không actor edge đạtmargin0.2. Predicate:sửa184,làm sai12,giữ634,vẫn sai12.
GT→detector residual:46actor edges cùngđúng,8cùngsai,2được sửa,4bị làm sai;
net50→48. Scorer không hoàn toàn bất biến trước cropnoise.

## Threshold khóa và coverage quan hệ

AP residual×100:holding75.27,riding87.43,sitting_on80.12,carrying58.03,
eating82.50,talking_on49.49. Known-positive support142,18,28,27,23,5.
Precision/recall:holding81.82%/6.34%;riding93.75%/83.33%;sitting_on77.78%/25%;
eating92.31%/52.17%;carrying/talking_on abstain hoàn toàn.

60accepted claims =49known(43positive,6negative)+11unknown. Knownmicroprecision87.76%,
conditionalrecall43/243=17.70%. Nếu tính denominator329positives của reference500
thì thu hồi43/329=13.07%; đây vẫn làrecall restrictedcohort, không end-to-endHOI recall.
Không tính11unknown làđúng/sai. GTboxes trên cùngmatchedsubset cũng thu hồi43TP,
nhưng có48knownaccepted và5FP; boxdetector thêm1FP trong tập này.

## Diễn giải và bước tiếp

Residual giữ được tín hiệu hữu ích hơn CLIP trên detectorboxes đượcmatch, dù một số
actor quyết định kém hơn GTcrops. Coverage detector/object làbottleneck riêng; không
giải quyết bằng việc tăng trọng sốloss của scorer.

Có thể giữ scorer này để xây prompt chọnlọc cho từngdetectorinstance, object-only
fallback khi thiếuquan hệ, rồi chuẩn bị pilotfusion. Trước production cần audit tất cả
detector-proposed pairs, vì lượt này chỉ có cặp GT đượcthamchiếu vàmatch thành công;
chưa biết falseclaims trên extra/unknown proposals. Chưa caption quality/novelty proof.
Không chỉnh threshold trên cohort này rồi báo như evaluation độc lập.

Artifacts:frozen_evaluation_summary,detector predictions,GT comparison cùngsubset,
sourceZIP SHA256.4matching+16supervised/protocoltests đãpass trước lượt này;
CPUevaluation thành công với modelvàcohort checksumgiữ nguyên.
