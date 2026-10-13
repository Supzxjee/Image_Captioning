# Căn chỉnh đối chứng theo cặp instance: bản triển khai đầu

## Phạm vi đã duyệt

Phát triển cơ chế chấm điểm claim A–relation–B có supervision và khả năng abstain,
làm nền cho fusion ngữ nghĩa, sinh nhiều caption và lựa chọn caption. Bản này chỉ
triển khai diagnostic của scorer; chưa thay E2/E3, chưa nối vào Decoder/reranker,
chưa được xem là phương pháp mới đã xác nhận.

CSA-Net tham chiếu: https://github.com/CrossmodalGroup/CSA-Net/blob/0cc570ba7c50ef08c46a21439fe73ef0a5df97ab/models/caption/csa_net.py
CSA dùng ngân hàng danh từ/động từ và attention để bổ sung đặc trưng vùng. Bản ta
chấm claim cụ thể theo cặp và học đối chứng theo instance. Các ý tưởng contrastive
learning/pair features cũng đã có trong lĩnh vực; cần literature review trước khi chốt novelty.

## Dữ liệu và nhãn

V-COCO chính thức: https://github.com/s-gupta/v-coco
Ghim revision 489cc4db74f2f10ab4b134f67da3874afbf245ab, dùng train và val, không dùng test.
Vai trò object/instrument liên kết bằng COCO annotation ID, đọc flattened array theo đúng cột.
Tham chiếu schema: https://github.com/s-gupta/v-coco/blob/489cc4db74f2f10ab4b134f67da3874afbf245ab/vsrl_utils.py

Sáu action-role: hold/obj, ride/instr, sit/instr, carry/obj, eat/obj,
talk_on_phone/instr; tương ứng holding, riding, sitting_on, carrying, eating, talking_on.
Không có wearing; cũng chưa giải quyết mọi loại object-object hoặc quan hệ không gian.
Đây là giới hạn diagnostic, không phải vocabulary cuối của đề tài.

- Positive: actor action=1 và role object đúng B có box dùng được.
- Negative: actor được gán action=0, nên claim hành động đó với B là negative theo annotation.
- Unknown=-1: actor chưa được annotate, role thiếu, hoặc actor action=1 nhưng B không phải
  object được annotate. Không suy đoán một B khác là negative.
- Crowd không dùng; cross-image role IDs và thiếu annotation fail rõ ràng.
- Chỉ giữ ảnh Karpathy train/restval. Caption val/test bị loại khỏi cả relation train và dev.
- Relation train/dev theo official V-COCO train/val, kiểm tra không trùng image ID.
- Diagnostic tối đa 512 train/128 dev pairs, khoảng nửa positive-containing pairs,
  ưu tiên giữ wrong-actor counterparts đã annotate; thống kê số thực tế trong pair_summary.
  Distribution diagnostic không đại diện full benchmark.

## Scorer và loss

CLIP ViT-B/16 frozen trích 3 vector 512 chiều: crop A, crop B, union A/B. Resize cả
crop về 224x224 để CLIP center crop không loại B; có biến dạng tỉ lệ, cần đánh giá sau.
Ngân hàng text frozen gồm 6 prompt theo object class của B.

Ordered concatenation [A,B,union] qua MLP tạo vector căn chỉnh; cosine với từng text
target tạo score riêng cho mỗi claim. Không dùng geometry để gán nhãn; không lấy
heuristic/VLM proposals làm ground truth. GT object classes/boxes được dùng trong diagnostic.

Loss = masked BCE(scores/0.1) + 0.2 * mean(relu(0.2 - positive_score + negative_score)).
Đối chứng gồm:

1. Sai predicate trên cùng cặp, nếu predicate đó có nhãn 0.
2. Sai actor với cùng B/cùng ảnh/cùng predicate, nếu actor thay thế có nhãn 0.

Wrong-object negatives chưa triển khai vì vắng role không đủ để khẳng định false.
Mask BCE bỏ mọi unknown; text encoder/targets không train. Đây là metric alignment
head, chưa phải cơ chế cross-attention/fusion hoàn chỉnh.

## Diagnostic

Hai GPU worker trích frozen features; đầu MLP full-batch nhỏ train CPU 20 epochs.
So sánh trên cùng pairs/features:

- CLIP union-text cosine không train.
- Supervised pair scorer, chỉ masked BCE.
- Cùng scorer/cùng khởi tạo, thêm contrast loss.

AP per relation chỉ trên known labels có cả positive lẫn negative; macro AP báo kèm
từng lớp và support. Không gọi đây là official V-COCO role AP hoặc detection mAP.
Threshold được fit chỉ từ train: >=90% empirical precision và >=5 known accepted.
Không đạt thì abstain. Đây chưa phải calibration hay đảm bảo precision trên dev.
Precision/recall dev có thể null khi không có positives/accepts; không chuyển null thành 0.
Chưa đo caption CIDEr/CHAIR vì chưa nối vào caption model.

Điều kiện để tiếp tục: kiểm tra class/edge coverage; scorer học được; dev có cải thiện
hợp lý so với hai mốc; xem lỗi đúng/sai instance. Diagnostic nhỏ chưa chứng minh hiệu quả
hay tính mới. Sau đó mới làm full relation training, detector boxes, fusion và caption candidates.
