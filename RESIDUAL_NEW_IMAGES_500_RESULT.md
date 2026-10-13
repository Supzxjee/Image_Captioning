# Frozen residual: kết quả trên 500 cặp từ 388 ảnh mới

Nguồn feature: residual_new_images_500_features_review.zip, extraction910fda9;
checkpoint R50 seed42/20epochs/scale0.1 đã khóa bằng SHA256. Không huấn luyện lại,
không chọn seed mới, không fit threshold. CPU inference bằng model eval/no gradients.
Không có giao ảnh với1000IDs trong manifest diagnostic/audit trước.

"Mới" chỉ đối với scorer experiments/audits được liệt kê. GT boxes/classes,
caption train/restval, curated pairs; không caption-heldout hoặc official HOI evaluation.
Không biết overlap với CLIP pretraining. Đây là lần đầu đọc predictions của cohort này.

## Chất lượng trên nhãn đã biết

| Phương pháp | Macro AP×100 | Actor wins | Predicate wins |
|---|---:|---:|---:|
| Frozen CLIP union cosine | 24.01 | 57/76 | 868/1149 |
| CLIP prior+bounded residual | 70.01 | 64/76 | 1109/1149 |

Actor edges thuộc61ảnh; predicate edges thuộc240ảnh, không phải các observations
độc lập hoàn toàn. Actor improvement: giữ57cặp CLIP đã đúng,sửa7cặp trên7ảnh,
vẫn sai12,làm sai0. Predicate:giữ851,sửa258,làm sai17,vẫn sai23.
Mean actor gap tăng0.01532→0.03104, nhưng margin0.2 vẫn không actor edge nào đạt.

| Relation | Positive support | CLIP AP×100 | Residual AP×100 | Precision/recall ở threshold khóa |
|---|---:|---:|---:|---|
| holding |185|69.74|70.15|100%/4.86% (9known accepted) |
| riding |21|9.25|90.09|89.47%/80.95% (19accepted) |
| sitting_on |46|13.80|76.16|71.43%/21.74% (14accepted) |
| carrying |32|24.17|52.31|abstain |
| eating |30|21.94|77.65|82.35%/46.67% (17accepted) |
| talking_on |15|5.17|53.67|abstain |

Known labels theo thứ tự373,488,365,489,451,483; unknown không thành negative.
Macro AP70.01 không phải accuracy/precision70.01. Tín hiệu trên ảnh mới ủng hộ
scorer có ích hơn CLIP trực tiếp trong phạm vi này, chưa chứng minh caption improvement
hoặc novelty. Không thay đổi protocol rồi tái báo cohort này như evaluation mới.

## Coverage và tính đúng sai

Residual nhận71claims:59known và12unknown. Trong59known có50positive/9negative,
micro precision84.75%,micro recall15.20% của329known positive claims.
12unknown phải review, không được mặc định đúng hoặc sai. Nguồn nhãn không đầy đủ
cho mọi pair link. CLIP nhận12knownclaims,9positive. Không dùng accepted count
như sốcaption hoặc sốảnh có quan hệ đúng.

Carry/talking thresholds2.0 được khóa từ R50 do train chưa đạt quy tắc nhận,
nên hai lớp này vẫn bị abstain bất chấp AP cải thiện. Holding recall còn thấp;
sitting_on precision thấp hơn mục tiêu90% của train. Chưa đủ điều kiện tự tạo full
prompt cache từ mọi lớp hoặc khẳng định đáp ứng precision90% trên ảnh mới.

## Checksum Windows/Linux

Lượt CPU đầu fail vì file exclusion trên Windows cóCRLF, còn Git blob/Kaggle cóLF.
JSON nội dung giống nhau. Sửa evaluator chỉ chấp nhận SHA của các biến thể LF/CRLF
của đúng bytes frozen manifest, không chấp nhận thay image IDs. Checkpoint hash và
pair JSON hash giữ kiểm tra byte-exact. Thêm regression test mutation IDs bị từ chối.
16tests PASS; không cần chạy lại GPU hoặc thay nội dung cohort/model/threshold.

## Bước tiếp hợp lý

Giữ residual+center làm scorer ứng viên, bảo toàn cohort này để đối chiếu, không tune
threshold trên labels vừa đánh giá. Trước caption fusion cần kiểm tra cặp từ detector
và nguồn bằng chứng ảnh; GT-box evaluation chưa đo lỗi detect/match instance.
Kết hợp chọn lọc quan hệ với object-only fallback và review các accepted unknown/false
claims. Chưa triển khai detector-box evaluation/caption candidates/fusion trong lượt này.
