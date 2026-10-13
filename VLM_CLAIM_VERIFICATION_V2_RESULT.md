# Kết quả smoke 12: không mở rộng same-model claim verification

Nguồn: `vlm_claim_verification_v2_smoke_12.zip`, inference commit `bf5825a`,
workflow `97d665e`, Qwen2.5-VL-3B revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`, hai GPU workers.
Kaggle code version không được ghi trong manifest.

## Kết quả thực tế

12 pairs/12 claims; 8 parse errors, 3 verdict unclear, 1 contradicted, 0 eligible.
8 parse errors đều do `subject_identity` là tên lớp (`person`) thay vì trạng thái;
`object_identity` cũng là tên đồ vật. Không chuyển tên lớp thành supported bằng suy đoán.
Raw verdict của 8 record đó là supported. Không báo 0/12 là lọc sạch hallucination.
Summary `variant` vẫn ghi v1 do tên cố định trong merge; manifest task là
`verify_claims_v2_smoke12` mới xác định phiên bản prompt.

Original pair indices giữ trong records:

| Index | Claim | Quan sát từ raw answer |
|---|---|---|
| 4 | holding tie | Vẫn khẳng định cầm tie; lỗi đã thấy là cầm hot dog |
| 6 | holding skateboard | Vẫn khẳng định tay nắm board; lỗi đã thấy là board nằm dưới đất |
| 55 | sitting_on skis | Vẫn khẳng định ngồi trên skis; ảnh cũ cho thấy ngồi ghế cáp treo |
| 31 | holding frisbee | Contradicted, nhận ra đồ vật được người khác cầm; một ví dụ có ích |
| 8,85 | holding glove/tie | Unclear nhưng direct_evidence chép nguyên placeholder của prompt |
| 9,64,67,68 | wearing tie / sitting_on surfboard | Raw supported, bị loại vì schema; không phải semantic rejection |

Local parser tests đảm bảo strict gating, không chứng minh model tuân thủ schema hoặc
nhìn đúng ảnh. Prompt/schema do chúng ta thiết kế vẫn chưa ổn. Ngay cả khi sửa parser,
raw answers cho thấy các lỗi semantic quan trọng chưa được giải quyết.

## Quyết định

Không chạy lại 100 cặp hoặc tạo full cache bằng giao thức này. Không đưa các claim
vào pseudo-ground-truth. Không coi kết quả này là bằng chứng mọi VLM đều không dùng được:
chỉ cấu hình/model/prompt hiện tại chưa cho thấy verifier đáng tin cậy.

Bước phát triển tiếp cần supervision quan hệ theo cặp instance: xây dựng nhãn đã kiểm
chứng cho cả positive, wrong-predicate và wrong-instance negatives; tách ảnh train/dev/eval;
dùng appearance/region features của từng instance và union để học scorer quan hệ.
Teacher chỉ đề xuất để review. Đánh giá precision cùng coverage trước khi nối vào
semantic prompt/fusion và caption training. Chưa lựa chọn dataset hay kiến trúc cuối cùng.
