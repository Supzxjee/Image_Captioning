# Kết quả C/T — 26/10/2026

**Đối chứng đạt kiểm tra; recipe không đạt ngưỡng tiếp tục. Giữ A.**

Hai nhánh chạy đủ1.250updates, cùngseed/order,TRAIN10k/dev1000 và cache2.000aux.
Dự đoán A, per-imageCIDEr và toàn bộ output CHAIR của hai lượt trùng hoàn toàn.
14runtime source hashes/arm hợp lệ. Protocol chỉ khác tênarm,output/stagepath và
thời gian staging; stageSHA,cohort,rows,vocabulary,policy và cache receipt trùng.
Worker receipts xác nhận frozen weights/anchor giữ nguyên và tensor cuối đồng nhất
hai rank.50training records/arm đầy đủ, loss/gradient hữu hạn. Không chạy model local.

| Nhánh | CIDEr | CHAIRs | CHAIRi | Object recall | Độ dài TB |
|---|---:|---:|---:|---:|---:|
| A |1.205513|5.20% (52/1000)|3.2475% (53/1632)|45.8039% (1490/3253)|10.213|
| C |1.186127|5.50% (55/1000)|3.4335% (56/1631)|45.6809% (1486/3253)|10.265|
| T |1.178205|4.40% (44/1000)|2.8178% (45/1597)|45.0661% (1466/3253)|10.305|

## Chênh lệch CIDEr theo cặp ảnh

10.000paired image bootstrap resamples,seed42,cùng1.000dev images.

| So sánh | Trung bình | 95% CI | Caption thay đổi |
|---|---:|---|---:|
| C−A |−0.019385|[−0.036395,−0.002481]|291|
| T−A |−0.027307|[−0.048591,−0.005977]|374|
| T−C |−0.007922|[−0.024298,+0.008236]|243|

C lẫn T đều dưới A trên lượtdev này. CI T−C cắt0: chưa đủ bằng chứng quy toàn bộ
mức giảm cho stale term; recipe C chung đã làm giảm CIDEr so với A.
Không suy ra mọi cơ chế căn chỉnh đều thất bại.

T giảm8caption bị CHAIR gắn cờ so với A:16ca không còn bị gắn cờ nhưng8ca mới xuất hiện.
CI của chênh lệch CHAIRs T−A là[−.018,+.001], nên mức giảm8ca vẫn chưa có CI tách0.
T so với C giảm11ca,CI[−.020,−.002]. Đây là nhãn tự động CHAIR, không phải kết luận
factual correctness do người kiểm. Số object mentions giảm35, đối tượng đúng được
nhắc (unique theoảnh) giảm24. Độ dài trung bình không giảm.

Recall T−A giảm0.7378điểm phần trăm, vượt giới hạn0.5điểm đã khóa.
Gate không đạt cả CIDEr và recall, dù CHAIRs/i point estimates tốt hơn A.

## Diễn giải và quyết định

- Giữ checkpoint A; không promote C/T, không chạy full/sweep/thêmseed cho cấu hình này.
- Không có post-training four-condition probe, nên chưa biết prompt reliance/cơ chế
  nội bộ đã đổi như thế nào. Không gọi đây là bằng chứng căn chỉnh thành công/thất bại.
- Đã gửi Astra báo cáo compact để chốt bước nghiên cứu tiếp, không yêu cầu đọc ZIP lần nữa.
- Một seed trên development đã tái sử dụng; CI không bao phủ seed variance hoặc thay
  thế final test độc lập. Chưa có novelty hay caption gain được xác lập.

## Chi phí và giới hạn audit

Staging C3.770.27s(~62.84phút),T1.132.47s(~18.87phút).
Train+end greedy C178.68s,T194.22s; I/O staging chiếm phần lớn thời gian.
Review ZIP không có checkpoint/HDF5 bytes; SHA đó là receipt do runtime báo, không
được xác minh trực tiếp local. Đã tái tính metric aggregates từ predictions/per-image
scores/CHAIR và kiểm source/metadata/logs. Không rerun CIDEr model/generation.
ZIP T chứa thêm nested C ZIP; artifact thừa đó không được dùng trong phân tích.

Kết quả chi tiết và provenance: `RESULT.json`. Tái tạo audit từ texts đã lưu bằng
`python audit_result.py`; các texts lớn nằm trong ignored `review/`.
