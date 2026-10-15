# Instance-local relation prompts: cache-only development pilot

**Không đưa vào pipeline caption:** cạnh tranh đối tượng trong cùng một người tăng recall nhưng làm precision thấp hơn đối chứng. Chưa cần chạy full Kaggle nhánh này.

## Thay đổi đã thực hiện

[Module](captioning/instance_relation_prompts.py) giữ từng cặp `(person instance, object instance, predicate)` với hai bounding box, thay vì chỉ giữ điểm lớn nhất của lớp đối tượng trên toàn ảnh. Margin là chênh logit với đối tượng cạnh tranh mạnh nhất **của cùng người và cùng hành động**, kể cả đối tượng cùng lớp. Một người khác không được dùng làm đối chứng. Margin không giả định chỉ có một đích đúng: mỗi cặp có thể được chấp nhận riêng.

Khi không có đối tượng đối chứng, cặp vẫn được xét bằng ngưỡng điểm tuyệt đối; không bị ép về feature0 như cơ chế caption V1. Nhánh không có detection hoặc tổ hợp ngoài ontology abstain; không tạo nhãn phủ định hay kết luận hallucination.

Ngưỡng điểm và margin chọn **chỉ trên 111 ảnh calibration**, riêng từng predicate, với precision thực nghiệm >=90%, tối thiểu5 nhãn đã biết. Mục tiêu tối đa TP; tie ít FP hơn, sau đó ưu tiên không dùng margin. Grid margin cố định trước khi chạy: `None, -2, -1, 0, .5, 1, 2`. Precision trên calibration không bảo đảm precision trên development.

## Kết quả trên 128 ảnh development dùng lại

| Phương án | Nhận đúng | Nhận sai | Precision trên nhãn biết | Recall /196 positive | Nhận chưa có nhãn |
|---|---:|---:|---:|---:|---:|
| Raw UPT, cùng mục tiêu chọn ngưỡng | 80 | **12** | **86.96%** | 40.82% | 4 |
| Raw + cạnh tranh theo cùng người | **94** | 19 | 83.19% | **47.96%** | 4 |

Phần thêm đến từ mở gate `ride`: calibration13 đúng/1 sai nhưng development thêm14 đúng/7 sai. Gates `hold` và `sit_on` không đổi so với đối chứng mới. Không chọn lại predicate hay ngưỡng sau khi xem development.

Đối chứng raw ở pilot này khác kết quả acceptance trước (80 đúng/16 sai): pilot ưu tiên ít calibration FP hơn khi cùng TP, còn lần trước chọn đuôi thấp nhất đạt điều kiện. **Không quy mức 16→12 sai cho cơ chế instance-local.** So sánh hợp lệ ở đây là 80/12 với94/19 dưới cùng mục tiêu calibration.

Gate mở rộng đã đặt trước: precision development>=90%, số TP không thấp hơn raw và FP thấp hơn raw. **FAIL.** Không thay caption scorer đã chốt; không train hoặc chạy notebook.

## Artifact và mức độ bằng chứng

[Runner](run_instance_relation_prompt_pilot.py) dùng lại239 cache UPT đã có, kiểm SHA256 từng file và metadata, không chạy model inference. Source calibration111 ảnh/187 cặp, development128 ảnh/310 cặp; đều loại trừ toàn bộ10.000 ID của caption validation/test. ID holdout chỉ dùng loại trùng; không đọc caption/reference/metric test để fit. Source development đã được xem ở các trial trước, không phải đánh giá độc lập mới.

Đã xuất521 prompt instance trên115/128 ảnh development tại `C:/KhoaLuanTotNghiep/Code/output/instance_relation_prompt_pilot_v1/prompt_cache.json`. Mỗi prompt có chủ thể, đối tượng, hành động, hai box, điểm và margin. **521 không phải số prompt đã được xác nhận đúng**: precision trong bảng chỉ đo trên các cặp source được match và có nhãn. Những cặp khác trong graph chưa có nhãn đầy đủ.

Giữ hai box trong cache chưa giải quyết việc caption đang nhắc tới instance nào. Chưa ghép cache vào decoder, chưa tăng độ phủ parser caption, chưa đo CIDEr/CHAIR và chưa xác nhận tính mới khoa học. Đây là prototype cấu trúc và kiểm tra một cơ chế chọn prompt, không phải mô hình mới đã thành công.

[Summary và policy](audit/instance_relation_prompt_pilot_v1.json). Mười tests PASS gồm gate/unknown handling, singleton, đối chứng cùng lớp/cùng người, baseline, journal và bảo toàn candidates. Script local cùng phiên bản tạo kết quả; không khởi chạy Kaggle.

## Quyết định tiếp theo

Giữ UPT caption λ=.1 và fixed reranking làm đối chứng đã đo. Không mở full caption experiment cho bộ chọn margin này. Muốn phát triển tiếp cần bằng chứng phân biệt **đúng target instance** và kiểm soát sai lệch từ calibration nhỏ; source actor-absent negatives hiện tại không thay thế bộ nhãn sai đích đầy đủ. Chưa dùng số521 hay recall tăng để kết luận prompt đã tốt hơn.
