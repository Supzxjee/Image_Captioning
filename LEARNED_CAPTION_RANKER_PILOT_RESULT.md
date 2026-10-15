# Learned caption ranker: pilot hoàn tất, chưa qua gate mở rộng

**Nhánh vùng ảnh có tín hiệu giảm hallucination trên512 ảnh validation, nhưng CIDEr thấp hơn fixed và global. Không mở full theo tiêu chí đã đặt trước.** Không phải kết quả test mới.

| Chính sách | Epoch chọn trên tune256 | CIDEr ↑ | BLEU-4 ↑ | METEOR ↑ | CHAIRs ↓ | CHAIRi ↓ | Object Recall ↑ | Caption đổi / fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed .3/.4/.3 | — | **1.263698** | **0.372916** | **0.293225** | 3.1250% | 2.2164% | 44.0906% | 0 |
| Scores-only | 0 | **1.263698** | **0.372916** | **0.293225** | 3.1250% | 2.2164% | 44.0906% | 0 |
| Global | 1 | 1.259200 | 0.369174 | 0.290690 | 2.9297% | 2.0806% | 44.3356% | 91 |
| Regional | 3 | 1.251104 | 0.369891 | 0.289870 | **2.5391%** | **1.6603%** | **45.2541%** | 165 |

Các số trên chỉ so sánh cùng subset512 ảnh/candidate pool. Không so CIDEr1.263698 của subset này với1.210723 của full validation5000 hoặc1.223864 của test để tuyên bố tăng/giảm.

## Kết quả thực chất

- Regional so với fixed: CIDEr **−0.012594864**; CHAIRs giảm từ **16 xuống13 caption**. Sửa3 caption hallucinated, không tạo thêm trường hợp mới trong subset này.
- Hallucinated object mentions **17→13**; tổng mentions được nhận diện **767→783**; object GT được nhắc đúng **720→739** trên cùng1.633 GT objects. Vì vậy kết quả CHAIR tốt hơn không chỉ đi kèm việc bỏ hết đối tượng; chưa chứng minh mọi thay đổi caption đều đúng.
- Global sửa2 / tạo thêm1, giảm ròng1 caption hallucinated. Regional so với global sửa2 / tạo thêm0.
- Scores-only đã train10 epochs nhưng không epoch nào thắng khởi tạo trên tune; bộ chọn đúng quy tắc giữ epoch0. Checkpoint có residual head bằng0 và512 predictions giống hệt fixed.
- Global chọn epoch1; regional chọn epoch3, đúng cực đại tuneCIDEr trong histories. Loss train giảm ở các epoch sau nhưng tune không cải thiện tương ứng. Chưa có bằng chứng nên tăng số epoch.

Gate yêu cầu regional CIDEr>=fixed+.005, CHAIRs/i<=fixed, Recall>=fixed−.01, đồng thời CIDEr hơn scores và global. Regional qua các điều kiện CHAIR/Recall, **không qua điều kiện CIDEr**. Không đổi mục tiêu sau khi thấy kết quả để gọi gate thành công.

## Mức độ chắc chắn và giới hạn căn chỉnh

Bootstrap ghép cặp20.000 lần, seed42, corpus IDF giữ cố định:

| Regional − fixed | Chênh trung bình | Khoảng percentile95% |
|---|---:|---:|
| CIDEr | −0.012594864 | [−0.041592568, 0.016658542] |
| CHAIRs | −0.005859375 | [−0.013671875, 0] |

Mẫu chỉ512 ảnh, một seed, validation đã được xem; chưa có bằng chứng ưu thế ổn định. CHAIR đánh giá hallucination đối tượng, không đo đầy đủ đúng/sai quan hệ hoặc đúng instance. Checkpoint regional strict-load được và head đã cập nhật, nhưng điều đó không chứng minh attention đã tìm đúng vùng cho từng từ. Review không chứa private features lớn để chạy lại forward hoặc can thiệp vùng ảnh độc lập.

Training sử dụng preference CIDEr, không có nhãn đúng/sai quan hệ được xác minh. Đây là một giới hạn của supervision, chưa đủ xác định là nguyên nhân duy nhất khiến CIDEr không tăng.

## Kiểm tra artifact

Nguồn `learned_caption_ranker_1024_512_review.zip`,4.461.555 byte; SHA256 `b26b36118a1e9a2b57ceb99f6f07ab7e4f1d96ac50151d97903c5edbb80af162`. Core commit `404179bfa5b68381f86cce7f7a7d796a0c5dad5c`.

[Audit](audit_learned_caption_ranker_pilot.py) xác minh:

- Fit1024/tune256 thuộc train/restval, validation512 thuộc val;1.792 COCO IDs duy nhất; không trùng test5000.
- Checkpoint/prompt/dataset/detection hashes đúng baseline; hai phase beam/features mỗi phase có2 shard896 ảnh; candidates/decoder scores không bị thay khi thêm CLIP scores.
- Ba checkpoint hash đúng, strict-load đúng kiến trúc, mọi parameter hữu hạn; epoch chọn khớp tune histories và protocol hash. Không công bố weights lên GitHub.
- Reference validation512 khớp subset của bundle validation5000 cũ; fixed selection tính lại từ detector cache khớp.
- CIDEr/BLEU/ROUGE bốn nhánh tính lại khớp trong1e-10. CHAIR tái tổng hợp từ records cùng ID/caption/GT objects khớp; chưa chạy lại toàn bộ scorer. METEOR chỉ đối chiếu file/summary, chưa tính độc lập.
- Gate và transitions tính lại đúng. Không chạy thêm notebook, training hoặc inference model trong lần audit này.

[Kết quả audit máy đọc được](audit/learned_caption_ranker_1024_512_result.json). Local archive audit/checkpoints: `C:/KhoaLuanTotNghiep/Code/output/learned_caption_ranker_1024_512_audit`.

## Quyết định

Giữ cả ba head để đối chiếu, giữ fixed/UPT đã chốt làm mốc. Không thay mô hình hiện tại, không chạy full hoặc tăng epoch tự động. Bước tiếp theo cần kiểm tra vì sao thay đổi ứng viên làm mất CIDEr, và thiết kế tiêu chí học/chọn có kiểm soát regression trên tập train/tune; không chỉnh rule theo512 validation này rồi gọi là cải thiện độc lập. Tín hiệu giảm CHAIR đáng giữ lại nhưng chưa đạt mục tiêu đồng thời cải thiện chất lượng caption.

## Phân tích tiếp theo và notebook đang dùng

[Phân rã165 thay đổi](LEARNED_RANKER_CHANGE_DIAGNOSTIC.md):3 caption sửa hallucination đóng góp CIDEr dương; phần giảm đến từ các thay đổi khác, chưa chứng minh phải hy sinh CIDEr để giảm hallucination. Chưa chọn ngưỡng hoặc train lại. Người dùng xác nhận tên notebook hiện dùng là **QFormer UPT Val5000 20261015**; không suy phương pháp/split từ tên giữ lại của notebook.
