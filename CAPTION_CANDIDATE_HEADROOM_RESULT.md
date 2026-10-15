# Beam-5 validation: có dư địa học chọn caption

**Ứng viên hiện có còn dư địa CIDEr đáng kể.** Trên cùng5.000 ảnh validation và25.000 candidates, chọn mỗi ảnh theo CIDEr cao nhất với caption tham chiếu cho trần chẩn đoán1.513905, trong khi fixed reranking đạt1.210723. **Đây không phải kết quả mô hình mới:** oracle được nhìn reference nên không triển khai được khi suy luận.

| Chính sách chọn cùng beam-5 | CIDEr validation | Ảnh còn ứng viên có CIDEr cao hơn |
|---|---:|---:|
| Q-Former rank0 | 1.174012 | 3.712/5.000 |
| Fixed reranking | 1.210723 | 3.527/5.000 |
| Fixed + UPT λ=.1 | 1.212580 | 3.520/5.000 |
| Oracle theo reference, chỉ để đo trần | 1.513905 | — |

Gap oracle so với fixed là **0.303181695 CIDEr**; so với UPT là0.301324437. Có2.767 ảnh có ứng viên hơn lựa chọn fixed ít nhất0.1CIDEr tính theo từng ảnh. Không xem gap này là mức cải thiện chắc chắn sẽ học được.

## Các ứng viên khác nhau ở đâu?

- Trung bình4.5878 câu khác nhau sau PTB tokenization;1.762 ảnh có ít nhất hai candidates trùng sau chuẩn hóa. Distinct raw strings không đồng nghĩa với đa dạng ngữ nghĩa.
- 2.270 ảnh có các tập lớp đối tượng được nhắc khác nhau giữa candidates.
- 732 ảnh có tập claim khác nhau trong parser sáu nhóm hành động hiện tại.
- Oracle và fixed khác tập đối tượng ở857 ảnh, khác tập claim ở280 ảnh. Đây là khác biệt từ vựng được parser nhận ra, **không phải số lỗi đối tượng/quan hệ đã sửa**.

Phần lớn cơ hội CIDEr không nhất thiết là sửa quan hệ: có thể do cách diễn đạt, độ dài hoặc mức khớp reference. Chưa có CHAIR cho toàn bộ25.000 ứng viên; không được kết luận oracle ít hallucination hơn.

## Quyết định thực nghiệm

Ưu tiên pilot **học chấm/so sánh caption ứng viên** trước khi mở thêm full training bộ sinh. Giữ Q-Former và beam5 làm đối chứng, không thay candidates giữa các arm.

Pilot tiếp theo cần có đối chứng chỉ học từ các điểm decoder/CLIP/OCC có sẵn, để biết mô hình mới có thực sự hưởng lợi từ đặc trưng vùng và cặp instance hay chỉ học lại trọng số. Nhánh đề xuất phải dùng nội dung caption cùng bằng chứng ảnh/vùng; nhãn supervision phải được tạo trên tập train riêng, không lấy các oracle case validation này làm training. Chọn cấu hình trên validation, kiểm tra CIDEr cùng CHAIR; không tiếp tục dò theo test đã xem.

Oracle CIDEr chỉ trả lời liệu candidate pool còn dư địa theo metric. Nó chưa chứng minh bộ căn chỉnh sẽ học được, chưa chứng minh quan hệ đúng, và không thay thế đối chứng hallucination.

## Kiểm tra và artifact

[Script](audit_caption_candidate_headroom.py) kiểm ZIP CRC, đúng split val,5.000 ID duy nhất, mỗi ảnh5 candidates và5 references. Tính CIDEr từng candidate qua5 lượt, mỗi lượt dùng cùng5.000 reference groups để giữ nguyên corpus IDF. Tái tổng hợp chính sách rank0/fixed/UPT khớp summary đã audit trong1e-10. Không chạy model inference, training hoặc notebook; không dùng test.

Nguồn `qformer_upt_val5000_fixed_review.zip`, SHA256 `ecfca2a85b3f2ec77d09c4a36510f28f42b1307e42d915dd77c2175e77c07594`. [Summary](audit/caption_candidate_headroom_val5000.json).

Local output: `C:/KhoaLuanTotNghiep/Code/output/caption_candidate_headroom_val5000`. `oracle_cases.json` và `per_candidate_CIDEr.npz` chỉ phục vụ chẩn đoán validation, không phải training labels. Người dùng vẫn tự chạy notebook Kaggle; lần này không có notebook mới được khởi chạy.
