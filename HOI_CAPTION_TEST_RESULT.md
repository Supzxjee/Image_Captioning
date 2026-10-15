# Q-Former + frozen UPT: independent test 5.000 ảnh

**UPT giữ được mức tăng CIDEr nhỏ trên test với λ=.1 đã khóa từ validation, nhưng chưa giảm hallucination so với fixed re-ranking.** Không huấn luyện thêm mô hình, không chọn lại trọng số theo test.

| Cấu hình | BLEU-4 | METEOR | ROUGE-L | CIDEr | CHAIRs ↓ | CHAIRi ↓ | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q-Former rank 0 | **0.370666** | 0.285039 | 0.573118 | 1.188245 | 4.32% | 3.0299% | 43.7508% |
| Fixed: decoder/CLIP/OCC=.3/.4/.3 | 0.369356 | 0.290791 | 0.577798 | 1.221815 | **3.48%** | **2.3712%** | 45.3131% |
| Fixed + UPT, λ=.1 | 0.370402 | **0.291138** | **0.578257** | **1.223864** | 3.50% | 2.3831% | **45.3385%** |

UPT tăng **0.002049017 CIDEr**, khoảng **0.168% tương đối**, so với fixed. BLEU-4 tăng 0.001045636 so với fixed nhưng vẫn thấp hơn rank 0. Mức tăng rank 0 → UPT gồm phần lớn lợi ích từ fixed re-ranking đã có; không quy toàn bộ cho UPT.

CHAIRs tăng từ **174 lên 175 caption hallucinated**: UPT sửa một caption và tạo thêm hai caption so với fixed. So với rank 0, pipeline UPT sửa 82 / tạo thêm 41, giảm ròng 41 caption hallucinated. Đây là đếm hallucination đối tượng theo CHAIR, chưa phải đánh giá lỗi quan hệ, thuộc tính hoặc tính đúng toàn bộ câu.

## Khóa cấu hình và phạm vi

Nguồn: `qformer_upt_test5000_locked_review.zip`; SHA256 `0787a180d4be96a50604fc2d09702ee032d4e3839c61695f9f75ff7612eb90c1`. Core `1c48b50b18111a33628670957df3dfdd2d0eb68a`; summary `TEST_COMPLETE`, split test, count5000; `trials=[]`, `retuned_on_test=false`.

Quyết định validation có SHA256 `4d812d8b4bcc02f4d24823b5eaf50d72a8fe41d5d6d735c036934b09bdf6ac38`. Dùng cùng Q-Former gốc epoch10, prompt cache gốc, CLIP revision, detection cache và frozen UPT như validation. Các ảnh test không trùng COCO ID với validation. Không thay bằng checkpoint E3.

UPT chỉ thay **33/5.000 caption (0.66%)** so với fixed. Có 279 ảnh không có cặp người–đối tượng, 950 ảnh có claim có điểm UPT và 443 ảnh có feature khác 0; chỉ **291 ảnh (5.82%)** có feature khác nhau giữa candidates. Trong 33 caption đổi, CIDEr từng ảnh tăng ở 22 và giảm ở 11; đây không phải chấm đúng/sai quan hệ bằng người. Phạm vi tác động còn hẹp: parser sáu nhóm hành động, hỗ trợ theo lớp đối tượng, chưa xác định duy nhất đúng instance của caption. Thiếu detection không được coi là bằng chứng đối tượng không tồn tại.

## Diễn giải và quyết định

Validation tăng CIDEr 0.001857258; test tăng 0.002049017 so với fixed. Có tín hiệu tăng điểm caption ở cả hai split, **chưa đạt mục tiêu giảm hallucination** và chưa đủ xem việc thêm UPT tiền huấn luyện là đóng góp phương pháp mới.

Lưu cấu hình và kết quả này làm đối chứng. Không dò thêm λ trên test để cải thiện số. Nếu tiếp tục phát triển, cần cải thiện độ phủ và tính đúng instance/quan hệ trên tập phát triển riêng; giữ test cho đánh giá chốt, không dùng 33 caption test làm nhãn điều chỉnh phương pháp.

Người dùng tự chạy notebook Kaggle. Trợ lý chỉ đọc kết quả và audit local, không khởi chạy notebook mới.

## Kiểm tra artifact

[Mã audit](audit_hoi_caption_test.py) đã kiểm tra CRC, hash, 5.000 ID duy nhất, không trùng validation, hai shard 2.500 ảnh của cả beam và UPT, candidate pool và các điểm gốc không đổi; replay parser, công thức feature và hai lựa chọn cố định. CIDEr/BLEU/ROUGE được tính lại, khớp trong 1e-10. Hai mốc rank0/fixed tái hiện đúng metric test đã báo cáo trước đây.

CHAIR được tái tổng hợp từ record từng caption với cùng ID, GT objects và caption; chưa chạy lại toàn bộ CHAIR scorer. METEOR chỉ đối chiếu file metric với summary. Không chạy lại UPT toàn bộ 5.000 ảnh vì ZIP không có raw teacher tensors.

Bootstrap ghép cặp theo ảnh, 20.000 lần, seed42, giữ IDF toàn tập cố định: khoảng percentile95% của CIDEr(UPT)−CIDEr(fixed) là **[0.000556622, 0.003799018]**. Khoảng không chứa0 trong phép kiểm tra này; kết luận chỉ có điều kiện trên checkpoint, candidate pool và quyết định validation đã khóa, chưa chứng minh độ ổn định qua nhiều seed huấn luyện.

[Kết quả audit máy đọc được](audit/hoi_caption_test_result.json). Đối chiếu 33 caption lưu local tại `C:/KhoaLuanTotNghiep/Code/output/hoi_caption_test_audit/changed_captions_33.json`; không dùng nhãn từ những caption test này để điều chỉnh phương pháp.
