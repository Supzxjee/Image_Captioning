# Q-Former + UPT: kết quả validation 5.000 ảnh

**UPT tăng CIDEr rất nhỏ so với fixed re-ranking; chưa cải thiện hallucination.** Đây là kết quả validation dùng chọn trọng số, chưa phải kết quả test độc lập. Không huấn luyện thêm mô hình.

| Cấu hình | BLEU-4 | METEOR | ROUGE-L | CIDEr | CHAIRs ↓ | CHAIRi ↓ | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q-Former rank 0 | 0.370322 | 0.283012 | 0.571214 | 1.174012 | 4.28% | 2.9793% | 42.9119% |
| Fixed re-ranking: decoder/CLIP/OCC = .3/.4/.3 | 0.374110 | 0.290747 | 0.578156 | 1.210723 | **3.56%** | **2.3983%** | 44.5756% |
| Fixed + UPT, λ = .1 | **0.374985** | **0.291059** | **0.578652** | **1.212580** | 3.58% | 2.4090% | **44.6195%** |

UPT tăng **0.001857258 CIDEr**, khoảng **0.1534%** tương đối so với fixed. Phần lớn mức tăng so với rank 0 đến từ fixed re-ranking đã có, không được quy hết cho UPT. Caption có hallucination tăng từ **178 lên 179**: UPT sửa một trường hợp và tạo thêm hai trường hợp so với fixed.

## Tác động và giới hạn

- Cùng 5.000 ảnh và cùng beam candidates, log-probabilities, CLIP scores cho cả ba nhánh; không loại ảnh không có cặp UPT.
- 256 ảnh không có cặp; 998 ảnh có ít nhất một claim phân tích được và có điểm UPT; 487 ảnh có feature khác 0; chỉ **323 ảnh** có feature khác nhau giữa các candidates.
- UPT đổi caption trên **43/5.000 ảnh (0.86%)** so với fixed: CIDEr từng ảnh tăng ở 31, giảm ở 12. Đây là đối chiếu với reference, không phải nhãn người về tính đúng của quan hệ.
- Parser chỉ phủ sáu nhóm hành động và các mẫu từ vựng rõ ràng. Điểm là hỗ trợ tồn tại theo lớp đối tượng trong ảnh, chưa ràng buộc đúng từng instance; không xác minh đầy đủ thuộc tính, số lượng hay mọi quan hệ.
- Dùng UPT tiền huấn luyện là đối chứng thực nghiệm, chưa tự tạo thành đóng góp phương pháp mới.

Bootstrap ghép cặp theo ảnh, 20.000 lần, seed 42, giữ IDF toàn tập cố định, cho khoảng percentile 95% của chênh CIDEr **[0.000158711, 0.003673189]**. Khoảng này chỉ mô tả validation đã dùng chọn λ, **không hiệu chỉnh quá trình chọn trọng số**, không thay thế test độc lập hay chứng minh ưu thế tổng quát.

## Chọn trọng số và bước tiếp theo

Grid đã cố định trước lượt chạy: λ = 0 / .05 / .1 / .2; CIDEr lần lượt **1.210722882 / 1.211267237 / 1.212580140 / 1.212365442**. Chốt **λ=.1**, giữ decoder/CLIP/OCC=.3/.4/.3 cho đánh giá test; không chọn lại theo test. File khóa: [audit/hoi_validation_locked_selection.json](audit/hoi_validation_locked_selection.json).

Bước tiếp theo là đánh giá độc lập ba nhánh trên test với cấu hình đã khóa. Chưa cần train thêm. Giữ fixed re-ranking làm đối chứng: kết quả hiện tại chưa đạt mục tiêu giảm hallucination. Không so trực tiếp CIDEr validation 1.212580 với CIDEr test lịch sử 1.221815.

**Người dùng tự chạy notebook Kaggle.** Lần này chỉ đọc artifact, tính lại metric và cập nhật hồ sơ; không khởi chạy notebook mới.

## Bằng chứng kiểm tra

Nguồn người dùng: `qformer_upt_val5000_fixed_review.zip`, 6.034.546 byte; SHA256 `ecfca2a85b3f2ec77d09c4a36510f28f42b1307e42d915dd77c2175e77c07594`. Core commit `8e7a552bf81631fd1480a27ed04db105fd72c277`; summary `VALIDATION_COMPLETE`, split `val`, count 5000.

[Mã audit](audit_hoi_caption_validation.py) xác minh CRC/hash, đủ 5.000 ID duy nhất, hai shard 2.500 ảnh, danh tính ảnh, nguồn beam không đổi, parser/feature, lựa chọn cả grid; tính lại CIDEr/BLEU/ROUGE cho ba nhánh khớp trong 1e-10. CHAIR tái tổng hợp từ record từng caption khớp; chưa chạy lại toàn bộ CHAIR scorer. METEOR đối chiếu file metric với summary, chưa tính độc lập. ZIP không chứa raw teacher tensors để tái kiểm tra toàn bộ inference UPT.

[Kết quả audit máy đọc được](audit/hoi_caption_validation_result.json). Phân tích 43 caption thay đổi lưu local tại `C:/KhoaLuanTotNghiep/Code/output/hoi_caption_validation_audit/changed_captions_43.json`.
