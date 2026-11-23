# Ablation chữ prompt M3 trên TEST5000 — 23/11/2026

Nhánh zero-text hoàn tất chấm TEST ở `ducanh3/notebook7bc83c4c76/198`. Checkpoint vẫn là **v193, best epoch 8**, SHA256 `0aa81a0c0168c64987b77e0c86203fa5c4f7ccb858de806ed8842130ce37e1e0`; không chọn lại checkpoint bằng TEST. [Số liệu và kiểm tra](TEST_RESULT.json).

## Câu hỏi và đối chứng

Chữ trong prompt đối tượng–quan hệ riêng ảnh có giúp mô hình không? Hai nhánh cùng kiến trúc 53.183.736 tham số được học, seed 42, lịch 10 epoch CE và quy tắc chọn checkpoint bằng CIDEr cao nhất trên VALID. Zero-text giữ bank, M2, adapter M3, liên kết ROI, mask và hình học; chỉ đặt embedding chữ object/relation của M3 bằng 0 trong cả huấn luyện và suy luận. Nhánh full nhận chữ prompt đúng ảnh. Hai nhánh dùng beam 5, tối đa 20 token, lấy top1; **chưa SCST hoặc rerank** trong phép so sánh này.

## Kết quả

Điểm caption nhân 100; CHAIR là phần trăm, chênh lệch CHAIR là điểm phần trăm. Cột chênh lệch tính full trừ zero-text.

| Thang đo | Zero-text | Prompt đúng ảnh | Chênh lệch |
|---|---:|---:|---:|
| BLEU-1 | 77,49 | **77,63** | +0,13 |
| BLEU-4 | 37,68 | **37,74** | +0,06 |
| METEOR | 28,70 | **28,83** | +0,13 |
| ROUGE-L | 57,74 | **57,99** | +0,25 |
| CIDEr | 121,18 | **121,64** | +0,46 |
| SPICE | 21,69 | **21,85** | +0,16 |
| CHAIRs (%) | 4,52 | **3,96** | −0,56 |
| CHAIRi (%) | 3,15 | **2,80** | −0,35 |

Caption có đối tượng ảo giảm từ **226 xuống 198 trên 5.000 ảnh**, tức giảm ròng 28 caption, tương đương 12,39% so với zero-text. Lượt nhắc đối tượng ảo giảm 230 → 205, trong khi tổng lượt nhắc đối tượng tăng 7.294 → 7.317. Recall đối tượng tăng 43,3380% → 43,4904%, không có dấu hiệu giảm số đối tượng được nhắc ở mức tổng hợp. Đây không phải bằng chứng từng caption đều mô tả đầy đủ hơn.

NLL zero-text 2,181292 so với full 2,182139: full cao hơn 0,000847. Vì vậy không nói tất cả đại lượng đều tốt hơn; tám thang đo chính cùng cải thiện theo hướng mong muốn, còn NLL hơi kém hơn.

## Diễn giải

TEST cùng chiều với VALID về giảm ảo giác đối tượng. Trên VALID, CIDEr gần bằng nhau và CHAIRs giảm 4,96% → 4,00%; trên TEST, full đồng thời giảm CHAIR và tăng nhẹ CIDEr. Kết quả **ủng hộ giả thuyết chữ prompt riêng ảnh có ích cho kiểm soát ảo giác đối tượng, với chất lượng caption được giữ ở mức tương đương về số điểm quan sát và tăng nhẹ trên TEST**. Chưa thực hiện phép thử tương đương/noninferiority hoặc kiểm định ý nghĩa thống kê.

Đối chứng này rõ hơn so với bỏ hẳn adapter: số tham số khai báo, ROI và hình học được giữ nguyên. Tuy vậy, zero input khiến các cột trọng số tương ứng chữ không nhận tín hiệu huấn luyện như nhánh full, nên không khẳng định dung lượng hoạt động hoàn toàn bằng nhau. Chỉ có một seed cho ablation, best epoch khác nhau (8/9) theo cùng tiêu chí VALID, khác I/O staging lịch sử; TEST đã xuất hiện trong các vòng phát triển trước. CHAIR chỉ đánh giá đối tượng, chưa xác nhận giảm sai thuộc tính, hành động hoặc quan hệ.

## Kiểm tra và quyết định

ZIP SHA256 `71458942487a5338888c64b9162ab2bcc11f2f9c41e07189969a46703f88fe49`, CRC và 74 file khớp manifest. Khớp 31 source SHA với notebook đã giao, đúng checkpoint/source/mode, worker mỗi GPU 2.500 ảnh; guard trên 8 ảnh/GPU xác nhận text invariance, ROI/geometry/mask giữ nguyên, memory delta 0. Kiểm tra đủ 5.000 caption, 25.000 beam, token → text/top1, cohort/reference, seal của các phase chấm, trung bình CIDEr/SPICE, NLL có trọng số và toàn bộ đếm CHAIR. Bộ chấm có cùng digest với full seed42. Checkpoint/NPY lớn không có trong ZIP; dựa trên receipt/runtime guard, không nạp binary hay chạy Torch/scorer local.

Giữ kết quả này làm ablation M3-text cho bài viết. Chưa tự mở thêm lượt train, seed, rerank zero-text hoặc đối chứng M2-only; M2-only vẫn là mục còn lại của kế hoạch đã nêu. Không sửa Word.
