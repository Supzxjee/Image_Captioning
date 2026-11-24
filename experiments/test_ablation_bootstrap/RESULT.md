# Độ bất định của ablation trên TEST5000

Đã hoàn tất ngày 24/11/2026: 100.000 mẫu paired bootstrap trên 5.000 ảnh, dùng lại điểm CIDEr và số đếm CHAIR đã lưu. Không huấn luyện, sinh caption, chạy lại scorer hay dùng GPU. [Số đầy đủ](RESULT.json), [protocol trước resampling](PROTOCOL.md), [nguồn và SHA](INPUT_MANIFEST.json), [dữ liệu từng ảnh](PAIRED_INPUT.csv).

**Kết luận:** điểm quan sát ủng hộ prompt/M3, nhưng bằng chứng cho riêng embedding chữ còn yếu. Chưa đủ cơ sở khẳng định prompt làm tăng CIDEr hoặc giảm hallucination một cách chắc chắn. Lợi ích giảm CHAIR của toàn bộ M3 so với M2-only rõ hơn ở khoảng 95% từng chỉ số, nhưng không giữ được khi điều chỉnh cho sáu phép so sánh.

## Kết quả

Chênh lệch là full prompt trừ đối chứng. CIDEr tính theo thang nhân 100; CHAIR tính bằng **điểm phần trăm**, không phải phần trăm tương đối. CIDEr dương hoặc CHAIR âm có lợi cho full prompt.

| So sánh | Chỉ số | Chênh lệch | Khoảng 95% từng chỉ số | Khoảng 99,17% Bonferroni |
|---|---|---:|---:|---:|
| Full − zero-text | CIDEr | +0,458 | [−1,005; +1,938] | [−1,527; +2,453] |
| Full − zero-text | CHAIRs | −0,560 | [−1,120; 0,000] | [−1,320; +0,180] |
| Full − zero-text | CHAIRi | −0,352 | [−0,743; +0,036] | [−0,879; +0,166] |
| Full − M2-only | CIDEr | +0,232 | [−1,192; +1,673] | [−1,689; +2,165] |
| Full − M2-only | CHAIRs | −0,740 | [−1,340; −0,160] | [−1,540; +0,060] |
| Full − M2-only | CHAIRi | −0,493 | [−0,918; −0,070] | [−1,060; +0,076] |

Khoảng chứa 0 nghĩa là phép phân tích chưa phân biệt rõ chênh lệch với 0; không chứng minh hai cấu hình tương đương hoặc prompt vô ích. Tương tự, chưa được dùng kết quả này để tuyên bố chất lượng caption không giảm: chưa định trước ngưỡng và kiểm định không thua kém.

## Những gì hai cặp đối chứng cho biết

**Full so với zero-text** giữ adapter, vùng ảnh và hình học, nên gần với câu hỏi về giá trị của chữ prompt hơn. CHAIRs quan sát giảm từ 4,52% xuống 3,96%, nhưng khoảng 95% chạm 0; CHAIRi và CIDEr đều có khoảng chứa 0. Đây là xu hướng thuận lợi của một lần huấn luyện, chưa phải kết luận chắc chắn về đóng góp riêng của embedding chữ.

**Full so với M2-only** kiểm tra toàn bộ M3, gồm adapter, ROI/hình học và chữ prompt, đồng thời khác tổng tham số. CHAIRs giảm 4,70% → 3,96%; CHAIRi giảm 3,295% → 2,802%. Hai khoảng 95% loại trừ 0, nhưng khoảng điều chỉnh không loại trừ 0. Có tín hiệu giảm lỗi đối tượng ở cấp hệ thống; chưa tách riêng được nguyên nhân do chữ hay các thành phần còn lại của M3. CIDEr +0,232 vẫn không rõ ràng.

## Thay đổi trên từng ảnh

| So sánh | Cả hai không bị CHAIR gắn cờ | Cả hai bị gắn cờ | Đối chứng bị cờ, full không | Đối chứng không bị cờ, full bị cờ |
|---|---:|---:|---:|---:|
| Full / zero-text | 4.687 | 111 | 115 | 87 |
| Full / M2-only | 4.670 | 103 | 132 | 95 |

Chênh lệch ròng 28 hoặc 37 caption che đi các thay đổi hai chiều. Không được gọi 115/132 ảnh là lỗi đã được sửa bằng bằng chứng thị giác: đây chỉ là trạng thái do scorer CHAIR xác định, chưa phải nhãn xác nhận của con người.

## Cách tính và kiểm tra

Ghép theo COCO ID, xác nhận cùng 5.000 ID và cùng caption tham chiếu; mỗi mẫu lấy lại 5.000 ảnh có hoàn lại, dùng chung chỉ số lấy mẫu cho cả ba cấu hình. CIDEr lấy trung bình điểm từng ảnh với document frequencies của bộ tham chiếu ban đầu giữ cố định. CHAIRs lấy tỷ lệ caption có ít nhất một đối tượng ảo. CHAIRi tính lại tỷ lệ **tổng** đối tượng ảo trên **tổng** đối tượng được nhắc trong mỗi mẫu.

Seed PCG64 20261124; 100.000 mẫu; percentile quantile tuyến tính. Khoảng 95% là từng endpoint. Khoảng Bonferroni dùng alpha 0,05/6 cho sáu endpoint, độ phủ đồng thời chỉ ở mức xấp xỉ bootstrap. Không tính p-value. Tính thống kê CPU mất khoảng 61 giây; không phải thời gian huấn luyện.

Bốn kiểm tra NumPy/stdlib đạt: ghép đúng ID dù đảo thứ tự và chặn ID thiếu/trùng; CHAIRi là ratio-of-totals và chặn mẫu có mẫu số 0; hai nhánh giống hệt có chênh lệch đúng 0; resampling đồng bộ khớp tham chiếu và không đổi theo kích thước batch. SHA nguồn, điểm trung bình và số đếm khớp các audit trước. CSV kèm SHA đủ chạy lại `bootstrap.py` mà không cần checkpoint/scorer hoặc ZIP gốc; `prepare_inputs.py` chỉ cần khi dựng lại CSV từ cache đã kiểm tra.

## Giới hạn và khuyến nghị

Đây là phân tích hồi cứu sau khi biết điểm tổng hợp, trên một seed cho các đối chứng. Bootstrap theo ảnh không thay thế huấn luyện nhiều seed và không khắc phục TEST đã được dùng trong phát triển. CIDEr giữ reference IDF; CHAIR chỉ đo lỗi nhắc đối tượng, không xác nhận thuộc tính, hành động hay quan hệ. Không chọn checkpoint hoặc chỉnh công thức dựa trên kết quả bootstrap.

Khuyến nghị: giữ mô hình hiện tại, trình bày lợi ích là **xu hướng giảm ảo giác**, và ưu tiên lặp đối chứng zero-text với seed đã định trước, cùng quy trình với full prompt, nếu tiếp tục củng cố đóng góp của chữ prompt. Không mở thêm cơ chế mới chỉ vì khoảng tin cậy hiện còn rộng.
