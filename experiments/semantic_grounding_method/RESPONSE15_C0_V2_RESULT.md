# C0 semantic response15 v2 — kết quả user-run

## Kết quả và giới hạn

V2 hoàn tất15/15cặp, không huấn luyện. 109tensor bare C0 khớp checkpoint; digest trước/sau069e1954…; sanity cùng input PASS. User runtime TeslaT4, worker421.87s (~7phút; không phải toàn bộ wall time/quota). Lượt đầu lỗi wrapper/bare được giữ lịch sử, không còn là trạng thái hiện tại.

| Chỉ số mô tả | Kết quả |
|---|---:|
| Changed phản ứng đúng chiều | 14/15 |
| Changed ưu tiên đúng trên cả ảnh gốc và sửa | 7/15 |
| Stable ưu tiên đúng trên cả hai ảnh | 9/15 |
| Đồng thời đạt Changed và Stable cả hai ảnh | 4/15 |
| Caption greedy khác chữ | 13/15 |
| Mean absolute Stable contrast delta | 1.5761 |

Changed ưu tiên đúng riêng ảnh gốc10/15, ảnh sửa12/15. Stable riêngảnhgốc11/15, ảnhsửa10/15. Stable mất preference đúng E17,E39; E37 từ sai sang đúng. E05 là cặp duy nhất Changed delta sai chiều. Không có thang chuẩn cho Stable delta1.5761 để gọi đạt/trượt. E02 editedmargin −0.0101 rất gần0, không coi một dấu nhỏ là robust.

## Diễn giải

C0 đã nhận tín hiệu thay đổi ảnh trong điều kiện giữ nguyên prompt; không phải hoàn toàn phớt lờ ảnh. Tuy nhiên14đúngchiều không đồng nghĩa14đúngnghĩa: chỉ7cặp đổi preference đúng hai phía, và4cặp đồng thời giữ Stable đúng hai phía. Điều này ủng hộ kiểm tra khả năng chọn lọc, chưa chứng minh loss đề xuất tốt hơn pairedCE. Không diễn giải thành attention grounding/token-to-region hoặc causal mechanism đã xác lập.

Caption E02 chuyển “a red double decker bus driving down a street.” → “a blue double decker bus driving down a street.”; E09 thêm “blue” cho vase và giữ “pink flowers”. E18/E19 caption khôngđổi và khôngnhắcmàu: không tự tính làlỗi. E34 đồngthời đổi hot dog→sandwich; đây là thayđổi nội dung câu cần xemảnh, chưa tự gắn nhãn hallucination. E39 shirt contrast mất preference đúng nhưng caption khôngnhắcshirt: probeerror khôngtựđộng là captionerror. C22 captiongốc ghi pink, nhãn đãuserconfirm purple: không sửa nhãn hoặc tự gọi captionđúng; nhạy với cáchgọimàu.

## Khuyến nghị

Không chạy lại C0 diagnostic này hoặc mở training lớn chỉ dựa vào14/15. Nhờ Codex2 phản biện prior/delta và thiết kế pairedCE/change/selective-stability đã ghi; nếu user chốt pilot, cần đủ nhãn train và tập tách riêng trước khi so ba nhánh cùngdata/exposure.15cặp đã xem chỉ là diagnostic, không dùng lại làm bằng chứng generalization. Chưa có CIDEr/CHAIR gain hoặc phương pháp mới được xác nhận.

## Kiểm tra artifact

Đọc ZIP một lần thành cache localignored `review/response15_v2_user`;30imagehashes,11sourcehashes đối chiếuGitcommit,15unique sampleIDs,summary+mọimargin táitính,records_partial==records,identity/sanity/statusPASS. Agent không chạy Torch/model/notebook. [Chi tiết và records](RESPONSE15_C0_V2_RESULT.json).

| ID | Changed gốc | Changed sửa | Stable gốc | Stable sửa |
|---|---:|---:|---:|---:|
| C01 | 3.4760 | -1.6072 | 5.2935 | 4.9928 |
| C17 | -0.2101 | -2.6616 | 3.5194 | 2.2993 |
| C22 | 0.0741 | -1.5904 | 7.6132 | 7.6665 |
| E02 | 4.1908 | -0.0096 | 4.8602 | 0.2078 |
| E05 | 3.4276 | 3.5576 | 2.3237 | 2.5686 |
| E09 | -0.4109 | -4.1067 | 4.4221 | 1.4142 |
| E17 | 2.4740 | -1.9877 | 2.7549 | -0.2551 |
| E18 | -2.0539 | -2.6252 | -0.3190 | -0.6085 |
| E19 | -0.6332 | -1.0660 | -0.7798 | -0.5370 |
| E21 | 0.9620 | -0.9279 | 0.8805 | 0.9586 |
| E30 | -2.8220 | -4.1987 | -1.0638 | -2.8197 |
| E33 | 1.2454 | 0.9291 | 1.7776 | 1.5873 |
| E34 | 1.6153 | 1.1176 | 4.0180 | 3.3155 |
| E37 | 2.2316 | -3.5679 | -1.2710 | 3.5647 |
| E39 | 2.8603 | -1.6969 | 1.5517 | -1.5064 |
