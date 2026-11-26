# Beam SCST: kiểm tra tốc độ cùng phiên

Ngày kiểm tra: 26/11/2026. Bản tối ưu giảm thời gian mỗi update khoảng **4,5–5,8%**. Có ích để dùng tiếp, nhưng chưa tạo thay đổi lớn về thời gian toàn epoch.

## Nguồn và phạm vi

- ZIP `beam_scst_v199_speed_compare_run1_review.zip`, SHA256 `b57e6aa31e4bdfaa1b584cd3b83f015c0692041901e9a525f26700c139580850`; 69 thành viên được đọc một lần và lưu cache. [Danh mục](SPEED_ZIP_MANIFEST.json).
- Checkpoint nguồn: `ducanh3/notebook7bc83c4c76/199`, full `checkpoint.bin` SHA256 `5b8d33fc0b26f2c77ad1cee4a58d3e4a718f13107a2d2bd9a0884c427faed88c`. Save Version của output benchmark chưa được người dùng xác nhận.
- Hai T4, thứ tự reference/candidate/candidate/reference; khôi phục cùng model, Adam và RNG trước mỗi block. Hai warmup và bảy phép đo mỗi block; bỏ tail 7 ảnh khi ngoại suy batch toàn cục 32. Mỗi biến thể có 12 quan sát, lấy thời gian rank chậm hơn mỗi update.
- Chỉ gom chuyển prompt theo shard và giảm đồng bộ scalar ở kiểm tra finite logits. Giữ nguyên FP32, beam 5, EOS, batch, loss, reward, optimizer và kiến trúc. [Protocol](THROUGHPUT_PROTOCOL.md).

## Kết quả đã đo

| Số đo | Bản gốc | Bản tối ưu |
|---|---:|---:|
| Trung vị giây/update | 3,1733 | 2,9902 |
| Trung bình giây/update | 3,1533 | 3,0124 |
| Chuẩn bị/chuyển batch, CPU wall trung bình (giây) | 0,1382 | 0,0174 |
| Beam forward, GPU event span trung bình (giây) | 1,0089 | 0,9784 |
| Backward, GPU event span trung bình (giây) | 2,0475 | 2,0373 |

Thời gian giảm 5,77% theo trung vị hoặc 4,47% theo trung bình; throughput tăng tương ứng 6,12% hoặc 4,68%. Cả sáu vị trí full-batch đều có trung bình candidate thấp hơn reference, nhưng không phải mọi quan sát riêng lẻ đều nhanh hơn. Peak allocated lớn nhất hai rank là 5,08 GiB; không tách riêng theo biến thể.

Ngoại suy 3.541 update: tiết kiệm **499–648 giây, khoảng 8–11 phút**. Theo trung vị, thời gian huấn luyện khoảng 3 giờ 07 phút xuống 2 giờ 56 phút. Đây là ngoại suy từ mẫu ngắn đã warmup, **chưa phải thời gian một epoch mới**; không gồm setup, đánh giá hoặc cách Kaggle tính quota.

Các event span có thể chứa khoảng chờ/launch và chồng lấp công việc CPU; không cộng chúng thành tổng thời gian hoặc coi là thời gian kernel riêng biệt. Forward và backward vẫn là phần lớn công việc được đo. Kết quả không ủng hộ giả thuyết thiếu cache là nút thắt chính còn lại.

## Kiểm tra và quyết định

Hai rank đều PASS: dữ liệu, đường beam, selected log probabilities, loss và RNG khớp ở guard một update; gradient/tham số được so với rtol 1e-6, atol 1e-7. Logits không hữu hạn được phát hiện. Trong các block đo, mean reward giống nhau; chênh lệch loss SCST lớn nhất khoảng 3,77e-8. Không suy ra toàn epoch sẽ bitwise giống nhau.

42 file nguồn khớp SHA; thống kê được tính lại từ cả hai rank bằng [audit_speed.py](audit_speed.py), chỉ đọc text/metadata đã lưu. [Số liệu audit](SPEED_RESULT.json). Không chạy Torch, mô hình, checkpoint nhị phân hoặc scorer local. 38 optimizer update thử nghiệm bị bỏ; không lưu trọng số dùng tiếp, không chấm VALID/TEST và không có kết quả CIDEr mới.

Khuyến nghị: dùng candidate có guard cho epoch 2 nếu người dùng chốt chạy tiếp. Phải bắt đầu từ full checkpoint gốc v199 với optimizer/RNG, không dùng trọng số benchmark. Epoch 2 cần hợp đồng chuyển epoch rõ ràng; runner resume trong epoch cũ kết thúc tại cursor 113.287 nên không thể chỉ đổi nguồn checkpoint rồi gọi lại. Chưa triển khai hoặc chạy epoch 2 trong lần review này.
