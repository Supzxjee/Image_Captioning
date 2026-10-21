# DRSA: audit dữ liệu cũ, 22/10/2026

**DATA_AUDIT_COMPLETE_ORACLE_NOT_READY.** Không chạy model, không tạo nhãn human mới, không triển khai notebook oracle.

## Điều kiện Astra đã chốt

Oracle đóng băng checkpoint original SHA `5f8eead3…`, giữ legacy memory cố định và sửa đúng 32 Q slots sau `build_memory`. Dự kiến 8 ảnh dựng basis, 4 ảnh chẩn đoán lỗi quyết định toàn ảnh. Objective oracle: full-vocabulary decision CE + 0.1 protected KL; norm cap 2%, tối đa 100 bước; gradient basis so với random basis cùng không gian bảo vệ. Đây là label-assisted intervention, chưa chứng minh grounding hoặc gain so với A.

## Dữ liệu thực sự có

| Hạng mục | Số lượng | Ý nghĩa |
|---|---:|---|
| Nhóm đã có nhãn human về selected pair | 21 | Không đồng nghĩa nhãn toàn ảnh |
| Nhóm cùng ảnh, hai đích selected-pair đối nghịch | 4 | Loại khỏi cách chuyển trực tiếp sang whole-image task |
| Nhóm cross-image còn lại | 17 | 34 ảnh riêng biệt, có bản ảnh local |
| Ảnh shortlist dựng basis | 8 | Đề xuất assistant, chưa khóa |
| Ca sai quan hệ rõ qua sàng lọc assistant | 1 | COCO 200984 |
| Ca sai đối tượng bổ sung | 3 | Không gộp thành lỗi quan hệ |
| Ảnh đủ toàn bộ điều kiện oracle đã khóa | 0 | Thiếu protected contrasts, target/prefix và xác nhận global |

Đã xem contact sheet 34 ảnh; xem thêm ảnh đầy đủ 200984 và 503488. Các nhận định toàn ảnh mới là **assistant screening**, không nâng cấp thành nhãn human. File JSON kèm theo giữ đủ 34 ID, nhãn cặp, caption lịch sử, provenance và trạng thái pending.

Shortlist basis: **440092, 368370, 520204, 45949, 540716, 570474, 526325, 181841**. Không gán tự động các ảnh còn lại thành bốn ca chẩn đoán.

## Caption lịch sử giúp sàng lọc

Đọc `before_records.json` từ ZIP FULLFRAME đã có, không chạy lại inference. Protocol dùng đúng checkpoint 5f8eead3…; điều kiện before/off dùng global memory với auxiliary slots bị mask, trước adaptation. Đây là historical baseline proxy. Chưa độc lập xác minh parity của sinh câu với exact-original path bỏ hoàn toàn auxiliary slots; phải kiểm khi chạy oracle.

| COCO ID | Caption before/off | Nhận định sàng lọc |
|---|---|---|
| 200984 | a man is holding a cell phone to his ear. | Điện thoại ở trước mặt, không ở tai: ứng viên lỗi quan hệ rõ |
| 503488 | a man walking down a road with a bag of luggage. | Hai người mang skateboard: sai đối tượng |
| 521602 | a fish fish fish in the sand at the beach. | Người mang surfboard: sai đối tượng/lặp câu |
| 160647 | a group of people in the water with a paddle boat in the water. | Paddleboarders: sai loại đối tượng |

Không tính việc caption chỉ nói “đi bộ”, “đứng trên bãi biển”, “có laptop” là sai quan hệ chỉ vì thiếu hành động mang ván/dùng điện thoại. Không dùng caption sau adaptation để chứng nhận lỗi của original.

## Chưa đủ để triển khai recipe 8+4

1. Xác nhận target là sự kiện rõ toàn ảnh, không chỉ selected pair; không ép số ít khi ảnh có nhóm người.
2. Cần đủ bốn ca relation-error nếu giữ gate 3/4; audit hiện chỉ chứng nhận một ứng viên qua assistant screening, không tuyên bố đã rà hết mọi lỗi trong mọi dữ liệu cũ.
3. Chưa có protected contrasts đã xác minh và đang đúng ở baseline. Không thay danh từ ngẫu nhiên để tạo negative.
4. Khóa decision prefix/token theo tokenizer thực tế; không lấy prefix tiếng Anh bằng cách đoán token.
5. Không gọi các development cases đã xem là held-out confirmation.

Đã gửi Astra bản tóm tắt thiếu dữ liệu, đề nghị chốt một bước nhỏ: oracle một ca với giới hạn kết luận rõ ràng, hoặc mở rộng screen caption đã có. Chưa tự thay gate, chưa thu thập 48 nhóm mới.

Tái tạo audit bằng `audit_existing_data.py`; script có assertions về số nhóm/ảnh/shortlist và checkpoint SHA. SHA của các đầu vào được ghi trong JSON. Local contact sheet là artifact ignored, không push ảnh lên repo.
