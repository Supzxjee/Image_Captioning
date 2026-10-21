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

## Phản hồi Astra sau audit

Astra chọn mở rộng sàng lọc có giới hạn, **chưa làm oracle một ảnh, chưa hạ gate 3/4**. Ở 200984, `holding` và `looking` không loại trừ nhau; đoạn sai là **`to his ear`**. Không dùng gradient `looking - holding` làm đại diện cho lỗi này. Phải tìm bản sửa tối thiểu đúng toàn ảnh trước khi khóa prefix/token. Đây cũng là giới hạn của việc chuyển nguyên contrast từ tác vụ chọn cụm sang sửa caption.

Bước kế tiếp tối đa 24 ảnh mới, chưa thực hiện trong audit này:

- Dùng `origin_predictions` đã lưu từ nhánh A với exact-original path nếu provenance checkpoint được xác nhận; ưu tiên hơn proxy masked slots.
- Chọn caption có khẳng định quan hệ người–ván hoặc điện thoại–tai/mặt bằng quy tắc văn bản, khóa thứ tự hash trước khi xem ảnh; không chọn theo reference/CIDEr.
- Ghi caption/đoạn sai, bằng chứng nhìn thấy, bản sửa tối thiểu toàn ảnh, nội dung đúng cần bảo vệ. Không tự tạo protected negatives hoặc decision tokens để đủ bảng.
- Đủ tổng cộng bốn ca hợp lệ thì trình bảng để khóa; hết 24 ảnh chưa đủ thì dừng, báo số thực tế và quyết định lại phạm vi công khai.

Kiểm tra đầu nguồn tiếp theo đã thấy `origin_predictions.json` có 1000 dòng trong artifact A. Trường `image_id` ở đó là **Karpathy imgid**, khác `coco_id`; ví dụ imgid 7936 tương ứng COCO 207561. Khi join ảnh phải dùng bảng `groups.json`, không ghép tên ảnh từ `image_id` trực tiếp. Chưa xem ảnh mới thuộc đợt 24 này.

Chưa tự thay gate, chưa thu thập 48 nhóm mới. Các ảnh đã xem vẫn là development có chủ đích, không phải mẫu đo tỷ lệ lỗi hoặc test độc lập.

Tái tạo audit bằng `audit_existing_data.py`; script có assertions về số nhóm/ảnh/shortlist và checkpoint SHA. SHA của các đầu vào được ghi trong JSON. Local contact sheet là artifact ignored, không push ảnh lên repo.
