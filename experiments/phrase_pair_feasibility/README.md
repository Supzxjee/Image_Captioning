# Audit dữ liệu trước pilot sinh cụm — 20/10/2026

**Chỉ chuẩn bị dữ liệu và review. Chưa huấn luyện, không chạy notebook/model.**

## Cách dùng cho người chấm

Mở `review/index.html` bằng Chrome/Edge; có thể mở trực tiếp file, không cần server hay Internet. Giữ thư mục `images/` cạnh HTML. Bản ZIP local chứa cả ảnh gốc, metadata và UI; giải nén trước khi mở.

Làm **10 nhóm đầu trước**. Mỗi nhóm có hai cặp người–vật và hai mô tả C1/C2:

1. Kiểm tra box S xanh và O cam của mỗi cặp; đánh giá đúng instance, không phải người bất kỳ trong ảnh.
2. Với mỗi cặp, chấm C1 và C2 riêng: đúng, sai hoặc không rõ. Không ép nhóm thành hai mô tả đối lập.
3. Mở ảnh resize224×224; chấm bằng chứng phân biệt còn thấy hay không. Số tâm patch chỉ là kiểm tra hình học, chưa chứng minh CLIP hiểu nội dung.
4. Nhập tên, tải `phrase_pair_feasibility_review.json`, gửi lại. Trình duyệt tự lưu nếu cho phép; nên tải JSON trước khi đổi máy.

Ví dụ giả: nếu một người vừa cầm vừa mang vật, cả hai mô tả có thể đúng; không coi một mô tả là sai chỉ vì nhãn V-COCO không đánh dấu nó. Với người đứng trên ván, có thể vẫn đang mang/cầm ván theo một nghĩa khác; chấm theo ảnh và đúng vật, ghi chú nếu cụm mơ hồ. Nếu người ngồi xe đồng thời đứng nhổm trên bàn đạp thì câu “sitting” có thể không đúng: source action ride chỉ là gợi ý chọn ảnh.

## Tập đã chuẩn bị

- 30 nhóm, 60 lượt event, 53 ảnh duy nhất; 7 nhóm cùng ảnh. Không ảnh nào xuất hiện ở hai nhóm khác nhau.
- 5 họ: bicycle posture2, skateboard support9, surfboard support9, chair posture1, phone position9.
- Loại495 ảnh đã biết trong F fit/calibration/probe, probe cũ và review prompt50. Không tuyên bố đã rà soát toàn bộ lịch sử hoặc near-duplicates.
- Chỉ caption train/restval. Tất cả có thể đã được backbone học caption; không phải ảnh mới độc lập. Audit này chưa phân fit/dev/probe và không được dùng làm probe mới sau khi đã chấm/sửa thiết kế.
- Không chọn theo model error. Chọn hash cố định, luân phiên họ, ưu tiên cùng ảnh trong mỗi họ. Đảo thứ tự event bằng hash; không điền ma trận nhãn dự kiến vào UI.
- V-COCO train/val cung cấp action/role gợi ý; COCO instances cung cấp box. Mọi mô tả sinh theo proxy đều **chưa được xác minh đúng**, bốn ô semantic truth đều `unreviewed`.
- Không có ứng viên đối chiếu horse phù hợp luật mining hiện tại sau loại trừ; không suy ra dataset không có người đứng cạnh ngựa. Không nới luật hoặc thêm mô tả giả để đủ quota.

### Dấu hiệu cần xem trước khi train

13/60 lượt event có S hoặc O không có tâm patch trong grid14×14; 28/60 có một đầu ít hơn4 tâm patch. Các số này báo nhu cầu fallback/chi tiết nhỏ, **không chứng minh feature CLIP không có thông tin**, vì patch tokens cuối chứa ngữ cảnh toàn ảnh.

Prompt JSON có cặp hai object được chọn khớp GT với IoU≥0.5 ở48/60 lượt event. Đây chỉ là **độ có mặt trong selected objects của prompt JSON**, không phải raw UPT proposal recall, độ đúng predicate hay metric model caption. Raw cache trước lọc không được dùng trong audit này. Chưa có human usable rate; chưa chốt việc tạo170 nhóm hoặc training300updates.

## Mã và kiểm tra

- `data.py`: lấy candidate, chọn nhóm, kiểm tra số tâm patch và selected-object matching.
- `prepare.py`: đọc dữ liệu local và ZIP prompt do user cung cấp, không thực thi mã từ ZIP; xuất manifest và summary.
- `assets.py`: lấy ảnh có sẵn hoặc tải ảnh COCO công khai, xác minh JPEG/kích thước, lưu SHA256.
- `render.py`: tạo HTML offline, không gửi nhãn ra mạng.
- `review_result.py`: kiểm tra hash/ID/enum của JSON human; không đổi unreviewed/unclear thành negative, không tự duyệt huấn luyện.
- `test_data.py`: 5 tests về giới hạn box, lựa chọn nhóm, coverage, nhãn review và artifact thật.
- `test_ui.cjs`: kiểm tra logic UI/JSON export trong DOM giả lập cô lập; không điền nhãn vào browser/user data.

Đã xác nhận đủ53 JPEG đúng kích thước metadata; 5 data tests và isolated UI logic PASS. Chưa xác minh giao diện qua browser screenshot: IAB không kết nối được localhost của máy này. Không chạy GPU/model, không tạo nhãn human thay người dùng.

## Trình tự sau audit

Nếu đủ nhóm có ma trận ngữ nghĩa rõ, box đúng và bằng chứng dùng được, mới chốt pilot. Phương án đang được ưu tiên để thảo luận là **cùng encoder/adapter: CE so với CE+đối chứng hai chiều**, để kiểm tra lợi ích objective trực tiếp. Chưa triển khai hai nhánh, chưa lựa chọn hyperparameters hay tuyên bố novelty. Không mặc định so hard masks là câu hỏi chính trước khi xác nhận dữ liệu.

Nếu nhóm phần lớn mơ hồ/chi tiết quá nhỏ, dừng cách mining/cache này và điều chỉnh dựa trên audit; không tăng loss để cứu. Nếu scoring tốt nhưng sinh cụm không tốt hơn, chưa scale. Sinh cụm tốt cũng chưa phải caption tốt; cần milestone caption matched sớm sau đó.

Ảnh gốc và ZIP review local không được commit lên repo. `candidate_manifest.json` và `PREPARATION.json` là snapshot chuẩn bị, không phải kết quả semantic human hoặc model.
