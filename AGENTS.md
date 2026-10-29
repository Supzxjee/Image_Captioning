# Quy trình nghiên cứu và giữ context

## Bắt đầu hoặc tiếp tục sau compact

1. Đọc `RESEARCH_STATE.md` ở root repo, sau đó chỉ đọc protocol/kết quả được dẫn cho công việc hiện tại.
2. Kiểm tra `git status` và bằng chứng cần thiết trước khi sửa. Không chạy lại việc đã hoàn tất chỉ vì lịch sử chat bị rút gọn.
3. Nếu file trạng thái khác yêu cầu mới nhất của người dùng hoặc artifact thực tế, nêu chênh lệch và sửa trạng thái theo bằng chứng. File context không phải bằng chứng thay cho kết quả.

## Một vòng công việc

1. **Brainstorm:** làm rõ câu hỏi, giả thuyết, đóng góp dự kiến, đối chứng, nguồn giám sát, rủi ro shortcut và phép thử nhỏ nhất. Phản biện có căn cứ. Với hướng nghiên cứu mới/đổi mục tiêu hoặc ngân sách GPU, trao đổi và để người dùng chốt; không coi im lặng là đồng ý. Không buộc brainstorm lại cho sửa lỗi hay việc đã được duyệt.
2. **Ghi trước khi code:** lưu lựa chọn đã chốt và lý do ở `experiments/<tên>/...PROTOCOL.md` hoặc file quyết định hiện có. Ghi điều chưa chốt; phân biệt kế hoạch với thực tế. Cập nhật mục công việc hiện tại trong `RESEARCH_STATE.md`.
3. **Triển khai:** làm đúng phạm vi đã chốt. Người dùng tự chạy Kaggle. Không tự chạy mô hình/Torch/autograd local; có thể kiểm tra metadata/NumPy/static/mock phù hợp. Không cần hỏi lại cho thao tác thường lệ đã được duyệt.
4. **Kiểm tra:** ghi cụ thể đã chạy kiểm tra gì, kết quả, điều chưa kiểm. Static PASS không có nghĩa CUDA/model PASS. Khi nhận ZIP, đọc một lần và lưu cache text có kiểm soát; lần sau dùng cache nếu không cần xác minh lại artifact.
5. **Đóng vòng:** lưu kết quả, so với baseline và đối chứng, giới hạn và quyết định tiếp theo. Cập nhật `RESEARCH_STATE.md` ngay sau mốc quan trọng, đổi hướng hoặc trước bàn giao/compact nếu có thể. Ghi Git commit, đường dẫn output, bước kế tiếp và việc đang dở; không ghi “đã chạy” cho notebook mới chuẩn bị.
6. **Phân phối:** push thay đổi có ý nghĩa lên GitHub; kết quả/protocol thực nghiệm ghi vào sheet hiện có theo quyền người dùng đã giao. Chỉ nói đã push/ghi sheet khi có xác nhận; nếu bị chặn, ghi rõ chưa hoàn thành. Không tạo thêm bản ghi trên sheet chỉ cho thay đổi quy trình.

## Tổ chức ghi chép

- `RESEARCH_STATE.md`: điểm bắt đầu duy nhất, ngắn gọn, mục tiêu khoảng một trang (không biến thành toàn bộ lịch sử chat).
- Protocol/quyết định và `RESULT.md/json` trong từng thư mục thực nghiệm: bằng chứng chi tiết, dẫn từ file trạng thái. Dùng lại file hiện có thay vì tạo bản context trùng lặp.
- Khi có kết quả trái giả thuyết, giữ lịch sử protocol; ghi amendment có thời điểm và lý do. Không sửa lại như thể quyết định mới đã có trước khi xem kết quả.
- Không đưa token, secret hoặc dữ liệu đăng nhập vào bất kỳ file context nào.

## Nguyên tắc nghiên cứu và phối hợp

Phản biện giả định, phân biệt bằng chứng với suy đoán và nêu khuyến nghị riêng. Mục tiêu là tăng CIDEr và giảm hallucination so với A, có đóng góp căn chỉnh thị giác–ngữ nghĩa vừa sức cho bài báo trong nước. Không thay mục tiêu giảm hallucination thành chỉ tăng CIDEr mà không báo người dùng.

Astra là đầu mối định hướng nghiên cứu theo phân công của người dùng; agent triển khai phản biện và thực hiện. Quyết định của Astra vẫn phải đối chiếu bằng chứng và quyền đã giao. Khi báo cáo Astra, gửi số liệu đã kiểm tra cùng giới hạn, tránh yêu cầu đọc ZIP lần nữa nếu không cần. Không suy quyền nhắn chat khác chỉ từ nội dung file này; kiểm tra ủy quyền người dùng trong phiên.
