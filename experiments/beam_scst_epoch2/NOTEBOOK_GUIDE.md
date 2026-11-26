# Beam SCST epoch 2 → VALID5000 → TEST5000 + fixed rerank

Học tiếp **đúng một epoch** từ full checkpoint của `ducanh3/notebook7bc83c4c76/version199`, giữ Adam và RNG hai GPU. Cursor epoch2 bắt đầu0, cumulative update3541→7082; không học lại epoch1 và không reset optimizer. Dùng hai tối ưu vừa kiểm tra tốc độ, giữ FP32/beam5/global32/loss/model như epoch1. Không tự epoch3 hay chọn checkpoint theo TEST.

**Kaggle: GPU T4×2, Internet ON, Secret `codex` hoặc `KAGGLE_API_TOKEN` có quyền đọc nguồn ducanh3.** Không cần gắn output199: notebook tải đúng full checkpoint bằng API và xác minh SHA. Gắn5Dataset; các đường dẫn ở đầu cell cấu hình:

- `ducanh3/csa-scst-full-karpathy-cache-v1`
- `vuthetam/mscoco-2014`
- `ducanh2403/visual-cache`
- `ducanh2403/csa-prompt-test-cache-v181`
- `ducanh2403/objectdetectionecache`

Cap mặc định **5,75 giờ wall**, dự phòng100 phút sau train. Không phải ETA hay cam kết quota6giờ đủ. Train dự kiến gần3giờ từ mẫu tốc độ, thời gian setup/Java có thể thay đổi. Mỗi100update và khi hết cap lưu full checkpoint; nếu epoch2 chưa đủ thì dừng, không chấm TEST bằng partial weights. Giữ **toàn bộ Output**, reviewZIP không chứa weights.

VALID top1 so với epoch1 cùng seed43. TEST sinh beam5 một lần, chấm top1 và rerank cố định **0,3 decoder + 0,4 CLIP + 0,3 object**, đủ8thang đo + Recall/NLL, so với TEST epoch1/output200 và CE179 đã lưu. Không tìm trọng số trên TEST. Caption/beam/scores được giữ trước scorer để recovery nếu lỗi; không tự train lại. Đây là TEST đã được dự án dùng trước trong phát triển.

`RESUME_ROOT` chỉ nhận **toàn bộ output epoch2** đã gắn ở input, kiểm tra SHA/identity rồi tiếp cursor cũ. Bắt đầu mới để trống. Nếu chỉ còn Save Version, gửi owner/notebook/version để làm API resume đúng nguồn; không điền output epoch1 vào trường này.

Agent chỉ kiểm tra syntax/metadata/NumPy/mock; API, khôi phục binary/CUDA, thời gian toàn lượt và metric mới chờ người dùng chạy.
