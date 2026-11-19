# M3 và sơ đồ tổng thể M1–M2–M3

- `m3_prompt_visual_alignment`: hình chi tiết M3, gồm chuẩn bị prompt riêng ảnh, nhánh đối tượng và nhánh quan hệ.
- `pipeline_m1_m2_m3`: sơ đồ tổng thể nối bank, đặc trưng thị giác, prompt, hai memory, decoder và reranking.
- Mỗi hình có bản `.svg` chỉnh sửa được, `.drawio` gồm các đối tượng riêng và `.png` xem trước. Bản sao ở Downloads cùng tên.

## Đọc M3 theo luồng

UPT cung cấp đối tượng, quan hệ và box. Các box được ghép với YOLO theo lớp và IoU; box YOLO đã ghép quyết định trọng số chồng lấp trên lưới patch 14×14. CLIP Text Encoder ViT-B/16 mã hóa các cụm prompt và chuẩn hóa L2 thành `t_o`, `t_r`.

Nhánh đối tượng dùng trọng số `W_o` để lấy đặc trưng `v_i` từ memory `V_o` của M2. Nối `t_o`, `v_i` và tọa độ UPT `b_i` (4 chiều), rồi qua Linear1028→512, ReLU, Linear512→512 để tạo slot. Cross-attention dùng `Q=V_o`, `K=V=slots`, chỉ cho phép patch đọc những slot hợp lệ có box chồng lấp patch đó. Cộng phần cập nhật vào `V_o` để có `V_o′`.

Nhánh quan hệ lấy hai đặc trưng `v_s`, `v_t` từ cùng `V_o` gốc. Nối với `t_r` và vector hình học tương đối `g` (8 chiều), rồi qua Linear1544→512, ReLU, Linear512→512. Attention dùng `Q=V_r` và các slot quan hệ làm K/V; mask là hợp vùng hỗ trợ của hai đầu quan hệ. Cộng phần cập nhật vào `V_r` để có `V_r′`. `g` không đơn thuần là nối hai tọa độ box: nó chứa độ lệch tâm, tỷ lệ kích thước, độ chồng lấp và khoảng cách.

Hai memory `V_o′`, `V_r′` đều có196×512, đi riêng vào hai nhánh cross-attention của decoder ba lớp. Patch không có slot hợp lệ giữ nguyên qua residual; các slot đệm không đóng góp vào kết quả.

M1 xây bank chung từ caption TRAIN; M3 dùng prompt riêng của ảnh đang xét. Hai loại thông tin này không thay thế nhau. Rerank dùng điểm hỗ trợ đối tượng từ detector; detector bỏ sót không xác nhận rằng đối tượng thực sự vắng mặt.

## Kiểm tra

Đã đối chiếu code được dẫn ở DESIGN.md, kiểm tra XML SVG/draw.io, AST generator,35/19 connector chỉ dùng đường thẳng hoặc gấp khúc vuông, kích thước PNG và SHA bản sao Downloads. Đã mở ảnh render, chỉnh tràn chữ, bỏ đường giao nhau trong tổng thể và tách đường hình học khỏi đường trọng số patch. Không chạy mô hình, notebook Kaggle hoặc sửa báo cáo.
