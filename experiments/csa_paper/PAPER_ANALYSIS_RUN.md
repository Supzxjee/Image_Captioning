# Phân tích bổ sung cho bài báo

## Chạy tiếp

Hai notebook `CSA_PROMPT_SPICE_SUBSETS_CONTROL_CPU.ipynb` và `CSA_PROMPT_SPICE_SUBSETS_PROMPT_CPU.ipynb`: **Accelerator None, Internet ON**, bật secret API `codex` của đúng chủ notebook. CONTROL chạy tài khoản ducanh2403, tự tải caption version184; PROMPT chạy ducanh3, tự tải version61. Không cần checkpoint, ảnh hoặc GPU; không train/decode/rerank lại. Nếu đã gắn output đúng lượt thì điền SOURCE_ROOT để không dùng API.

Notebook tải mã theo commit4aabab2, xác minh SHA captions/references/status, dùng pycocoevalcap1.2 và Java8 như lượt chấm trước. Chỉ chấm SPICE normal/fixed_rerank, đối chiếu điểm All với kết quả185/62. Giữ per-image Objects/Relations/Attributes, undefined là null và có số ảnh hợp lệ. Không chuyển undefined thành0. Giới hạn4h CPU là STOPcap, không phải dự báo thời gian. Gửi review ZIP mỗi nhánh sau khi xong. Chỉ gọi COMPLETE khi đủ cả hai chính sách.

Khi nhận đủ hai đầu ra, dùng compare_spice_subsets.py để tính chênh lệch trên cùng tập ảnh có điểm hợp lệ ở cả hai nhánh. Không chỉ trừ hai trung bình có mẫu số khác nhau.

## Đã tổng hợp từ metadata hiện có

[Chi phí](paper_analysis/COSTS.md): CONTROL49.239.288, PROMPT53.708.024 tham số caption được học, thêm4.468.736 (~9,08%). Thời gian pha GPU có sẵn chỉ là log hai phiên riêng và chưa gồm trích xuất prompt/ảnh, không dùng để khẳng định PROMPT nhanh hơn hay công bố độ trễ end-to-end.

[Danh sách24 mẫu](paper_analysis/review.html), [JSON đầy đủ](paper_analysis/example_shortlist.json): 3ảnh/nhóm ×4nhóm chuyển cờ CHAIR ×2chính sách. Sau rerank: bỏ cờ137ảnh, phát sinh cờ70ảnh, cả hai bị cờ90ảnh, cả hai không cờ4703ảnh. Giảm ròng67khớp227→160. Đây là shortlist tự động, chưa kiểm chứng ảnh; không gọi các caption không bị CHAIR đánh dấu là đúng. Ảnh tải từ COCO qua Internet khi mở HTML. Người dùng kiểm ảnh trước khi chọn minh họa trong bài.

## Kiểm tra và giới hạn

Hai pure-data tests PASS (undefined/paired denominators), AST toàn bộ code cell PASS, input payload SHA được kiểm lại từ text cache, không import Torch hay chạy model/scorer/Kaggle local. CPU Java/SPICE/API runtime chưa thực thi; chưa có số SPICE con mới. Một seed, khác capacity và checkpoint khởi đầu C10/P9, TEST đã được xem vẫn là giới hạn. Không huấn luyện mới; RegionCLIP tạm hoãn. Không sửa báo cáo Word.
