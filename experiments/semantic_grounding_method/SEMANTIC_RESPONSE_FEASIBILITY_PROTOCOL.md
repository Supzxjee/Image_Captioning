# MED: kiểm tra nhãn Changed/Stable trước GPU

31/10/2026, user đã đồng ý kiểm tra thử mẫu nhỏ. Phạm vi: đọc metadata và ảnh, không model/Torch/Kaggle/training.

- Nguồn ứng viên: EasonFan/hallu_med revision `38f969ac98d7db856ee8d57b1a23c4032cfff09e`, file `med_sft_data.json` và `edit_image.tar`. Chỉ lấy mẫu từ train; không dùng MED benchmark/real benchmark.
- Mục tiêu 20 cặp đủ ảnh nếu truy cập được, không chọn theo mức thuận mắt. Ưu tiên truy xuất tuần tự từ archive và khớp exact metadata để tránh tải cả5GB; ghi rõ đây là convenience feasibility sample, không đại diện toàn dataset. Nếu gặp dữ liệu không tải được thì ghi lỗi, không âm thầm thay mẫu.
- Trước khi tải: ghi metadata identities, index gốc và đường dẫn. Chưa được coi là đã tách khỏi dev COCO vì cần mapping VG→COCO; mọi mẫu chỉ dùng feasibility, không fit/eval.
- Xem ảnh gốc/sửa ở kích thước đầy đủ và thumbnail224. Ghi: chỉnh sửa có rõ không, có tác động phụ không, một cặp cụm Changed đúng/sai trước/sau, tối thiểu một Stable cụ thể, độ rõ ở224, nhận/loại/chưa rõ và lý do.
- Không tự chuyển difference description thành truth label. Không nhận Changed chỉ vì caption nguồn nói như vậy; nếu ảnh không xác nhận thì bỏ/unknown. Không suy nhãn Stable từ từ ngữ giống nhau.
- Nhãn là `assistant_visual_review`, chưa phải người dùng xác nhận/ground truth. Không đưa file ảnh nguồn có bản quyền vào git; lưu review local ignored, chỉ push metadata/nhãn/kết quả dẫn nguồn phù hợp.
- Đầu ra: manifest + review sơ bộ + trang xem cặp ảnh + kết luận feasibility. Không tuyên bố novelty hay hiệu quả mô hình từ việc chấm dữ liệu. Chưa lập notebook huấn luyện.
