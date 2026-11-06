# Chấm bổ sung chỉ số TEST đã khóa — 06/11/2026

Người dùng duyệt: bổ sung BLEU-1, METEOR, SPICE cho caption đã lưu. Hai notebook CPU riêng tài khoản: CONTROL lấy đúng184, PROMPT lấy đúng61. Chấm normal và fixed_rerank trên cùng TEST5000. Không tải checkpoint, ảnh, CLIP/UPT; không sinh lại caption, không train hoặc thay lựa chọn rerank.

Khóa SHA từng payload refs/captions/result/test_lock/run_status từ review text đã audit139checks. Chấm bằng pycocoevalcap1.2/PTB, thêm BLEU1/METEOR/SPICE, replay BLEU4/ROUGE/CIDEr trong sai số1e-7. CHAIRs/i lấy lại kết quả lịch sử đã audit, có caption và result SHA đúng; không gọi là CHAIR được tính mới. Cả raw và thang×100 được xuất. Chỉ COMPLETE khi đủ2×5000 và8chỉ số; không xếp hạng từ subset nếu scorer lỗi.

Dùng Java8 Temurin8u472b08 khóa SHA từ release chính thức; CoreNLP3.6.0 tải qua HTTPS và ghi SHA thực tế (không tuyên bố SHA model đã pin độc lập). Smoke PTB/BLEU/METEOR/SPICE trước full. Từng phase có heartbeat và timeout; preserve output/Spice cache, không tự retry/chấm lại model. Parent STOPcap4hCPU, không phải ETA. API Secret của chính chủ notebook, không ghi token vào log. CPU scorer thực tế do người dùng chạy Kaggle; tác giả chỉ chạy static/stdlib fixtures/metadata.

Nguồn: https://github.com/salaniz/pycocoevalcap (Java1.8, CoreNLP3.6.0); https://github.com/adoptium/temurin8-binaries/releases/tag/jdk8u472-b08 .

Kiểm tra metadata: TEST có4990ảnh×5refs và10ảnh×6refs. Giữ nguyên toàn bộ tham chiếu gốc, không cắt thành5; hai nhánh dùng cùng SHA references.
