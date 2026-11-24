# Tiếp tục M2-only từ v243 — 24/11/2026

Lịch 10 epoch đã được duyệt; đây là tiếp tục lượt đang dở, không mở thí nghiệm mới. Nguồn do người dùng xác nhận: **ducanh2403/2-vi-cai-kia-qua-nhieu/243**. Run1 dừng ở safety deadline, đã lưu epoch9/cursor5616/update147304/4.713.688 caption events. Tám VALID epoch hoàn tất; best tạm epoch8 CIDEr119,3362. Chưa có CHAIR hoặc kết quả TEST, không coi đây là ablation cuối.

Khóa latest `checkpoint.bin` SHA `16b76195ae52ebe150d87b81843906a862fc13903a005b031407388753475bab` và best8 SHA `a5d9dbab46e0f188d3db28b7caa66d2687abfbd625a17ddf4dc04d90642a4510`, receipts/identity/source/progress/traininglog và 48 file của tám VALID epoch. Còn **29.806 update**, gồm 12.095 cuối epoch9 và 17.711 epoch10. Nạp latest để khôi phục optimizer/RNG/cursor; không nạp best8 làm điểm bắt đầu, không reset LR hoặc thay lịch10epoch.

**T4 ×2, Internet ON, Save & Run All.** Giữ ba Dataset visual-cache, vuthetam/mscoco-2014 và csa-semantic-banks-v130 như run1. Không thêm prompt175/ROI, không tạo Dataset checkpoint mới. Bật Secret `codex` hoặc `KAGGLE_API_TOKEN` có quyền đọc nguồn v243. `RESUME_ROOT=''` mặc định tải qua API đúng v243; không fallback sang latest. Nếu đã gắn toàn bộ Output v243 thì có thể điền RESUME_ROOT; ZIP review không có trọng số nên không dùng để resume.

Giữ nguyên toàn bộ **42 training sources**, SOURCE_SHA256, SOURCE_ORIGINS và source_receipt của run1 để strictidentity hoạt động. Verifier/API chỉ nằm ở wrapper. Nguồn CLI được pin, kiểm bằng mock rằng yêu cầu gửi đúng owner/slug/version_label v243. Kiểm CUDA trước download, SHA của cả binary và 54 metadata/history trước khi nạp. Identity gồm software và stage seals phải khớp; không tự cài lại Torch, đổi FP32, DDP, batch32, sampler, bank hoặc decoder.

Giới hạn phiên tiếp tục **6 giờ**, không phải dự báo thời gian hoặc quota đã đo lại. Hạn này chỉ cho phần còn lại, không lặp cap11.5h và không chạy lại147304update. Preflight bỏ64step trên khởi tạo mới; worker chính nạp latest đúng v243 rồi tiếp tục. Sau đủ177110update/5667470events mới chấm best/end VALID. Best vẫn chọn maxVALID CIDEr; chưa SCST/rerank/TEST.

Gửi `m2_only_bank_alignment_seed42_resume243_review.zip`, giữ toàn bộ Output chứa checkpoint.bin, best.bin, source/receipts/history. Host chỉ kiểm metadata/static/mock; binary restore/optimizer/RNG/CUDA và chất lượng phần còn lại do người dùng chạy. Thực nghiệm chỉ một seed và khác tổng tham số so với baseline/full; không khẳng định đóng góp cuối trước khi hoàn tất.
