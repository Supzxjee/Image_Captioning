# TEST5000 của M2-only best9 v245 — 24/11/2026

Người dùng đồng ý chấm TEST sau bảng VALID hoàn tất của M2-only. Chuẩn bị notebook local, người dùng tự chạy Kaggle. Không chạy Torch/model/scorer hoặc tải checkpoint/API thực local; không train, rerank, chỉnh tham số bằng TEST hoặc đổi checkpoint.

Khóa **ducanh2403/2-vi-cai-kia-qua-nhieu/245**, best.bin epoch9/seed42, SHA `3fe31a921742e841c0d3863aac9ab9f00b0cd6b46289f3cf36029ac1fa168226`. Snapshot epoch10/cursor0/159399updates/5100723captionevents/history9, bestVALID CIDEr1.1992600798055864. Không dùng final10 SHA d26af3af… hoặc best8 của run1v243 dù CHAIR khác. Toàn bộ42 training source SHA trong checkpoint phải khớp khóa huấn luyện đã kiểm.

Giữ nguyên BankOnlyCaptioner, core, M2 và decoder từ lượt train; strictstate166values,49.239.288tham số. Hai bank lấy từ buffer `core.nouns`5269×512 và `core.verbs`1594×512 trong checkpoint. Giữ M2, không tạo/nạp hai adapter M3 và không đọc ROI/prompt/detector. Đầu mỗiGPU, kiểm trên8ảnh TEST thực: hai memory bằng core.encode, thay prompt giả không làm đổi memory, bank finite/L2 và đúng số tham số; lưu guard riêng, không thay trọng số. Kiểm beam/rescore trước khi lưu caption.

Chỉ hai Dataset `ducanh2403/visual-cache` và `vuthetam/mscoco-2014`; stage đúng5000TESTpatch≈1.4GiB, không đọc feature TRAIN/VALID hoặc copy fullcache. Same cohort/vocabulary/preprocessing/FP32/T4×2, beam5/max20/rawlogprob/no length normalization, lấy top1 như baseline/zero/full. ReferenceTEST chỉ để chấm metric/NLL sau sinh câu; không dùng chọn câu.

API pinnedCLI chỉ tải best.bin exactv245, verifySHA trướcload; Secret `codex` hoặc `KAGGLE_API_TOKEN` có quyền đọc nguồn. CHECKPOINT_ROOT trống mặc định; có thể gắn đúng best.bin nếu muốn, vẫn kiểmSHA. Không tạo Dataset checkpoint mới, không cần cache bank130/prompt175/181. Nguồn nghiên cứu/evaluator assets metadata được nhúng/hash từ scaffold TEST đã chạy, không tải code nghiên cứu từ GitHub. Dependencies/Java/CHAIR vẫn cần Internet.

Lưu top1 và5beam/ảnh trước CPUmetrics. Giữ nguyên tokenize/basic/METEOR/ROUGE-L/CIDEr/SPICE/CHAIR và phaseinputs từ matchedbaselineTEST thành công. Chỉ complete khi đủ8thang đo chính (+Recall/NLL), mỗi phase cóseal để phục hồiCPU nếu lỗi; không tự decode lại. Cap2h gồmsetup/scorers là giới hạn, không ETA/billing. Không tự cài lại Torch hoặc fallback sang trọng số khác.

Hostchecks: compile/hash/importclosure theo đường suy luận (không đi vào contract_guard chỉ dùng khi train), undefinedglobals, exactAPIroutev245, metadata/weights accept/reject, byte-identical trainingmodel và AST-equivalent scorers, guaranteed bank-only factory/realGPUguard hooks, allmetrics completeness. ActualAPI/binary/CUDA/Java/caption quality pendinguser-run.

Mộtseed M2-only và priorTESTdevelopment exposure của dự án phải ghi rõ; không gọi đây là untouchedTEST. Khác dung lượng tham số giữa baseline/M2/full, cùng quy tắc chọn maxVALID CIDEr. Chưa kết luận significance, noninferiority hoặc đúng thuộc tính/hành động/quan hệ từ CHAIR. Không sửa Word hoặc ghi dòng kết quả Sheet cho notebook chưa chạy.
