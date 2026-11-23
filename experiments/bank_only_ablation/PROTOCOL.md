# Đối chứng M2-only — duyệt ngày 23/11/2026

Người dùng đồng ý hoàn tất đối chứng còn lại sau zero-text TEST v198; báo còn 30 giờ GPU. Chỉ chuẩn bị notebook local và kiểm tra static/mock. Người dùng tự chạy Kaggle, không chạy mô hình/Torch/scorer local.

## Câu hỏi và lựa chọn

Baseline 3 lớp → M2-only đo toàn bộ bổ sung bank + căn chỉnh bank–ảnh. M2-only → zero-text đo bổ sung adapter M3 với vùng/hình học và không có chữ. Zero-text → full đo chữ prompt riêng ảnh. Đây là chuỗi đối chứng theo thành phần, không bảo đảm cộng tuyến hoặc tương tác bằng 0; tổng tham số khác nhau giữa các cấu hình, trừ cặp zero-text/full.

Giữ M1 noun/verb bank TRAIN-only v130; giữ toàn bộ M2 (hai aggregation, object alignment 1/2, relation alignment). Bỏ hẳn hai adapter M3; không đọc prompt v175, ROI, detector hoặc hình học. Hai memory gốc của M2 đi thẳng vào decoder parallel 3 lớp/dropout 0,2. Dùng constructor full rồi xóa riêng hai adapter để giữ nguyên thứ tự RNG và khởi tạo tất cả tensor còn lại; không dùng checkpoint đã học. Dự kiến 49.239.288 tham số, phải xác nhận thực tế trên Kaggle; full 53.183.736, baseline 29.553.424.

## Lịch và dữ liệu khóa trước

Fresh seed42, 10 epoch CE; 113.287 TRAIN/5.000 VALID, 566.747 caption events/epoch, global batch32, 17.711 update/epoch, 177.110 update tổng. Sampler image_grouped_42_plus_epoch, FP32/per-caption, Accelerate1.15.0/DDP T4×2, không sharing/AMP/compile. Giữ nguyên loss, optimizer, lịch LR, gradient clipping, vocabulary, patch extraction và decoder implementation từ scaffold baseline đã chạy. Best = CIDEr cao nhất trên VALID mỗi epoch; bằng điểm giữ epoch sớm. Beam5/max20/raw cumulative logprob, chưa SCST/rerank/TEST.

Ba Dataset: `ducanh2403/visual-cache`, `vuthetam/mscoco-2014`, `ducanh2403/csa-semantic-banks-v130`. Dùng H5 trực tiếp, không chép full patch. Kiểm SHA và thứ tự/vector L2 của bank bằng helper v130 đã dùng; stage chỉ hai bank NPY và seal vào manifest/identity. Không tải model pretrained hay chạy CLIP/UPT/YOLO. Nguồn được nhúng và hash trong notebook; không tải code nghiên cứu từ GitHub khi chạy. Evaluator/dependencies vẫn cần Internet, không cần Secret.

## Kiểm tra và an toàn tiếp tục

Host: compile/hash/importclosure/undefined globals; mock constructor chỉ xóa M3, routing chỉ core.encode; bank staging/seal mock, không nhận input prompt; recipe/selection/scoring giữ nguyên và từ chối resume sai phương pháp. Kaggle: xác nhận 49.239.288 tham số, mọi tensor/buffer còn lại bằng constructor full cùng seed, bank/M2 còn đủ, adapter không tồn tại, output bằng core.encode và bất biến khi thay prompt, positive control zero bank phải đổi memory; decoder/causal/beam/DDP first32 và tail27, 64 timing update bị bỏ. Main worker khởi tạo lại, không train từ guard.

Cap11.5h/phiên là giới hạn, không ETA/billing. Giữ đủ lịch10epoch dù chạm cap: lưu checkpoint/cursor/optimizer/RNG, giữ toàn bộ Output rồi resume chính arm này. Không tự tăng cap, đổi precision/batch hoặc restart từ đầu. Chỉ nhận resume cùng method/seed và identity/source/data/software; không nhận full/zero-text/baseline checkpoint. Chấm VALID best/end bằng cùng scorer đã chạy; METEOR/SPICE/TEST để lượt tiếp theo sau review, không thay chọn checkpoint.

## Giới hạn và tiêu chí đọc kết quả

Một seed; reused VALID và TEST exposure của dự án phải được khai báo. M2-only có ít hơn full 3.944.448 tham số, nhiều hơn baseline 19.685.864; không coi là đối chứng dung lượng bằng nhau. Historical fullseed42 dùng local patch staging, arm này dùng direct H5 như zero-text và baseline mới. Không khẳng định tăng điểm hoặc tương đương số học trước khi có kết quả. Không sửa Word, mở cơ chế mới hoặc tạo dòng kết quả Sheet cho notebook chưa chạy.
