# Zero-text M3: seed 43 và 44 — kế hoạch đã chốt 25/11/2026

Người dùng đồng ý lặp đối chứng zero-text để ghép với Full prompt đã có seed42/43/44. Sau đó báo **không đủ quota GPU**. Vì vậy chỉ chuẩn bị notebook và kiểm tra static/mock ở local; **chưa chạy thêm**, chưa có kết quả mới. Đợi người dùng có quota và tự chạy Kaggle. Không suy diễn số giờ còn lại hoặc thời điểm quota hồi.

## Câu hỏi và lựa chọn cố định

Kiểm tra mức giảm hallucination quan sát được khi có embedding chữ riêng ảnh có ổn định qua seed hay không. Chạy cả seed43 và44 đã định trước, kể cả nếu một seed cho kết quả không thuận lợi. Không thêm seed theo chiều kết quả. Dùng lại full-prompt seed43/44 đã huấn luyện/chấm TEST; không train lại full.

Đối chứng zero-text seed42 vẫn là checkpointv193/TESTv198. Full43 checkpoint ducanh3/notebook7bc83c4c76/v179, TESTv181; Full44 checkpoint ducanh2403/2-vi-cai-kia-qua-nhieu/v236, TESTv238. Nguồn đối chứng mới chỉ được ghi sau khi người dùng báo Save Version; không bịa version hoặc tải checkpoint để khởi tạo.

## Huấn luyện

Kế thừa notebook zero-text seed42 đã hoàn tất và runner H5 dùng cho Full43/44: 10 epoch CE từ đầu, 113287 TRAIN/5000 VALID, 566747 caption events/epoch, 17711 update/epoch, tổng177110; seed43 hoặc44. Giữ full53.183.736tham số, toàn bộ bank/M2, adapterM3, ROI/mask/hình học. Chỉ object_text/relation_text trong M3 thành0 ở train và mọi VALID/inference. Noun/verb bank không đặt0.

Giữ decoder3parallel/dropout0.2, global32, FP32/T4x2/Accelerate1.15.0/DDP, per-caption/directH5/no sharing/AMP/compile; giữ optimizer/LR/gradient clip và sampler seed+epoch. Best=maxVALIDCIDEr mỗi epoch, tie giữ sớm; beam5/max20/no length normalization. Không thay lịch học hoặc chọn epoch theo CHAIR. Không SCST/rerank/TEST trong notebook train.

Gắn bốn Dataset: `ducanh2403/visual-cache`, `vuthetam/mscoco-2014`, `ducanh2403/csa-semantic-banks-v130`, `ducanh2403/csa-prompt-full-karpathy-v175`. Dùng cache sẵn, không chạy encoder/detector hoặc tạo lại cache.

Giữ cap11.5h của scaffold cho **một invocation**, không phải dự báo hoàn tất10epoch hay số giờ quota. Không khẳng định một phiên đủ; seed42 đã phải resume. Checkpoint/latest/best/optimizer/RNG/cursor và toàn bộ metadata được giữ để tiếp tục đúng seed. Đổi RUN_TAG khi resume; không dùng checkpoint full-prompt, seed42 hoặc seed khác. Khi có Save Version mới sẽ chuẩn bị API resume khóa nguồn thực tế nếu cần. Không tăng cap hoặc chạy lại từ đầu để né việc thiếu quota.

## Kiểm tra và phân tích đã định

Local chỉ compile/hash/import-closure/static/mock; dùng lại kiểm tra zeroing không mutate input và giữ các trường ROI/hình học; kiểm tra chỉ SEED, giới hạn factory theo seed và mô tả provenance đổi, các hàm train/selection/checkpoint/semantic model/data/guard không đổi. Không import Torch hoặc chạy model/autograd/scorer local. Kaggle guard thực mới xác nhận CUDA, đúng full initialization cho seed đang chạy, số tham số và hành vi zero-text.

Sau đủ10epoch và audit output, chấm TEST cùng quy tắc8chỉ số đã khóa. So Full−zero theo từng seed42/43/44, báo đủ từng seed và mean±sampleSD. Trọng tâm CHAIRs/CHAIRi; CIDEr kiểm tra chênh lệch chất lượng, không khẳng định tương đương từ việc CI chứa0. Không coi 3×5000 ảnh là15000 quan sát huấn luyện độc lập. Bootstrap theo ảnh và biến động giữa seed là hai nguồn bất định khác nhau; ba seed vẫn ít.

Giữ giới hạn: TEST đã được dùng phát triển; hồi cứu sau kết quả seed42; cùng số tham số khai báo nhưng text0 làm một số cột trọng số không hoạt động; CHAIR chỉ đo đối tượng. Không bảo đảm nhiều seed sẽ cho kết luận thuận lợi. Không mở cơ chế mới hoặc sửa Word trong việc chuẩn bị này.
