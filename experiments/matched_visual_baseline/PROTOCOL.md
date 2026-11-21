# Baseline thị giác cùng decoder — 22/11/2026

## Quyết định đã được người dùng chốt

Người dùng yêu cầu train lại baseline sau khi thống nhất dùng ba lớp và cùng decoder với mô hình cuối. Chuẩn bị notebook local để người dùng tự chạy Kaggle; không tự chạy mô hình hoặc notebook. Một seed42, fresh initialization, 10 epoch CE. Không mở GPT-2, SCST, TEST hoặc nhiều seed trong lượt này.

## Câu hỏi và đối chứng

So sánh mô hình cuối có bank + prompt với baseline chỉ dùng patch thị giác, giữ nguyên projection 768→512, embedding, positional encoding, ba DecoderLayer parallel attention, dropout0.2, vocabulary head, sampler, dữ liệu, optimizer/lịch học, batch và số update. Baseline đưa cùng V đã project vào cả hai nhánh attention của decoder; không có semantic aggregation, alignment encoder hoặc prompt adapter trong mô hình được tối ưu.

Baseline là đối chứng loại bỏ toàn bộ nhánh ngữ nghĩa, không phải đối chứng chỉ nội dung chữ của M3; số tham số tổng khác nhau. Hai attention riêng vẫn giữ nguyên tham số dù cùng nhận V. Không so sánh với baseline bốn lớp như thể đó là một phép tách riêng căn chỉnh. Chưa có kết quả baseline mới và chưa bảo đảm chênh lệch sẽ có lợi.

## Tái sử dụng và khởi tạo

Tái sử dụng runner per-caption FP32 H5 trực tiếp đã dùng cho các seed của mô hình cuối, pinned commit5eb012279e6fd0b5e47a56710ed5e3efa7449d2f và các sửa seed/resume đã kiểm tra. Không feature sharing, AMP, compile hoặc thay batch. Để khởi tạo chính xác các tensor chung, khởi tạo mô hình đầy đủ theo mã gốc rồi loại bỏ các module/buffer ngữ nghĩa trước DDP/optimizer. Notebook phải kiểm tra tensor chung bằng nhau trước huấn luyện trên Kaggle; không nạp trọng số đã học.

Chỉ cần visual-cache và COCO JSON. Tắt semantic staging và thay SemanticContext bằng context rỗng; không đọc bank/prompt cache, không chạy lại CLIP/UPT/YOLO và không chép full patch cache sang memmap. Constructor dùng hai bank giả toàn0 kích thước1×512 chỉ để tái hiện thứ tự khởi tạo mã gốc, rồi xóa cả buffer và module ngữ nghĩa. Kích thước/nội dung bank chỉ là buffer, không ảnh hưởng thứ tự/kích thước các phép khởi tạo tham số trong mã đã pin; kiểm tra AST xác minh điều này, Kaggle so từng tensor chung với constructor full cùng seed. Đây là khởi tạo ngẫu nhiên, không sử dụng checkpoint đã học.

## Lịch học, checkpoint và giới hạn

TRAIN113287, VALID5000, toàn bộ566747 caption events/epoch; global batch32,17711 updates/epoch,177110 updates tổng; seed42, image-grouped42+epoch. FP32/Accelerate1.15.0/T4×2. Adam(.9,.99),eps1e-8,clip1; warmup một epoch8e-6→1e-4 rồi cosine đến1e-5. Chọn best bằng VALID CIDEr, tie giữ epoch sớm; beam5/max20, không length normalization/rerank. Giữ nguyên cách CE và lịch VALID, không chọn theo TEST.

Cap11.5h là giới hạn một lần gọi, không phải ETA hoặc số giờ GPU còn lại. Lưu checkpoint/optimizer/RNG/cursor và best; resume chỉ từ chính recipe baseline này. Không lấy checkpoint của full model hoặc baseline199 để train tiếp. Nếu hết cap, giữ toàn bộ output; không tự restart hoặc tăng cap. Ngân sách quota hiện tại chưa được xác nhận; người dùng tự quyết định lúc chạy theo quota thực tế.

## Kiểm tra

Local: compile/AST mọi cell và embedded source; xác minh pin/hash nguồn, local import closure, contract bỏ module/giữ decoder, scope checkpoint/resume và protocol; NumPy/mock nếu cần. Không import Torch hay chạy neural/scorer local.

Kaggle trước train: CUDA hai GPU, shared-weight initialization, baseline không phụ thuộc nội dung prompt, hai memory cùng V, causal/beam scoring, gradient/DDP first32 và tail27, discarded timing64steps. Lưu số tham số và hashes vào review. Static PASS không thay thế GPU PASS.

## Báo cáo sau chạy

So sánh trước rerank với full seed42 CE cùng10epoch; VALID chỉ để chọn checkpoint. TEST chỉ chuẩn bị sau khi xác minh output và người dùng cho chạy. Một seed không thay thế matched multi-seed; CHAIR chỉ đánh giá đối tượng. Không sửa file Word trong công việc này.
