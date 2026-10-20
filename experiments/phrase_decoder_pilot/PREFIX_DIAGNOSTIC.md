# Kiểm tra cặp mục tiêu và prefix — READY_NOT_RUN

Mở `PAIR_PREFIX_DIAGNOSTIC_2GPU.ipynb`. Một notebook dùng hai T4: GPU0 đánh giá CE, GPU1 đánh giá CROSS độc lập. Không training, optimizer, SCST hoặc thay đổi weights.

## Input

Giữ ba input cũ: checkpoint nền preserved SHA5f8eead3…, visual-cache, MSCOCO2014 JSON+ảnh. Thêm hai adapter từ output của hai notebook đã chạy:

- CE: `/kaggle/working/phrase_decoder_ce_21/adaptation.bin`
- CROSS: `/kaggle/working/phrase_decoder_cross_21/adaptation.bin`

Nếu tạo session mới, upload hai file đó thành dataset, có thể đổi tên `adaptation_ce.bin` và `adaptation_cross.bin`. Sửa `ADAPTER_CE` / `ADAPTER_CROSS` trong cell đầu thành đường dẫn Kaggle thực. Hai ZIP review không chứa adapters. Không dùng checkpoint F, checkpoint nền hay file khác thay adapter. Metadata, trainable key set, shape/dtype, original source checkpoint và complete reconstructed model hash được kiểm tra.

## Phép kiểm tra

Giữ ảnh, patch features và global G cố định; chọn lần lượt hai cặp người–vật thật trong mỗi ảnh. Dùng bốn nhóm same-image đã khóa: hai holdout (70642,208095), hai fit controls (không dùng fit để suy khái quát).

Mỗi arm/cặp có ba conditions:

1. **Natural:** cặp thật với boxes, membership, mask và geometry nhất quán.
2. **Evidence off:** nhánh token vùng bị mask như pilot trước; chỉ đo độ nhạy, không phải input tự nhiên tương đương.
3. **Explicit geometry swap:** tráo riêng vector geometry8D giữa hai event, giữ vùng/membership; không nhất quán và ngoài phân phối. Mask/membership vẫn chứa hình học, nên không thể kết luận đã loại hết hình học hoặc cô lập hoàn toàn visual.

Đầu ra: greedy từ BOS; tiếp tục sau `a person`; tiếp tục sau prefix chung thực của hai mô tả; top10 từ tiếp theo, logP/rank hai từ đầu tiên phân biệt mô tả, và logP từng token teacher-forced của cả hai câu. Với bốn nhóm này, prefix chung thực đều là `a person`. Không ép chọn từ hai đáp án: ghi cả cạnh tranh từ vựng đầy đủ qua top10 và greedy.

**Trước khi diễn giải:** cần tái tạo đúng initial/full-final model hashes, staged feature receipts, natural/off greedy và scores của pilot (rtol/atol1e-5). Không bỏ assertions để chạy tiếp. Weights được đóng băng và kiểm tra hash không thay đổi sau đánh giá. Checkpoint metadata liên kết tới protocol/manifest gốc, không chỉ tin tên file.

## Quyết định sau khi đọc kết quả

- Đổi cặp làm lựa chọn từ đúng nhưng BOS-greedy sai: có tín hiệu đọc được trong điều kiện prefix đó; chưa chứng minh exposure bias hoặc routing là giải pháp. Ưu tiên kiểm tra đường sinh.
- Teacher-forced ranking đúng nhưng quyết định từ vẫn sai: scoring không thay thế generation; chưa scale CROSS.
- Đổi cặp không có phản ứng đúng: chưa tách được encoder/bộ đọc; không xây routing trên niềm tin rằng biểu diễn đã đúng.

Không có nhãn semantic tự động. Đối chiếu ảnh+box, chấp nhận paraphrase, tách câu đúng nhưng thiếu chi tiết khỏi câu sai và nhầm instance. Chỉ hai nhóm holdout đã xem: đây là chẩn đoán ca cụ thể, không test mới hay ước lượng population.

## Trạng thái và gửi kết quả

Local: 3 pure-data tests PASS; AST/payload roundtrip, notebook chưa thực thi, kích thước <1MB, không gọi backward/optimizer step PASS. Local không có torch: chưa chạy synthetic CPU model tests, adapter restoration, replay hay inference. Notebook bắt buộc synthetic CPU tests và các guards thực trên Kaggle; không tải pretrained CLIP/không cài lại torch.

InternetON, chọn T4×2, Save & Run All. Chỉ stage lại 38 ảnh để dùng đúng preparation cũ; chỉ đánh giá8events same-image. Gửi `/kaggle/working/pair_prefix_diagnostic_4_review.zip`. ZIP chứa JSON/HTML/ảnh+box/source/log, không chứa weights. Nếu lỗi sau prepare, finally vẫn tạo review ZIP để đọc log; output thiếu không được coi là thành công.
