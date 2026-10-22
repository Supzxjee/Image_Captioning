# Audit interface attention hiện tại

Tài liệu này ghi **audit source trước smoke**: chỉ đọc source, không chạy torch/forward/backward/parity; không thay `captioning/models.py`. Sau đó người dùng đã chạy notebook và đạt technical PASS trên bốn ảnh; xem `SMOKE_RESULT.md`. Các câu “chưa xác minh runtime” bên dưới mô tả trạng thái audit ban đầu, không phủ nhận receipt đã nhận.

## Đường dữ liệu đã xác định

Trong cấu hình A/original, `UniversalVisionEncoder(load_backbone=False, visual_adapter='qformer', num_visual_queries=32, qformer_layers=2)` nhận cache CLIP `(B,197,768)`. `vis_projection` → ReLU/dropout → hai `VisualQueryLayer` tạo 32 query chiều 512. Toàn bộ 197 token, **gồm CLS**, đi vào Q-Former; không slice bỏ CLS ở đường này.

`prompt_to_visual_attn` dùng prompt làm query và 32 visual query làm key/value, rồi gate + residual prompt + layernorm tạo grounded legacy memory L. `ImageCaptioningModel.build_memory` nối **`[L;Q]`**, mask prompt padding, tất cả 32 Q hợp lệ. Đây không phải decoder chỉ đọc 32 Q.

`CaptionDecoder` dùng ba `nn.TransformerDecoderLayer`; logits vị trí j dự đoán target token j+1 (BOS ở vị trí 0). Danh từ nhiều từ có nhiều vị trí dự đoán, không phải một token CLIP. Caption tokenizer dùng whitespace, giữ dấu câu trong raw token, max length 30; regex tách mọi dấu câu rồi dùng chỉ số mới sẽ sai vị trí.

## Những gì interface đang thiếu

- Q-Former self/cross attention gọi `need_weights=False`, bỏ output weights.
- Prompt→Q attention cũng gọi `need_weights=False`.
- Decoder trả logits, không trả từng layer/head cross-attention; dùng `nn.TransformerDecoder` tiêu chuẩn. PyTorch runtime chưa được mở/chạy để kiểm chứng chính xác backend/kernel hoặc cách hook.
- Chưa có return path thống nhất cho decoder→Q và Q→patch còn gradient. Không có bằng chứng runtime rằng đổi sang `need_weights=True` giữ nguyên forward/RNG/numerical outputs.

Source cho thấy Q-Former/caption decoder không nằm trong `no_grad`; chỉ bước extract CLIP backbone được đóng băng. Có tiềm năng lấy attention có gradient bằng instrumentation nhỏ, **chưa xác minh được parity, gradient hay peak memory**. Không gọi đây là PASS kỹ thuật.

## Vì sao chưa duyệt phép nhân đơn giản

1. Nếu chỉ lấy decoder attention trên Q rồi chuẩn hóa lại, có thể che việc decoder chủ yếu đọc L. Phải giữ/đo khối lượng attention qua cả hai nhánh, không coi conditional attention là toàn bộ evidence.
2. Q còn đọc CLS. Bỏ CLS rồi renormalize có thể che đường global information tương tự. Cần ghi riêng CLS mass.
3. Hai tầng Q-Former có query self-attention, residual và FFN; ba tầng decoder cũng có residual và tự attention. Nhân hai attention cuối tầng không phải truy nguyên đầy đủ tất cả đường thông tin.
4. L chứa residual text và phần đã đọc từ Q. Không tự quy toàn bộ L về patch bằng ma trận prompt→Q.
5. CLIP spatial tokens đã contextualized toàn ảnh. Box-mass loss là ràng buộc routing, không chứng minh causal grounding, đúng instance hoặc caption gain.

## Phán quyết của audit source

**CONDITIONAL: phải chốt định nghĩa routing trước; chưa sẵn sàng triển khai loss/pilot.** Không thấy yêu cầu đổi kiến trúc lớn chỉ để quan sát attention, nhưng chưa chứng minh instrumentation không đổi baseline. Nếu phải tạo một head độc lập thì không còn đúng giả thuyết giám sát đường caption hiện có.

Một bước kỹ thuật sau này, nếu Astra chốt và nguồn box đầy đủ, phải có zero-loss forward parity, mask/order/CLS mass checks, gradient đến đúng modules và log khối lượng trên L/Q; user chạy trên Kaggle. Audit hiện tại không thực hiện bước đó.

## Nguồn

- `captioning/models.py`: VisualQueryLayer 12–37; LightweightQFormer 40–73; UniversalVisionEncoder 193–386; CaptionDecoder 401–430; ImageCaptioningModel 432–504.
- `captioning/tokenizer.py`, `captioning/data.py`, `experiments/upt_hidden_pair/run_hidden.py` (build_model) và `hidden_model.py` (A direct-memory).
- SHA256 source models.py hiện tại: `3cfbaef1f1f4e01b6621584a172234cc7e82739b86684f6bea612638d603d072`.

Các line number là source tại thời điểm audit; không tuyên bố đã độc lập kiểm weights hay runtime lịch sử.
