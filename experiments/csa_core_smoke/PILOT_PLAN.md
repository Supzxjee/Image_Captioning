# Pilot kế tiếp — quyết định Astra sau smoke

Ngày2026-10-22. Trạng thái: **RECIPE_PROPOSED; NOTEBOOK_NOT_IMPLEMENTED**. Người dùng đã cung cấp smoke review; file này lưu quyết định nghiên cứu tiếp theo, không phải kết quả chạy.

Giữ core2object+1relation, bank gốc frozen,32QFormerqueries,readout `[M_A;Vo;Vr]`. Khởi tạo mới checkpoint gốc của A và core seed42, không lấy trạng thái sau smoke. Không thêm gate, loss hoặc nguồn dữ liệu.

| | Pha1: thích nghi core | Pha2: học chung |
|---|---|---|
| Data |Train10k cũ |Cùng train10k |
| Updates/globalbatch |1250/8 |1250/8 |
| Cập nhật |Chỉ core CSA |Core và các tham số baseline mà A cập nhật |
| LRbase |Frozen |5e-6 |
| LRcore |Linearwarmup100steps lên1e-4 rồi giữ |2e-5 cố định |
| Loss |CaptionCE như A |CE200steps rồi CE+.05SCST |
| Reward |Không |CIDErD+.5object_count_v2 |

AdamW options theo A,globalnormclip1,evalmode nhưng gradients bật. Freeze decoder pha1 không được no_grad toàn decoder: vẫn cần gradient về core. Cuối pha1 kiểm baseline/bank state không đổi, lưu checkpoint phục vụ tiếp tục khi gián đoạn; chưa evaluate dev để chỉnh lịch. Pha2 tạo optimizer mới cho cả baseline/core, reset RNG/sampler theo lịch pha2 của A từng rank. Ghi norm core/base riêng và hệ số globalclip.

Lý do chia pha: cho core23Mtham số thích nghi với decoder cố định trước khi cập nhật pretrained. Đây là **giả thuyết recipe**, không phải nguyên nhân lỗi trước đã được chứng minh hoặc LR tối ưu.

Chỉ checkpoint cuối pha2 là kết quả chính. Rawgreedy dev1000 cũ: original,CSA-on,endCSA-off; cùng IDF/evaluator. Không test/rerank/chọn bestcheckpoint theo dev. Off sau train không phải baseline A. Có thể dùng A cũ khi source/checkpoint/dataorder/tokenizer/cache/objective/optimizer/runtime/evaluator phần chung khớp. Ngân sách CSA lớn hơn A thêm1250core-onlysteps; đây là so hệ thống, chưa cô lập tác dụng thuật toán khỏi tham số/tính toán.

Gate soA:ΔCIDEr≥.01,paired95%lower>0; vượtoriginal; CHAIRs/i không tăng, recalldrop≤.01. Nếu đạt mới review factuality nhỏ và mù trước quyết định xác nhận. On/off chỉ chẩn đoán; gainA mà on không tốt hơnoff chưa chứng minh memory hữu ích khi suy luận.

Nếu chạy hợp lệ không đạt gate, dừng recipe, chưa fulltrain hoặc tự tăngsteps.2500updates là ngân sách sàng lọc, không bảo đảm hội tụ core. Đo riêng staging/training/evaluation; smoke không bảo đảm tốc độ/peak mọi batch.
