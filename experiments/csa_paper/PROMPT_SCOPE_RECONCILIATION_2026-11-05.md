# Khôi phục ràng buộc prompt đối tượng–quan hệ của đề tài

05/11/2026. Người dùng cung cấp lại nội dung đề cương: khai thác nhãn đối tượng và bộ ba quan hệ **của ảnh đầu vào**, xây dựng prompt, dùng mô hình thị giác–ngôn ngữ tiền huấn luyện để biểu diễn ngữ nghĩa, hợp nhất với đặc trưng thị giác, sinh nhiều caption rồi đánh giá/rerank và kiểm soát hallucination. Đây là ràng buộc phạm vi, không phải một tùy chọn có thể bỏ vì CSA đã có bank. Hai đề tài cùng lab cần khác hướng; không tiếp tục Fact-RAG từ caption ảnh khác của Tâm.

## Sửa nhận định trước

Bank danh từ/động từ TRAIN trong CSA không thay thế prompt đối tượng–quan hệ có cấu trúc riêng từng ảnh. Mã `model.py:143–154` nhận regions/valid và noun/verb bank, không nhận prompt UPT/các bộ ba riêng ảnh. CSA có căn chỉnh, nhưng bank động từ không xác định rõ cặp subject–object; không mô tả pipeline117.0633 hiện tại như đã hoàn thành nhánh prompt của đề cương. Bản này là mốc so sánh CSA+SCSTCE+rerank hữu ích, cần giữ nguyên weights/results và nhãn phương pháp.

Đề xuất image-conditioned noun aggregation trước đó chỉ giải quyết một phần chọn bank; nếu không dùng prompt đối tượng–quan hệ thì vẫn thiếu phạm vi. Rút nó khỏi vị trí đề xuất hoàn chỉnh cho đề tài, có thể giữ như biến thể phụ sau này.

## Hướng kiến trúc cần thảo luận tiếp

Ảnh → ROI/features và detector/HOI (tận dụng UPT cache hiện có nếu kiểm được schema/coverage/provenance) → nhãn đối tượng+cặp box+quan hệ dự đoán → prompts riêng ảnh → frozen CLIP text encoder → căn chỉnh prompt với vùng/cặp vùng tương ứng → hợp nhất theo kiến trúc kế thừa CSA → Transformer Decoder →5caption → fixed ảnh–caption/object-consistency reranking hiện có.

Object prompt ví dụ `a horse` đi kèm object box; relation prompt `a person riding a horse` đi kèm subject/object box và confidence. Giữ bộ ba cùng ID cặp, không chỉ lấy động từ `ride`, không tách prompt rồi mất vai trò. Đề xuất ưu tiên bàn tiếp là dùng embeddings riêng ảnh làm đầu vào ngữ nghĩa có cấu trúc cho các nhánh CSA thay vì coi global banks là đủ; vị trí thay/mix bank, aggregation/attention/gate và xử lý prompt rỗng cần khóa trước khi code. Gắn box/cặp box là ràng buộc căn chỉnh đề xuất, chưa có triển khai hoặc kết quả chứng minh.

CLIP text encoder tạo biểu diễn ngữ nghĩa từ prompt, không tự phát hiện quan hệ đúng hay tạo tri thức đã được xác nhận. Confidence UPT không phải xác suất đã calibration; UPT HOI không bao phủ mọi quan hệ. Không biến prediction/box thành nhãn người đã kiểm; không dùng reference captions hoặc GT triples của ảnh mới làm đầu vào inference. Prompt không đủ tin cậy cần fallback đã định trước, không suy diễn thêm quan hệ từ caption tham chiếu.

## Những phần đã có và còn thiếu

| Thành phần | Trạng thái |
|---|---|
| Baseline Transformer không CSA, đối chứng CSA | Đã có thực nghiệm TEST, ghi đúng budget/cấu hình |
| CSA+SCSTCE+fixedrerank | Đã có TEST117.0633/CHAIRs4.40%, mốc so sánh chưa tích hợp prompt UPT |
| UPT/full prompt cache | Có lịch sử artifacts/Kaggle paths; kiểm lại schema/hash/coverage trước dùng, không tự chạy lại cache |
| Prompt riêng ảnh → CLIP → căn chỉnh vùng/cặp vùng → CSA | Chưa chốt thiết kế, chưa triển khai trên CSA hiện tại |
| Kiểm soát ảo giác đối tượng | Có CLIP/object rerank và CHAIR evaluation |
| Kiểm sai thuộc tính/hành động/quan hệ | Chưa được chứng minh bởi CHAIR/object consistency; cần tiêu chí riêng nếu tuyên bố |
| METEOR/SPICE/prototype web theo đề cương | Phải kiểm kê riêng; không tự nhận hoàn tất từ bảng metric hiện có |

## Quyết định phạm vi và giới hạn

Ưu tiên đảm bảo prompt thật sự đi vào forward của captioner, có liên kết vùng/cặp vùng và có đối chứng tắt/hoán đổi prompt, thay vì chỉ ghi prompt trong sơ đồ hoặc dùng ở reranking. Mốc117.06 được giữ nguyên, không gán cho kiến trúc mới. Hướng mới có thể kế thừa CSA và CLIP; không hứa novelty/gain. Học tiếp checkpoint hay từ đầu, loss, cohort/update/budget và đánh giá VALID trước TEST cần thảo luận và được duyệt trước; chưa có quyền chạy mô hình/notebook hoặc sửa trainer theo ghi chú này.
