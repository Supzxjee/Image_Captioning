# Root review đặc tả D/E

30/10/2026. Đã đọc [spec Astra](RELATION_KV_DE_PILOT_SPEC_2026-10-30.md), đối chiếu `captioning/models.py`, `concept_data.py`, `run_concept.py` và metadata CPU đã cache; không đọc ZIP lần hai, không Torch/HDF5/model/Kaggle.

Chấp nhận spec làm cơ sở triển khai pilot hẹp. D/E dùng cùng raw C0, hai decoder layers cuối và module mới, cùng loss/sampling; E chỉ chiếu value theo equivalent fact views. SCST+own-C0 KL giữ đường tới caption thật, BCE/MP chung không được nhận là đóng góp riêng E. Kiểm context và connectivity phải tránh PASS rỗng vì Wo=0. Caption reader-off vẫn giữ decoder đã học, không gọi đó là raw C0. Các gate là quyết định pilot, không bằng chứng thống kê hoặc bảo đảm bài báo.

## Correction tokenizer và OOV, trước khi triển khai

C0 là `word2idx`10000 với word tokens/`lower().split()`, **không BPE**. Từ “BPE” trong tin bàn giao Astra không phù hợp source; file spec dùng native tokenizer phải được hiểu đúng nghĩa này. Frozen input embedding là common technical amendment thay CLIP text, không novelty hoặc thử nghiệm đã có kết quả.

[Metadata audit](KV_NATIVE_TEXT_AUDIT.json):67unique fields; chỉ từ `drier` trong `hair drier` không có trong vocab. Hai fit facts bị ảnh hưởng, không có held fact OOV. Predicate contains/inside/touching đều có token. Fit58MPgroups có tối đa18facts/group, tổng259members; held16groups tối đa5, tổng46members. Đây là quy mô metadata, không forecast GPU.

Root ban đầu đề nghị fail mọi field OOV để tránh silent corruption. Sau khi biết một từ OOV nằm trong một cụm có token `hair`, root đề nghị và Astra đồng ý giữ **đúng native `vocab.get(word,<unk>)` cho mọi field**: không drop/rename fact, không sửa drier→dryer, không thêm CLIP encoder hoặc resize vocabulary. Ghi exact token IDs/OOV masks/rates và cảnh báo thông tin từ vựng bị mất; `<unk>` không phải nghĩa đầy đủ của hair drier. Mỗi field vẫn phải có ít nhất một token thường đã biết để tránh all-unknown field; empty/all-special/all-unknown field fail preflight, không tự bỏ mẫu. Policy giống D/E và cố định trước model. Đổi khuyến nghị vì bằng chứng mức OOV đã cụ thể, không vì kết quả mô hình.

## Bước tiếp theo

Chờ mount Dataset từ user-run notebook lưu version115, sau đó pin integrity và triển khai D/E theo spec đã review. D/E training notebooks chưa bàn giao và chưa CUDA/model validated; user tự chạy. Giữ cứng endpoint/budget/preflight, không âm thầm thêm source/quan hệ/loss/sweep khi pilot không đạt. P3 và unique-class cohort chưa chứng minh generic directional/multi-instance grounding. Không nhận semantic alignment thành công từ số fact hoặc identity của phép chiếu.
