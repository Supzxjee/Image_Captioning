# Chạy tiếp CONTROL/PROMPT bằng SCST + CE

**V2 sửa timeout chuẩn bị JPEG; người dùng yêu cầu chạy lại từ đầu.** Một lần quét thư mục, chỉ index detectorVALID,8luồng đọc/kiểmảnh, log tiến độ mỗi100ảnh; cap bước chuẩn bị30phút nằm trong cap tổng4giờ. Không thay mô hình/loss/lịch TRAIN. Output `_full1_v2` không đè V1. Nếu CONTROL V1 đã qua bước chuẩn bị và đang train bình thường, không cần dừng chỉ để đổi bản sửa I/O.

Hai notebook độc lập, mỗi notebook **T4 ×2**. Internet ON, Save Version → Save & Run All. Chỉ người dùng chạy; tác giả chưa chạy model hay notebook.

| Tài khoản | Notebook | Trọng số khởi đầu |
|---|---|---|
| Đức Anh / ducanh2403 | [CONTROL](CSA_PROMPT_SCST_CONTROL_FULL1_2GPU.ipynb) | best epoch10, version178 của `2-vi-cai-kia-qua-nhieu` |
| Ngân / ducanh3 | [PROMPT](CSA_PROMPT_SCST_PROMPT_FULL1_2GPU.ipynb) | best epoch9, version58 của `notebook7bc83c4c76` |

Notebook tải **chỉ best.bin của chính tài khoản đó**, kiểm đúng SHA; không cần tạo Dataset checkpoint trước. Bật Kaggle API Secret tên `codex` hoặc đổi `SECRET_NAME` ở cell đầu. Nếu đã có file exact best.bin trong Input, điền `WEIGHTS_ROOT`; khi đó không dùng API. Không thay bằng latest hay checkpoint PROMPT10.

## Gắn bốn Dataset

1. `/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134`
2. `/kaggle/input/datasets/ducanh2403/csa-prompt-full-karpathy-v175`
3. `/kaggle/input/datasets/vuthetam/mscoco-2014` — cần cả **ảnh JPEG gốc** và `dataset_coco.json` cho reranking.
4. `/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json`

Nếu mount khác, chỉnh đường dẫn ở cell đầu; giữ SHA và version nguồn. Chia sẻ quyền các Dataset riêng tư cho Ngân. Bank danh từ/động từ đã nằm trong checkpoint, không cần gắn bank130 riêng. Không cần chạy lại UPT hay cache175.

## Notebook thực hiện

- Nạp đúng CONTROL10/PROMPT9 và giữ adapter PROMPT đã học. Không train lại từ đầu.
- Kiểm CUDA, phụ thuộc, dữ liệu, CLIP assets, gradient/DDP trước lượt chính; cập nhật trong guard bị bỏ. Sinh lại ORIGINAL và bắt buộc replay CIDEr99,4587/98,8453 trước update1.
- Học **1 lượt113287ảnh TRAIN**, 14161update, global8, batch cuối7ảnh chia4/3. FreshAdam5e-6, CE+.05SCST, không thưởng đối tượng. Hai nhánh có cùng lịch ảnh trong giai đoạn này.
- ORIGINAL/END đều sinh5caption/ảnh trên **VALID5000**, giữ đủ ứng viên và điểm decoder. Đo CIDEr, BLEU4, ROUGE-L, CHAIRs/i, recall, NLL.
- Với cùng ứng viên, áp dụng lại .3decoder+.4CLIP+.3nhất quán đối tượng; không BLIP/tuning/oracle. Báo cả rank0 và rerank của ORIGINAL/END, không tự chọn checkpoint hoặc quay về mốc cũ vì điểm END kém.

Nguồn XE ban đầu C10/P9 có số epoch khác nhau; đây là so sánh hai pipeline đã chọn, không phải tổng ngân sách huấn luyện hoàn toàn bằng nhau. CHAIR chỉ kiểm đối tượng, chưa xác nhận đúng quan hệ. Không dùng TEST trong lượt này.

## Giữ output

Giới hạn **4 giờ theo đồng hồ mỗi notebook**, gồm chuẩn bị, guards, huấn luyện và đánh giá. Đây không phải ETA hay cam kết quota Kaggle. Lịch chạy có kiểm tra tốc độ và lưu checkpoint mỗi500update/khi dừng; không tự tăngcap hay chạy lại từ đầu.

Gửi:

- `csa_prompt_scst_CONTROL_full1_v2_review.zip`
- `csa_prompt_scst_PROMPT_full1_v2_review.zip`

Giữ toàn bộ thư mục output, nhất là `checkpoint.bin`, `checkpoint_receipt.json`, `source_xe`, `evaluations`, `training.jsonl`. Nếu chỉ reranking lỗi sau khi train xong, **giữ5caption/ảnh để chấm tiếp, không train lại**. Nếu interrupted, gửi ZIP trước; chỉ resume đúng nhánh/mã nguồn/dữ liệu/phần mềm từ output nguyên vẹn, không tự thay version.

## Đã kiểm và chưa kiểm

[Author checks](PROMPT_SCST_AUTHOR_CHECKS.json):10fixture NumPy/metadata/AST PASS, lặp lại trên đúng Git bundle của V2; cả hai notebook parse/emptyoutputs/regex/version/4h PASS;6hash mã mô hình trùng nguồn đã train. Không có tác giả chạy Torch, CUDA, scorer, API hay notebook. Guards model/DDP và tốc độ thực tế vẫn cần người dùng chạy.
