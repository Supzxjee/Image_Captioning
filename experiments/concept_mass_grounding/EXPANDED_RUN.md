# Chạy A/C0/C trên val_extra4000_cg

## 1. Lưu checkpoint trước — CPU

Mở `SAVE_CONCEPT_C_C0_V111_V112_DATASET.ipynb` trên Kaggle. Chọn Accelerator=None, Internet ON. Bật Secret `KAGGLE_API_TOKEN`; nếu secret tên `codex`, sửa `SECRET_NAME` ở cell đầu. Không dán token vào code/chat. OWNER mặc địnhducanh2403 phải thuộc token có quyền xem output.

Save & Run All. Notebook cài CLI commit đã kiểm version routing, thử mock requestv111/v112 trước network; chỉ tải `concept_mass_C_v2/checkpoint.bin` từ111 và `concept_mass_C0_v2/checkpoint.bin` từ112. Kiểm hai fullSHA rồi copy nguyên byte thành `C.bin`, `C0.bin` và `transport_receipt.json`. Không load model, không tự unzip torchweights, không tải source/cache hoặc dùng latest. Dùng license identifier`other` cho private research weights và giữ quyền upstream. Dataset creation chỉ chạy khi người dùng chạy cell; không ghi đè dataset cũ.

Dataset mặc định `ducanh2403/concept-mass-c-c0-v111-v112`. Nếu status đang processing/không đọc được, xem trang dataset; không chạy create lặp lại. Nếu slug/title tồn tại, giữ nguyên output và kiểm tra trước khi đổi tên để tạo dataset khác. Nếu đổi SLUG/OWNER, sửa WEIGHTS tương ứng ở bước2.

## 2. Đánh giá — T4×2

Mở `CONCEPT_MASS_EXPANDED_EVAL_4000_2GPU.ipynb`. Add Input:

- Dataset weights vừa tạo: `C.bin`, `C0.bin` ở gốc.
- `pair-source-fullframe-ce` có checkpoint A gốc.
- `prompt-cache`, `visual-cache`, `vuthetam/mscoco-2014` có `dataset_coco.json`.

Không cần detector, UPT cache hoặc trích lại CLIP từ RGB. Bật Internet và T4×2; kiểm path ở cell đầu, Save & Run All. User báo14hGPU cho tài khoản chạy. ETA50–100phút nếu I/O tương tự; đó không phải cam kết. Parent cap8400s(2h20), GPU phase1800s, stage3600s với120sclosegrace, metrics1800s. Accelerator bật trong chuẩn bị/metrics vẫn có thể tiêu quota. Phân biệt walltime với device-hours, không hứa cách Kaggle tính quota.

Stage4016rows:4000ảnhprimary đã khóa và16ảnhcurrentdev chỉ kiểm historical parity. Không dùng tên historicalremainder4000 vì cohort ấy còn797currentdevIDs. Không loại/refill ảnh khi cache thiếu. HDF5 được đánh dấu complete chỉ khi copy đủ; timeout giữ journal và logs, báo incomplete, không tự sửa cohort hoặc cap.

MỗiGPU chạy2000primaryimages cho mỗi model, không NCCLcollectives/optimizer/backward. Load A trước, strict-load trọng số C/C0, xác minh wrapperdigest và fullfileSHA đã khóa. TeacherA chỉ giữ để kiểm identity đúng như pilot, không dùng để đổi caption. Nativegreedy tối đa29next-tokensteps+BOS, batch8/rank, eval/no-grad; không đổi logits. Sentinel captions phải khớp historicalA/C/C0 trước primary. Weight digest phải giữ nguyên sau inference.

CPUmetrics officialCIDEr/BLEU/METEOR/ROUGE/fullCHAIR80; lưu per-imagecounts và score để pairedbootstrap20000draws(seed20261030). AggregateCHAIRi/recall là tỷ số tổngcounts; exactMcNemar cho sparseerrors. Giữ jointgate/margins của protocol, không chọn metric thuận lợi. Không báo pooled5000 trong notebook này, không trộn IDF hai tập. Không thêm semanticdiagnostic.

Gửi `/kaggle/working/concept_mass_expanded_eval_4000_review.zip`. Không có checkpoint mới vì không training. Nếu lỗi, ZIP chứa execution_status=false và log; không coi là kết quả method. Giữ datasetweights cho lần xác minh, không rerun training.

## Kiểm tra trước khi giao

13checksPASS:8metadata/statistics/transport và5notebookstatic/mockflow. Transport test dùng fakebytes, không load checkpoint. Notebook code compile và pinnedGitassets/hash verified, mỗi notebook dưới15KB. Không model/Torch/HDF5/API/Kaggle được chạy local bởi assistant. Actual API/upload, cache staging, GPU caption parity và metrics end-to-end chờ người dùng chạy. Khung model/decoding cũ được tái sử dụng, không nhận staticcheck là GPU PASS.

Thực nghiệm vẫn là one-seed/post-pilot/reusedvalidation. Đánh giá này không tự tạo novelty hoặc bảo đảm giảm hallucination. Không training/sweep/tuning tiếp trên4000ảnh nếu jointgate không được hỗ trợ.
