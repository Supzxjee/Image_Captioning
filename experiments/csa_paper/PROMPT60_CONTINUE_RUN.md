# PROMPT60: học tiếp thêm một epoch

Import `CSA_PROMPT60_SCST_CONTINUE_EPOCH2_2GPU.ipynb` vào notebook của **Ngân**, chọn GPU T4×2, bật Internet, Save & Run All. Token secret `codex` phải của tài khoản có quyền đọc `ducanh3/notebook7bc83c4c76/60`. Nếu gắn sẵn file checkpoint đúng SHA thì điền `WEIGHTS_ROOT`, không cần API.

Gắn đủ bốn Dataset như lượt trước:

- `/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134`
- `/kaggle/input/datasets/ducanh2403/csa-prompt-full-karpathy-v175`
- `/kaggle/input/datasets/vuthetam/mscoco-2014` — cần JPEG gốc và `dataset_coco.json`
- `/kaggle/input/datasets/ducanh2403/objectdetectionecache`

Notebook tải duy nhất checkpoint đúng version60, kiểm SHA và source identity, khôi phục model/Adam/RNG. Không cần checkpoint XE58/bank130 riêng. Input của người dùng phải có quyền truy cập trên tài khoản chạy.

Thêm tối đa14161update trên113287ảnh TRAIN; giữ kiến trúc, CE+.05SCST và fixedrerank .3/.4/.3. Xoay thứ tự ảnh seed43 và CE reference từ lượt trước. Chấm ORIGINAL/MID/END trên VALID5000, giữ bản có CIDEr sau rerank cao nhất nhưng CHAIRs/i không tăng và recall không giảm so với ORIGINAL. Có fallback về source60. Không TEST hoặc tuning trọng số.

Giới hạn4giờ đồng hồ notebook, dựa trên ngân sách13giờ GPU người dùng báo; không coi cap là thời gian dự báo hay số giờ quota tính phí. Giữ output nếu incomplete. `RESUME_ROOT` chỉ dùng output cùng recipe nếu người dùng chủ động tiếp tục; không tự restart.

Gửi `csa_prompt_scst_PROMPT_epoch2_v1_review.zip`. Giữ toàn bộ folder cùng `mid.bin`, `checkpoint.bin`, `best.bin`, `locked_selection.json`. Nếu best là original thì không có cải thiện mới và best giữ đúng binary source60. Các checkpoint mới có cursor tính theo **update thêm**, không cộng14161update đã học trong source.

Tác giả chỉ kiểm NumPy/metadata/AST/mock; actual policy replay, gradient, model/CUDA/DDP, metric parity và throughput do notebook chạy trước/trong lượt huấn luyện. Chưa có kết quả mới và chưa sửa báo cáo Word. [Protocol](PROMPT_SCST_CONTINUE_PROTOCOL.md).
