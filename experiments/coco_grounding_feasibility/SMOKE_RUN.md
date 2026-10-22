# COCO noun–patch routing: smoke interface, T4×2

Chạy **COCO_ROUTING_INTERFACE_SMOKE_2GPU.ipynb**, Internet ON, GPU **T4×2**, **Save & Run All**. Người dùng tự chạy. Notebook đã tạo nhưng **chưa chạy PyTorch/GPU ở máy local**, chưa có kết quả runtime.

## Bốn dataset cần gắn

1. `qformer-scst-object-w05-legacy`: `/kaggle/input/datasets/ducanh2403/qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth`.
2. `prompt-cache`: `/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt`.
3. `visual-cache`: `/kaggle/input/datasets/ducanh2403/visual-cache`.
4. `vuthetam/mscoco-2014`: `/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json`.

Nếu tài khoản khác có mount khác, chỉ sửa input paths ở cell đầu. Không thay original bằng checkpoint A-end/CSA/GV. SHA checkpoint, legacy cache và COCO JSON được kiểm trước. Không cần detector cache, UPT, VG, ảnh gốc hoặc annotation đầy đủ.

## Phạm vi

**4 ảnh TRAIN** đã có trong audit, COCO IDs 64627,529017,468966,580507; hai worker độc lập, mỗi GPU hai ảnh, không DDP collectives. Chỉ lấy bốn dòng feature/legacy và một caption/span mỗi ảnh. Mask bbox fractional union đã đóng trong notebook; dùng cho số học, không được coi là human grounding gold.

**0 optimizer updates.** Không SCST, không training CE, không lưu checkpoint, không evaluation CIDEr/CHAIR/dev/test. Caption greedy chỉ để kiểm parity với baseline trên chính batch này.

## Cách quan sát attention

Giữ nguyên tất cả lệnh MHA `need_weights=False` và output của baseline. Một pre-hook không thay args/output tái tính xác suất cross-attention từ **chính input Q/K và trọng số projection hiện có**, trong FP32 eval, không dropout. Đây là graph phụ dùng chung tham số, không có head hoặc tham số mới. Nó không phải tracing đầy đủ residual/legacy/CLS.

Trước khi kiểm checkpoint thật, synthetic CPU và CUDA tests đối chiếu xác suất per-head với public PyTorch MHA `need_weights=True`, cả bool/additive padding. Tolerance cố định **rtol=1e-5, atol=1e-6**, không tự nới khi FAIL. Forward thật không bị đổi sang kernel trả weights; vẫn phải kiểm logits/memory và greedy parity.

Routing dùng head means ở cross-attention tầng cuối decoder và Q-Former. D slice Q vẫn nằm trong phân bố toàn `[legacy;Q]`, C spatial slice vẫn nằm trong phân bố CLS+196 patch. Không renormalize. Support bằng raw path mass trong fractional bbox mask; epsilon=1e-8. Notebook ghi riêng decoder→Q, patch-path, CLS-path và bbox masses.

## Những gì phải PASS

- Synthetic attention khớp public MHA và pad weights bằng zero.
- Memory/mask/logits khớp baseline trong tolerance đã khóa; greedy token IDs giống hệt.
- Không thêm/thay parameter hoặc buffer; state hash trước/sau giống nhau.
- Full patch mask có bbox support bằng patch-path mass; empty mask bằng zero; path patch+CLS bằng Q mass.
- Loss routing truyền gradient hữu hạn, khác zero đến decoder cross-attention, Q-Former cross-attention, visual projection và learned queries. CLIP backbone không được load ở cached path, cache input không nhận gradient. Không suy rộng thành kiểm backward của một CLIP backbone đang chạy.
- Supervision index là vị trí **dự đoán** noun. Đổi noun/future tokens trong teacher input không thay logits/attention của vị trí dự đoán trước nó (kiểm trên token đầu span).
- Không optimizer, không thay trọng số. Chỉ `backward` một lần để đo gradient.

Một synthetic gradient test không thay thế gradient checkpoint thật. Parity của graph quan sát cũng chưa chứng minh causal attribution hoặc lợi ích caption. Runtime/memory chỉ áp dụng batch nhỏ và chưa phải upper bound cho training.

## Kết quả cần gửi

`/kaggle/working/coco_routing_interface_smoke_review.zip`

ZIP chứa protocol, source, stage receipt, runtime, synthetic checks, completion/gradient/mass reports hai GPU và notebook log. Không kèm model, cache hay tensor stage. ZIP được tạo cả khi lỗi sau bootstrap; nếu lỗi từ cell đầu/bootstrap thì gửi traceback cell đó. Muốn thấy lỗi gốc: `print(LOG.read_text(encoding='utf-8')[-16000:])`.

**PASS mới xem xét lấy full COCO annotations và audit noun spans tiếp.** Không tự chạy pilot sau smoke. Nếu FAIL, giữ output để sửa đúng implementation, không nới tolerance hoặc thay dataset cho qua.

## Kiểm tra local đã thực hiện

Chỉ parse source/cells, kiểm notebook chưa chạy và <1MB, so khớp embedded source, mask/ID/token offsets và việc không có optimizer update. Các test torch, numerical parity, gradient, greedy và GPU memory **đều chờ người dùng chạy Kaggle**.

Astra đã review **tóm tắt thiết kế**, không đọc code hoặc chạy model: không thấy lỗi thiết kế nghiêm trọng, đồng ý giao notebook có điều kiện chờ runtime PASS. Gradient của graph quan sát chỉ xác nhận loss có đường tối ưu qua các tham số/đầu vào dùng chung; không chứng minh decoder đã dùng đúng vùng khi sinh caption.
