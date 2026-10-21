# Lõi CSA trên nền Q-Former — kiểm chứng kỹ thuật

Chạy [CSA_CORE_SMOKE_2GPU.ipynb](CSA_CORE_SMOKE_2GPU.ipynb) trên Kaggle **T4×2, Internet ON, Save & Run All**. Người dùng tự chạy; chưa có kết quả GPU hay caption mới.

## Dataset cần gắn

Giữ các path trong cell đầu hoặc sửa theo mount của tài khoản:

1. `qformer-scst-object-w05-legacy`: checkpoint khởi đầu đã dùng cho A, **không phải checkpoint G/H/GV**.
2. `prompt-cache`: `prompt_clip_tokens_cache.pt` legacy của baseline.
3. `visual-cache`: CLIP ViT-B/16 last_hidden_state `(197,768)` đã cache.
4. `vuthetam/mscoco-2014`: chỉ cần `dataset_coco.json`, không chạy CLIP trên ảnh gốc.
5. `objectdetectionecache`: `objectdetectioncache.json`, phục vụ object reward giống A.

Không cần UPT full/hidden cache, UPT prompt cache hoặc VG. Hai bank và source CSA (~23,5MB bank) tải trực tiếp từ tác giả, kiểm hash. CHAIR cache chỉ phục vụ nhãn đối tượng TRAIN; không chạy CHAIR evaluation.

## Phạm vi

Theo [quyết định nguồn](../csa_inheritance_audit/CSA_INHERITANCE_FEASIBILITY.md), dùng source [CSA-Net commit 0cc570b](https://github.com/CrossmodalGroup/CSA-Net/blob/0cc570ba7c50ef08c46a21439fe73ef0a5df97ab/models/caption/csa_net.py). Giữ nguyên AST các phép tính forward, initializer, raw-dot similarity, nhiệt độ, 2 object stages +1 relation stage. Chỉ bridge hai lệnh load bank về CPU và đăng ký tensor dưới dạng buffer. Không vendor mã/bank tác giả vào repo.

`G=memory_baseline[:,-32:]` là32 latent Q-Former queries, không phải detector regions. Đầu ra `[Vo,Vr]` ghép **sau nguyên memory baseline**. Mask của hai nhóm64token đều valid. `rel_mask` kỳ lạ trong source tác giả chỉ giữ để kiểm parity, không dùng làm padding. Bypass bỏ hẳn module/tokens mới, phải khôi phục memory và logits baseline.

### Thứ tự kiểm chứng

1. CPU, float32, eval: output/mask và tất cả parameter/input gradients đối chiếu author reference với adapter, N=32 và150, B=2, `rtol=1e-4, atol=1e-6`. Đây là kiểm math-preserving loader/buffer adaptation **cùng author forward**, không phải hai implementation độc lập và không xác nhận paper equations.
2. Bank hash/shape/type/finite, không nhận gradient, có trong state_dict và di chuyển sang CUDA đúng. Kiểm append/mask/gradient trên tensor giả.
3. Checkpoint thật trên mỗi GPU: bypass memory/logits đúng trong tolerance;8ảnh TRAIN theo hash cố định, mỗi rank4ảnh,5references/ảnh. Không stage11000ảnh, không evaluate dev/test.
4. DDP:2CEsteps rồi2CE+.05SCSTsteps, rewardCIDErD+.5object_count_v2 và full TRAIN IDF như A. Không dropout, FP32/strict numeric profile. LRbase5e-6, core1e-4 chỉ là **probe vận hành**, chưa chốt recipe hoặc ngân sách pilot. Mỗi step đo loss/gradient/time/peak allocated+reserved;2ranks phải đồng bộ core sau cập nhật, banks không đổi.

Giữ cùng8ảnh qua4steps để đo chi phí và đường gradient, không dùng loss này chứng minh khả năng học/cải thiện. Không lưu checkpoint mới để tránh dùng nhầm model chỉ chạy smoke. Peak memory là phép đo nhỏ có first-step optimizer allocation và warm-up; chưa phải upper bound cho mọi batch/chiều dài caption.

## Gửi kết quả

`/kaggle/working/csa_core_smoke_review.zip` có protocol, source, parity, train8IDs, object targets, runtime, history2ranks, completion và notebook.log. Bank/model/cache không đóng trong reviewZIP. Nếu lỗi trong giai đoạn test/run, ZIP vẫn được tạo; nếu lỗi dependency/bootstrap, gửi log cell tương ứng.

Không tự động chạy pilot dài sau PASS. Dùng kết quả này để chốt tính khả thi và recipe với Astra. Chưa gọi đây là CSA-Net toàn hệ thống, phương pháp mới hoặc bằng chứng grounding.
