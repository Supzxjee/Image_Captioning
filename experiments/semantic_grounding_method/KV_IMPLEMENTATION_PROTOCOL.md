# Triển khai pilot D/E đã được duyệt

30/10/2026, sau khi user xác nhận Dataset mount:
`/kaggle/input/datasets/ducanh2403/relation-kv-preparation-v115`.

Nguồn khoa học là `RELATION_KV_DE_PILOT_SPEC_2026-10-30.md` và root review, không thay ontology/cohort/loss/endpoint theo kết quả. Chuẩn bị hai notebook, mỗi notebook một arm dùng T4x2, user tự chạy. Không local model/Torch/autograd hoặc tự start Kaggle. Static/NumPy/mock được phép.

Các lựa chọn triển khai cụ thể: FP32/deterministic như runtime C0, manual gradient all-reduce/average từ hai local caption batches4 (global8). Auxiliary BCE split4/rank; mỗi rank tính cùng full MP group, gradient averaging giữ đúng một group loss. Không detach denominator, không unknown negatives. Raw input H5 được đọc trực tiếp từ Dataset; không tạo lại feature cache, GT chỉ supervised auxiliary scores và reward/evaluation. Predicate embedding từ frozen native C0 word table; UNK native giữ trong mean, fail predicate-OOV/field-empty/all-unknown hoặc OOVset khác audit. Main4000 caption order giữ nguyên. Endpoint500, không chọn checkpoint.

CPU prepare xác minh8payload/full H5 SHA và input/checkpoint/detector/dataset/CHAIR hashes, xây training-only object reward với helper gốc. Nguồn object reward cần detector gốc (label-neutral tại confidence>=.9); relation bank vẫn class-free từ cache. Add Input gồm prepared dataset, concept-mass-c-c0-v111-v112, mscoco-2014, objectdetectionecache và prompt-cache. Không cần raw visual dataset.

Dependency amendment trước model, Astra đã đồng ý: CPU cache gốc chỉ có native prompts cho main5000; auxiliary-only có placeholder0. Supporting actualdecoder held diagnostic cần original98prompts từ legacycache d7910e9b… thêm thành `held_native_prompts.npz`, không sửa sourceH5/receipt và không dùng GT/text reconstruction. Check mapping IDs/finite/shape/mask/fullsourcehash và exact main-prompt sentinels, ghi overlay fileSHA và effective array-content SHA riêng. Thiếu prompt đúng thì failbeforetrain, không zero/drop diagnostic rồi nhận completed. Aux BCE/MP vẫn chỉ raw visual/proposals, không nativepromptforward.

GPU preflight phải kiểm numeric C0 parity, sampled-policy rescore, actual reader key-swap/null, throwaway CE/aux gradient connectivity, rank/init/frozen identities và20full-objective optimizer steps rồi reset init/RNG/optimizer. Forecast đầy đủ500steps+dev/held/intervention trong80%remaining7200s; không fit thì STOP_PREFLIGHT_NO_TRAINING. GPU worker watchdog7200s, CPU scorer1800s; caps không ETA. Quota gần20h/account, actual billing convention UNVERIFIED phải báo, không hứa quota=wall.

User-run eval: rawC0 và endpoint mỗi arm trên cùng1000dev, held99 observed scores/16query groups và actual decoder query likelihood, fixed64 reader-off intervention. CIDEr/CHAIR/recall/length/paired uncertainty; sparse relation-caption truth NOT_ESTABLISHED nếu chưa verified parsing. Chỉ khi nhận cả hai ZIP mới so E-D/ownC0; không promote từ một arm. Incomplete/failure luôn archive logs, không chấp nhận subset như endpoint.

Chưa có kiểm tra CUDA/model hoặc runtime mới tại thời điểm ghi protocol. Tiếp theo triển khai rồi chạy static/metadata/NumPy tests; người dùng kiểm GPU bằng notebook.
