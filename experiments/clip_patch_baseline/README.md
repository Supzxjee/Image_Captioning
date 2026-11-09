# Baseline CLIP patch và decoder chuẩn

## Đang xử lý: luồng đọc cache quá chậm

Fullv1 dừng trước update đầu: fullH5 median1,403s/update, conservativeprojection148,38h; không phải CUDA/OOM. [Kết quả](FULL_V1_RESULT.md). Không chạy lại fullGPU cũ hoặc tăng cap để né vấn đề.

Chạy **CLIP_PATCH_CACHE_IO_PROBE_CPU.ipynb**, Accelerator **None**, InternetON, gắn cùng `visual-cache` và `vuthetam/mscoco-2014`. Không tokenAPI, model hoặc scorer. So sánh4readers trên cùng2048TRAINcaptionevents: mountedH5 hai lượt, stage mẫu~0,6GB localfloat16, localmemmap hai lượt, kiểm hash đặc trưng. Ghi disk/RAM/staging/read+stack/startup; không chép full35GB. [Protocol](IO_PROBE_PROTOCOL.md), [authorchecks](IO_AUTHOR_CHECKS.json). Gửi **clip_patch_cache_io_probe_v1_review.zip**. Chưa đo Kaggle thật, không hứa speedup hay suy ra thời gian train từ local sample nhỏ.


## Lượt full10XE mới

Smoke192 đã hoàn tất: [kết quả](SMOKE_RESULT.md). Notebook **CLIP_PATCH_BASELINE_FULL_XE10_2GPU.ipynb** train mới10epoch XE trên113287TRAIN, cùng cache và T4×2; không nạp trọng số smoke hoặc mô hình cũ. Vẫn chỉ cần `visual-cache` và `vuthetam/mscoco-2014`; không API token/ROI/bank/prompt.

Đọc H5 trực tiếp với CPU LRU và prefetch, chỉ stage metadata/caption arrays nhỏ. Đo fullIO, gradient và beam trước train; parent10,5giờ STOPcap không phải cam kết runtime/quota tính phí. Kiểm tra quota tài khoản trước chạy. Khởi tạo mới seed42, global32, Adam1e-4 peak, warmup1epoch rồi cosine1e-5, khôngSCST/rerank/TEST. [Protocol đã khóa](FULL_XE10_PROTOCOL.md).

Mỗi epoch chấm VALID5000 beam5/max20 và NLL/CIDEr; maxCIDEr chọnbest, tiesearlier. Giữ `best.bin` và `checkpoint.bin` endpoint10. BLEU1/4, ROUGE-L, CHAIRs/i và recall cuối lượt lấy từ đúng caption của best/end; METEOR/SPICE/TEST làm riêng sau, không ghi là đã đo ở đây. Nếu interruption, giữ cảfolder; `RESUME_ROOT` chỉ dành cho exactrecipe và state/Adam/RNG/receipts, không tự train lại.

Gửi **clip_patch_baseline_full_xe10_v1_review.zip** sau chạy. Chưa có kết quả full; không hứa110CIDEr. Decoder mới giữ nguyên initializer/state layout của smoke, chỉ tách encode/decode để beam dùng lại projectedmemory.

## Lượt smoke đã chạy

Lượt đầu là kiểm tra dữ liệu, việc học và tốc độ; chưa phải train full hoặc đánh giá chất lượng caption.

Import **CLIP_PATCH_BASELINE_SMOKE_2GPU.ipynb** lên Kaggle. Chọn **GPU T4×2**, Internet ON, Save & Run All. Gắn hai Dataset:

- `/kaggle/input/datasets/ducanh2403/visual-cache` — H5 CLIP gốc197token/ảnh, không thay bằng ROI134.
- `/kaggle/input/datasets/vuthetam/mscoco-2014` — dùng `dataset_coco.json`.

Không cần token API, detector, bank hoặc prompt. Sửa hai biến path nếu mount khác. Thời gian STOPcap45phút, không phải dự báo thời gian/quota tính phí; ngân sách mới nhất người dùng báo30giờGPU.

Mô hình: CLIP ViT-B/16 frozen/cache →196patch (bỏCLS) →Linear/LayerNorm512 →Transformer decoder chuẩn4lớp →vocab. Không có3visualencoderlayers hay decoder hai memory của baseline cũ. Decoder/projection học mới từ đầu; không finetune CLIP hoặc nạp checkpoint cũ.

Chạy200update XE trên256ảnh TRAIN để kiểm causal mask, đường dùng ảnh, gradient DDP và loss trên cùng TRAIN probe có giảm không; đo tốc độ/bộ nhớ và lưu8caption TRAIN minh họa. Các số này không phải kết quả VALID/TEST và không chứng minh hội tụ hoặc đạt110CIDEr. Throughput trên pool nhỏ/page-cache không đảm bảo tốc độ trên toàn bộ corpus.

Gửi lại **clip_patch_baseline_smoke_v1_review.zip**. Sau khi đọc kết quả, mới chốt IO cho toàn bộ patch cache (~36GB dữ liệu FP16), số epoch XE, policy giải mã và quy tắc đánh giá. Chưa tích hợp bank/prompt vào baseline mới. Prototype dự kiến cần điều chỉnh mapping ROI→patch khi đưa prompt gắn vùng vào sau này.

Tác giả chỉ chạy kiểm tra NumPy/mock/AST/source package. Neural/CUDA/DDP đều do notebook người dùng chạy. Giữ các kết quả pipeline cũ, không sửa báo cáo Word. [Protocol](PROTOCOL.md).
