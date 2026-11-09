# Baseline CLIP patch và decoder chuẩn

## Chạy mới bằng LOCAL V3

Dùng **CLIP_PATCH_BASELINE_FULL_XE10_LOCAL_V3_2GPU.ipynb**, T4×2/InternetON, haiDataset nhưV2. V2 dừng vì30phútprep chỉchép60159/118287ảnh, chưa train. V3 bắtđầu mới nhưuser yêu cầu, tăngchỉ thời gianprep nội bộvàoutercall lên5400s(90phút), tổngcap10,5giờ giữnguyên. Không lấy lại/tmp hoặccheckpointV2. [Kết quảV2](LOCAL_V2_RESULT.md), [amendmentV3](LOCAL_STAGE_V3_PROTOCOL.md), [QAV3](LOCAL_V3_AUTHOR_CHECKS.json). Môhình/stagecode/trainrecipe/data/sampler/VALIDmetrics khôngđổi; GPUfullIOpreflight/budgetgatesvẫn bắtbuộc. Gửi **clip_patch_baseline_full_xe10_v3_review.zip**, giữbest/latest nếuđãtạo. Chưađo fullstageGPUthật;90phútlàcapkhôngphảiETA.

## Lịch sử LOCAL V2 (đã dừng khi chuẩn bị)

**CLIP_PATCH_BASELINE_FULL_XE10_LOCAL_V2_2GPU.ipynb** là bản thay thế V1 bị dừng0update. T4×2, InternetON, cùng2Dataset `visual-cache` và `vuthetam/mscoco-2014`, khôngAPI/ROI/bank/prompt. V2 đọc tuần tự H5 và chép33,17GiB TRAIN+VALID patchFP16 sang /tmp memmap; không nạp hết vàoRAM, không tríchCLIP lại, không đọcTESTfeature. Cần38GiBscratch và4GiBWorking; kiểm tra ở runtime. Chuẩn bị cap30phút, tổng10,5giờ giữ nguyên, chưa phảiETA.

Giữ mô hình/khởi tạo/captionshuffle/10XE/global32/Adam/LR/VALIDbeam5 như V1. Sau stage+SHA, đo lại fullIO/DDP/beam trước train;64update thử bị bỏ. Runtimeprojection/cap vẫn được kiểm, không hứa speedup24× từ mẫuCPU. Fullcorpus33GiB vượtRAMquan sát31GiB. Không dùng lại trọng số V1 (chưa train). [Amendment](LOCAL_STAGE_V2_PROTOCOL.md), [QA](LOCAL_V2_AUTHOR_CHECKS.json). Gửi **clip_patch_baseline_full_xe10_v2_review.zip**; giữ cảoutputfolder/best.bin/checkpoint.bin nếu đã tạo.

CPUprobe COMPLETE: [kết quả](IO_RESULT.md). Mounted lượt1/2 .936/.046s, local .039s; featurehashkhớp, không phải fullGPU timing. [V1 stopped](FULL_V1_RESULT.md). Không chạy lại notebook V1 cũ.

## Lịch sử: bản full10XE V1

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
