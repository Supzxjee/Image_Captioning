# Baseline CLIP patch và decoder chuẩn

Lượt đầu là kiểm tra dữ liệu, việc học và tốc độ; chưa phải train full hoặc đánh giá chất lượng caption.

Import **CLIP_PATCH_BASELINE_SMOKE_2GPU.ipynb** lên Kaggle. Chọn **GPU T4×2**, Internet ON, Save & Run All. Gắn hai Dataset:

- `/kaggle/input/datasets/ducanh2403/visual-cache` — H5 CLIP gốc197token/ảnh, không thay bằng ROI134.
- `/kaggle/input/datasets/vuthetam/mscoco-2014` — dùng `dataset_coco.json`.

Không cần token API, detector, bank hoặc prompt. Sửa hai biến path nếu mount khác. Thời gian STOPcap45phút, không phải dự báo thời gian/quota tính phí; ngân sách mới nhất người dùng báo30giờGPU.

Mô hình: CLIP ViT-B/16 frozen/cache →196patch (bỏCLS) →Linear/LayerNorm512 →Transformer decoder chuẩn4lớp →vocab. Không có3visualencoderlayers hay decoder hai memory của baseline cũ. Decoder/projection học mới từ đầu; không finetune CLIP hoặc nạp checkpoint cũ.

Chạy200update XE trên256ảnh TRAIN để kiểm causal mask, đường dùng ảnh, gradient DDP và loss trên cùng TRAIN probe có giảm không; đo tốc độ/bộ nhớ và lưu8caption TRAIN minh họa. Các số này không phải kết quả VALID/TEST và không chứng minh hội tụ hoặc đạt110CIDEr. Throughput trên pool nhỏ/page-cache không đảm bảo tốc độ trên toàn bộ corpus.

Gửi lại **clip_patch_baseline_smoke_v1_review.zip**. Sau khi đọc kết quả, mới chốt IO cho toàn bộ patch cache (~36GB dữ liệu FP16), số epoch XE, policy giải mã và quy tắc đánh giá. Chưa tích hợp bank/prompt vào baseline mới. Prototype dự kiến cần điều chỉnh mapping ROI→patch khi đưa prompt gắn vùng vào sau này.

Tác giả chỉ chạy kiểm tra NumPy/mock/AST/source package. Neural/CUDA/DDP đều do notebook người dùng chạy. Giữ các kết quả pipeline cũ, không sửa báo cáo Word. [Protocol](PROTOCOL.md).
