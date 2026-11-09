# Smoke192 — baseline patch chạy được, chưa có điểm chất lượng

User báo nguồn192; theo ngữ cảnh notebook chính `ducanh2403/2-vi-cai-kia-qua-nhieu`. [Metadata audit](SMOKE_RESULT.json):20assertions PASS; source8filepins, coverage256TRAIN/1280referenceevents, hai worker200updates, CUDA/DDP guards, throughput arithmetic và noVALID/TEST flags. ZIP đọc1lần/cached ngoài repo `.review_cache/clip_patch_smoke/archive_text.json`.

- Cùng TRAIN probe NLL9,2162→3,7613;27.461.904tham số học.
- Median step~0,0564s,~567captionevents/s global32;2T4, peak allocated~0,667GiB, peak reserved≤0,831GiB. Loader/H2D median~0,00857s.
- Stage256ảnh83,0s; worker20,7s gồm preflight/learning/samples, không phải fullnotebook/quota billing.
- Causal mask, image-conditioning, finite visual/model gradients và token-normalized DDP-gradient parity báo PASS trong runtime.
- SampleTRAIN còn lặp/chung chung sau200update; loss giảm chỉ cho thấy cập nhật được trên pool nhỏ, không chứng minh caption tốt, hội tụ hoặc110CIDEr.

Quyết định: chuẩn bị full10XE với đo actualfullIO/decode trước train. Cache H5 producer ghi chunk theo ảnh; runtime full phải ghi chunk/compression thực tế. Dùng streaming+boundedLRU/prefetch, tránh sao chép~36GB patch. Không ngoại suy thời gian toàn bộ từ pool256 trong RAM/page-cache.

Tác giả không chạy model/CUDA/scorer hoặc độc lập đọc lại feature arrays/checkpoint binaries. Metadata rawcache có `model_revision=null`, vẫn là giới hạn provenance. Không sửa Word.
