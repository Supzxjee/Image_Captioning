# Relation frames — data/cache readiness decision

28/10/2026, Asia/Bangkok. Dựa trên CPU audit/worker receipts root chuyển; Astra không đọc lại ZIP hoặc chạy model/GPU.

**Chấp nhận trạng thái `DATA_AND_CACHE_METADATA_ACCEPTED_MODEL_PREFLIGHT_PENDING`.** Thiết kế dữ liệu V2 và human audit đã qua; không mở lại brainstorm hoặc sửa cohort.

## Evidence được ghi nhận

- V2 manifest SHA `6ca6af23d0a31043fecdf33398f5b60098be07cbeea9fb91cbfec4b314c2cfa7`:256train/64held pairs,612distinct images.
- Human audit hoàn tất32/32effectiveconfirmed,16supported/16contradicted,0uncertain/definite disagreement/box issue trong sample. Đây là assisted sample audit, không independent corpus gold. Chi tiết override lineage trong `ASTRA_AUDIT32_DECISION_2026-10-28.md`.
- User xác nhận source version105. Worker source commit `dd49cbb`, pinned preparation/source hashes khớp package theo root audit; returncode0,371.47s,612/612feature+original-prompt coverage; không neural forward/training.
- Review ZIP SHA `846d4fb94125aa39d6a2825d14a6489c5f60a58e50ae92b30c5cce79fa50eb91`; receipt SHA `9bc63a85ce59084d029739d55baec4d2ca359c3afeb78ff320a22b1f310f5915`.
- Root xác nhận binary manifest exact match, independent image IDs/names/dimensions và1280target-box references pass;3metadata payload hashes verified. Metadata source/referent binding có bằng chứng, không chỉ count coverage.

## Ranh giới của acceptance

`inputs.h5` đầy đủ không nằm trong small review ZIP. Full-file SHA worker báo là `daf32d52859452eb2beb8c35e2aad6ed6418cf52c68e4ba646be633151a7a1e9`; worker báo row readback pass. Astra/root chưa độc lập đọc full payload từ small ZIP. **Verify SHA của full file trên Dataset mount thực tế ở transport/model preflight**, không coi receipt thay cho file integrity.

Prep receipt có pending-human flag là trạng thái lịch sử; human audit hoàn tất riêng sau đó đã supersede flag này. Không yêu cầu user chấm lại32mẫu hoặc tạo lại original prompts vì flag cũ.

Metadata/cache acceptance chưa chứng minh numerical model compatibility, gradient flow hoặc hiệu quả. Chưa có learned model scores để chọn configuration; primary cohort/gates giữ nguyên.

## Bước kế tiếp được khuyến nghị

1. Sau user upload, verify real Dataset mount/exact payload hashes, đặc biệt fullHDF5SHA trên. Không tự chạy upload API hoặc giả mount đã tồn tại.
2. Triển khai/kiểm model preflight theo protocol đã khóa trước optimizer: same coordinate/mask binding, inputs/vocabulary, finite tensors/logits/losses, expected parameter/frozen-state behavior và gradient/normalization checks phù hợp. Không tạo một vòng data audit mới.
3. Khi preflight pass và phạm vi GPU run đã được user duyệt: **200updates RelPos/RelContrast frame pretraining**, same initialization/order/cohort, endpoint200 duy nhất. Gate ranking/CI trên64held pairs và32/32strata giữ nguyên.
4. Chỉ khi pretrain gate pass mới thực hiện ba caption arms Tune/RelPos/RelContrast theo kế hoạch. Không chuyển thẳng từ cache receipt sang caption training, không thay recipe nếu pretrain fail.

Quyết định này xác nhận research readiness ở data/cache-metadata layer; **không phải authorization để launch GPU**, không thay current-quota verification hoặc yêu cầu model preflight. Giữ các kết quả T trước đây nguyên vẹn.
