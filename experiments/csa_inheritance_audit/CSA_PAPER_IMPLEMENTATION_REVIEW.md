# Codex1 đánh giá đặc tả X024

01/11/2026. Đã đọc toàn bộ spec tại commit99373eb và đối chiếu trực tiếp PDF user cung cấp, SHA63ae0db0ee4a1eeeb41a4a241a5ff186bae342caf180fe7e5b2d2875d86575eb. Đọc method/recipe PDFp5–8; render và xem Eq3–17 ởp5–6. Không importTorch/model/autograd, không chạy notebook/scorer.

## Quyết định

**ACCEPT cho M1 metadata và triển khai module theo paper; chưa GO huấn luyện lớn.** Kế hoạch đã đủ cụ thể, không cần một vòng brainstorm khác hoặc quay lại X020. Không lấy missingdetector4ds làm blocker bắt buộc. Việc đổi visual/text encoder được khai báo đúng; đây là baseline CSA theo paper với inputs thay thế, không exact full-system reproduction.

## Quyết định công thức trước core code

- ACCEPT Eq6 cosine có normalization, Eq8softmaxR không temperaturephụ, Eq11G*self+(1−G)*cross/gatescalar và không finaloutproj. Các điểm này khớp công thức nhìn trực tiếp.
- ACCEPT Eq15 hai gate riêng dùng previous-wordembedding, Eq16sum khôngchia√2. Không copy bug sharealpha hoặc đưa decoderC0 vào.
- ACCEPT Eq3 LWa→MLP K→K→K, softmax trên bankwords và projectedvalues là quy ước tường minh. Paper có two-layerMLP và W_a nhưng không đặc tả đầy đủ hidden/bias/axis; không claim numericalparity hoặc công thức buộc duy nhất cách này.
- ACCEPT OSA2 nhận selectedT1, FFN postnorm sau attention, gatevector của decoder là các convention đã khai báo, không phải chi tiết đều được in rõ trong paper. Giữ chúng ở một profile cố định, không sweep để cứu baseline.
- Bổ sung rõ khi tạo policy: **không thêm positional/geometry embedding cho region tokens** trong baseline; decoder dùng sinusoidal position như spec. Kiểm joint permutation của region records không làm đổi output ởeval trong tolerance.
- Bổ sung rõ trước beam decoder: **raw cumulative log-probability, không length penalty** ở baseline; EOScompleted beams giữ score, không thêm token/score sauEOS. Nếu sau này tìm được quy ước tham chiếu khác thì ghi amendment trướcrun, không âm thầm đổi.

## Ba điều kiện vận hành còn phải kiểm

1. **Cache/detector:** xác minh exact COCOID→row mapping, preprocessing khôngcenter-crop, raw197×768, boxcoordinate/ảnhsource, completion và actualN. Bboxcache không tự là visualfeatures; cap150 không sinh thêm detection nếu producer chỉlưu6hoặc10. Metadata producer chưa biết không được giả là đã biết. Thiếu provenance phải nêu rõ khả năng dùng/re-extract, không silently drop ảnh. Đây là gateassets/run, không ngăn viết contract/core.
2. **Banks:** tự dựng nouns/verbs từ TRAIN, pin source/POS/CLIP/tokenization/normalization, không dùng binaryRegionCLIP bank tác giả hoặc text từval/test. Việc đổi sangCLIP512D đúng phạm vi nhưng có thể ảnh hưởng chất lượng; không nhận đã giữ semantics tương đương RegionCLIP.
3. **Ngân sách:** không diễn giải15XE+30RL là run nhỏ. Nếu có đúng5caption/ảnh,113287trainimages,batch32 và giữ batchcuối thì1XEepoch≈17702optimizerupdates,15XE≈265530;1RLepoch≈3541,30RL≈106230. Đây là phép tính điều kiện, không actualmanifestcount/ETA. Mỗi XEepoch theo captionevents còn RLepoch theo uniqueimages; phải ghi cả denominator, không so epoch trực tiếp. Actualcounts/throughput/peakmemory và quota hiện tại mới quyết định lịch đầyđủ.

Pilot100XEupdates của spec hợp lý để kiểm vận hành, không yêu cầu thắng A/C0. Full training không tự được phê duyệt bằng smokePASS. Adam không tự đổi AdamW; RLbeammean surrogate không gọigreedySCST/unbiasedMC; không thêmobjectreward/KL. Baseline mới họccaptioner từđầu, không tiếp tụccheckpointC0.

## Bước thực hiện tiếp theo

M1 audit inventory local/cachemanifest/detector source và train-only word/vocab contract; sauđó tạo namespace `experiments/csa_paper/` và implementcore theo quyết định trên. Testlocal chỉmetadata/AST/NumPy; model/gradient/DDP do user-run Kaggle kiểm. Không yêu cầuuser duyệt lại từngdefaultlayer; hỏi khi thiếupayloadbắtbuộc, cần đổi extraction/data scope hoặc chốtbudget. Giữ kếtquả cũ nguyêntrạng, không sửaDocs hay sheetcho một review.
