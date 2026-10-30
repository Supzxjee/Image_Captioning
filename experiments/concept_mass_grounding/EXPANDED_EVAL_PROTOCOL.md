# A/C0/C — protocol evaluation mở rộng

30/10/2026. User đã đồng ý chốt một evaluation, không training. Không có notebook/run mới tại Astra. Không thêm controls, semantic probe hoặc sweep.

## 1. Cohort khóa: val remainder4000

Dùng `val_extra4000_cg`: val5000 trong nguồn nhóm đã có, trừ đúng dev1000 hiện tại của C/G/C0 (gain metadata). Giữ thứ tự source val5000; không chọn theo output. Root xác nhận4000uniqueCOCOIDs, không giao currentdev1000 hoặc TRAIN10k; adaptation4000⊂TRAIN10k.

Nguồn archive: `experiments/noun_evidence_proposal/data/noun_evidence_train10k_v1_dataset.zip`, SHA256 `ebf880fcc0ae23388057abc84ce796b64ee1f0f6c16f804d1cdbf98e73fb78fd`.
Currentdev metadata: `experiments/semantic_grounding_method/data/gain_metadata.json.gz`, SHA256 `62f43f771aec9feba31a2fbbb6fe7db2878f259979faac378b74a754822c7b67`.
Manifest khóa: `EXPANDED_EVAL_MANIFEST.json`, SHA256 `8f68ab9c45ab9c2d11618e837b8d945f20c0c3148c3b1705ff352cdf244a6fab`.

Packaging correction30/10: original Windows bytes(CRLF) hadSHA`31559d3f68a133fe9d923489875df21b641b84b8238d3a805ec54de9a99af661`; Git normalizes toLF, whose exact blobSHA is above. ParsedJSON and all4000IDs/order are unchanged. Runtime now pinsLF bytes, not a new cohort.

**Correction trước inference:** `audit/vcoco_grounded_caption_week1_groups.json` có historical `caption_tune1000` khác currentdev: overlap203/1000COCOIDs. Historical `caption_validation_remainder4000` còn797currentdevIDs nên không được dùng làm cohort này. Sửa tên cohort để tránh lẫn; ý định loại currentdev giữ nguyên. Đây là metadata correction trước run, không chọn lại ảnh sau output.

Đây là **reused validation**: fullval đã evaluate A/H/L và nounG/S; ít nhất4visual cases ngoài dev đã xem. Không gọi fresh/independent test. Không loại riêng cases đã xem để tạo vẻ mới.

Root pin manifest source-row index/COCOID/filename/split/refs/hashes trước inference; xác nhận toàn4000split=val qua pinned Karpathy JSON, không giao dev1000/TRAIN10k/adaptation4000. Thiếu tensor/ref/metricGT hoặc sai identity thì fail, không drop/refill ảnh. Không dùng historical caption_tune1000 để định nghĩa currentdev; cả hai đều không phải adaptation TRAIN. Rawstage4000≈1.38GB original CLIP+legacy prompt/mask; không feature extraction.

## 2. Checkpoints và decoding

| Model | Source | Full file SHA256 |
|---|---|---|
| A | original A dataset | 630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901 |
| C | endpoint500; ducanh2403/2-vi-cai-kia-qua-nhieu version111 | 99f7707f73fe5006592ea9ccbf04bce28a53a65a022cddc5c24ba6a78923c2f4 |
| C0 | endpoint500; cùng kernel version112 | 5d92cd5cb48077c7332bb3ff9b295cad1df5d76ede818adabef28035f055fa9d |

Versions do user cung cấp; mount paths chưa có. Export/mount version-specific rồi verify bytes, không lấy latest hoặc dùng filename thay hash.

Cùng native model/tokenizer/vocab, original visual/legacy caches và numerical policy C/G/C0. Greedy, max_len30 đúng convention cũ: BOS và tối đa29next-token steps, dừngEOS; batch8/rank; eval/no-grad. Không optimizer, logits adjustment, alias/event processing.

Load strict hoặc audited explicit key mapping, không bỏ missing/unexpected keys. Preflight kỹ thuật/hash/parity, không research diagnostic. Có thể dùng các sentinel dev batches có saved A/C/C0 captions trước để kiểm đường inference, không đưa vào4000metrics. Mismatch thì sửa kỹ thuật, không tune generation.

## 3. Metrics và IDF

Giữ official PTB/CIDEr/BLEU/METEOR/ROUGE và CHAIR80/aliases/object-matching như pilot, không thu hẹp evaluator theo72training classes. GT/ref chỉ evaluator.

Primary report trên4000: CIDEr, CHAIRs/i, absolute false captions/mentions, total mentions, recall numerator/denominator, mean words và per-image values.

CIDEr IDF của4000khác1000: không so absolute scores hai cohort. Nếu báo pooled5000, ghép đủ captions+refs rồi rerun scorer trên5000, không average CIDEr aggregates. Có thể reuse saved1000captions sau verify provenance/model/settings/IDs; không cần generate lại1000. Pooled là secondary vì chứa dev đã dùng chọn phương pháp.

Không thêm neutral-prefix/VSR/semantic-panel diagnostics.

## 4. Contrasts và quy tắc kết luận trước run

Primary1: C−C0 factuality và CIDEr/recall tradeoff (contribution preservation).
Primary2: C−A dual goal.
Secondary: C0−A continuation quality.

20000paired image bootstrap draws, seed20261030, cùng draws giữa models. CIDEr dùng per-image scores của common4000IDF; không đổi IDF trong mỗi draw. CHAIRi/recall recompute aggregate ratios từ counts, không mean image ratios. Report marginal95%CIs, không gọi simultaneous bands.

Margins được chọn **sau pilot, trước4000outputs**, là practical choices, không data-derived equivalence guarantees:
- C so C0 chấp nhận tối đa.01rawCIDEr cost.
- Recall không giảm quá.005absolute so C0 hoặc A.
Không đổi margins sau kết quả.

Để nhận claim chung “preservation có ích và đạt mục tiêu user”, cần **tất cả**:
1. lower95%CI CIDEr(C−A)>0.
2. upper95%CI CHAIRs(C−A)<0.
3. upper95%CI CHAIRs(C−C0)<0.
4. lower95%CI CIDEr(C−C0)>−.01.
5. lower95%CI Recall(C−C0)>−.005 và Recall(C−A)>−.005.

Report discordant counts và exact two-sided McNemar p cho hai contrasts CHAIRs; mỗi p<.05 là điều kiện bổ sung vì sparse errors. Zero discordants không chứng minh equivalence. Report CHAIRi/false-mention counts; nếu mentions xấu đi thì nêu tradeoff, không gọi hallucination giảm toàn diện.

Đây là joint/intersection gate, không chọn bất kỳ metric có significance. Marginal p/CI vẫn chịu one-seed/post-selection/reused-validation limits; bootstrap/exact tests không xóa selection bias. Không mở nhóm/class/threshold mới sau nhìn số.

Nếu chỉ Primary1 đạt: effect tương đối, dual goal chưa đạt.
Nếu chỉ Primary2 đạt: task improvement có thể có, preservation attribution chưa thiết lập.
Không qua/không rõ: kết thúc đợt này, không tuning hay tăngn tiếp trên4000.
Evaluation không tạo novelty; C0 gain không được đổi tên thành phương pháp mới.

## 5. ETA và caps

C0 receipts: prep5000rows2284.0s → scaling thô4000≈1827s≈30.5min. Rank0 prediction A9.94s/end10.28s mỗi500images trong resident wrapper → ba models×4000chia2ranks khoảng120s prediction loop. Worker elapsed≈699s gồm≈223s training. Không dùng2phút loop làm total ETA.

Planning ETA **50–100phút** khi I/O tương tự; không guarantee. Caps:
- dependency setup tổng900s;
- CPUstage/metadata/hash3600s;
- two-GPU startup/load/preflight/inference1800s;
- CPU officialmetrics/statistics1800s;
- packaging/close120s; parent tổng8400s≈2h20.

GPU phase30min trên2GPU ≤1device-GPU-hour. Nếu GPUs enabled suốt cả job, full parent cap có thể tương đương≈4h40device-hours allocation; không phải billing dự báo. Ghi actual phase và total riêng. Quota24h cũ không dùng; root đang hỏi current balance. Stage I/O là bottleneck, không thêm optimization experiment.

Timeout/missing checkpoint: giữ journal, mark incomplete, không score subset như complete. Resume cùng pinned inputs/config là kỹ thuật; không thay model/cohort để qua cap.

## 6. Bàn giao

Root triển khai một notebook eval/static checks theo protocol; hoàn tất version111/112 mounts và current quota trước user-run. Không cần thêm review nghiên cứu nếu không có thay đổi material. Sau run cache text một lần, audit complete/identity/metrics, report cả primary contrasts+limits, cập nhật state/Git/sheet.

Nếu không hỗ trợ dual goal/contribution, giữ kết quả empirical trung thực và trao đổi scope/thời hạn với user; không hứa accepted31/10.
