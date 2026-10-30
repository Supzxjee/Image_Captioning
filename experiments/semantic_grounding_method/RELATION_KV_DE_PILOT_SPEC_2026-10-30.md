# Pilot D/E: đặc tả phương pháp sau bước CPU

30/10/2026. Astra method review, bàn giao root triển khai trong phạm vi user đã duyệt pilot nhỏ. Không tự chạy training/Kaggle/model local. Không commit/push trong vòng bàn giao này. [Hypothesis](RELATION_KEY_FACT_VALUE_HYPOTHESIS_2026-10-30.md), [nearest-method closure](KV_NEAREST_METHOD_CLOSURE_2026-10-30.md), [CPU protocol](RELATION_KV_CPU_PREPARATION_PROTOCOL.md) vẫn là nguồn quyết định trước. Các lựa chọn loss/text/schedule dưới đây chốt phần trước đó còn pending, không sửa lại lịch sử như đã có trước CPU run.

## 1. Quyết định và phạm vi bằng chứng

**Đủ dữ liệu vận hành để thử D/E hẹp; không mở rộng ontology hoặc tìm thêm nguồn.** Theo audit root (không đọc lại ZIP): CPU READY_FOR_MODEL_PREFLIGHT, 3689.47s; 5472 staged images = 4000 caption TRAIN + 1000 caption dev + 472 auxiliary-only. Có 500 original facts sau dedup: fit401/393images, held99/98images. Fit contact false146/true138, containment false54/true63; held31/39 và13/16. Fit58 lexical true/false groups, held16. Một ambiguous-GT orbit/2 observations đã quarantine. Held không giao caption TRAIN/dev và không giao auxiliary fit; 19 fit images thuộc caption TRAIN. Các ngưỡng CPU là operational minimum, không power calculation.

Root kiểm 69990 metadata assertions, không phải 69990 experiments. Core đã thực thi là `bfc54da3078ce241282643a4018698942f67f253`; 16 tests user-run PASS. Reported inputs.h5 SHA256 `960e1995c6e7695597d95f9535f536576a5c74a03c6c26386da4f7c5b2a4da89` vẫn phải kiểm file thật trên Kaggle. Không nhận model/CUDA PASS từ metadata. Source-version/dataset mount do root pin trước bàn giao notebook.

P3 = contains/inside/touching, đúng literal physical containment/contact như CPU protocol. Đây chỉ là **một inverse pair và một symmetric predicate**. Nó đủ kiểm inductive bias value-sharing trên hai họ nghĩa này; chưa kiểm generic directional grounding, frame reasoning, đa dạng động từ, hoặc multi-instance ambiguity. Cohort GT-unique/different-class không kiểm khả năng phân biệt nhiều instance cùng class. Một kết quả tốt vẫn chưa đủ bằng chứng tổng quát hoặc đủ novelty để công bố. Không dùng ví dụ above/below trong hypothesis như thể đã được thử ở pilot này.

Ba câu hỏi riêng: (i) predicted boxes có match referents không — CPU mới kiểm nguồn/matching/coverage, chưa chứng minh mọi annotation đúng; (ii) representation có phân biệt observed truth và các fact khác nhau không; (iii) decoder có dùng branch để cải thiện free caption không. Không lấy câu trả lời (i) hoặc construction identity thay (ii)/(iii).

## 2. Model và đúng một biến D/E

Giữ F/projections/nonlinear geometry–predicate interaction và native decoder read như hypothesis. K<=6 predicted boxes; enumerate toàn bộ ordered pairs × P3 + null, tối đa91 slots. Không dùng GT labels, observed truth, auxiliary query nouns, fit bank hoặc reference captions để chọn/prune slots lúc forward caption/inference. E ghép đúng orbit `(i,p,j) ~ (j,inverse(p),i)` bằng mean trước Wv; D không ghép. Cả hai giữ cùng hai slots/orbit và null value0. False/unknown hypotheses vẫn ở bank như nhau, không truth-thresholding.

Trainable: common raw-feature projections, F, Wq/Wk/Wv/Wo, auxiliary text projections ở mục3 và hai native decoder layers cuối. Upstream CLIP/native visual projection/Q-Former/native prompt/input embeddings/native vocabulary head frozen. Không train vocabulary head hoặc thêm gate/planner. Wo weight=0 ở initial state, **Wo không có bias**; native hidden + Wo(context), không thêm LayerNorm làm mất parity C0. Wv không nhận trực tiếp auxiliary loss; đường caption phải học dùng nó. Keys chỉ được phép khác nhau, không mặc định đã học đúng direction.

Khởi tạo cả D/E từ **cùng raw C0** SHA256 `5d92cd5cb48077c7332bb3ff9b295cad1df5d76ede818adabef28035f055fa9d`. Serialize một common new-module init seed42 rồi nạp byte-identical vào hai arms; chỉ config value_projection khác. Param names/count/shapes và trainable scope phải giống. Raw C0 teacher immutable không có relation branch; teacher không dùng adapted decoder hoặc E/D evidence. Shared readout nghĩa là cùng cấu trúc/init và training protocol, không đồng train hai arms hoặc ép endpoint weights bằng nhau.

## 3. Frozen text: không cần trích CLIP text embeddings

CPU prep chưa trích text features. **Chọn native C0 frozen input-token embedding** để tránh thêm encoder/extraction. Dùng native tokenizer pin từ checkpoint; mean token embeddings của từng content field, bỏ padding/BOS/EOS; không position embeddings/dropout. Mỗi field phải có >=1 content token. Dimension512, float32 cached, dump token IDs/masks/hash và embedding-table hash. Multi-token category/predicate lấy mean trong field, không lấy mean cả câu.

**Tokenizer/OOV amendment, trước implementation/training:** native C0 là `word2idx`10000 với `lower().split()`, không BPE. Theo metadata audit root trên67 unique subject/predicate/object fields, chỉ từ `drier` trong `hair drier` là OOV; cả3 predicates known. Chốt cùng một native policy D/E: mỗi whitespace token dùng đúng `word2idx.get(token, native_unk_id)`, giữ `<unk>` như một content token trong mean. Không bỏ token OOV khỏi mean, đổi tên hair drier, thêm vocab/tokenizer/encoder, drop fact hoặc đổi split/group để né OOV. Raw category strings, original fact IDs và lexical_group vẫn nguyên; không merge groups theo token IDs/embeddings bị trùng. Query caption diagnostic cũng dùng chính native tokenizer, không tự sửa văn bản.

Đây thay khuyến nghị fail-all-OOV trong tin trao đổi trước, vì một token không biết trong một multiword field vẫn để lại `hair` và không làm pipeline cơ học vô nghĩa; quyết định dựa trên metadata trước model, không kết quả để cứu nhánh. Không nhận embedding đó phân giải đầy đủ hair drier hoặc gọi frozen native word embeddings là contextual/CLIP semantics. Log exact raw fields/tokens/IDs/UNK mask, unique-token/field OOV rates, occurrence rates theo fit/held và role, số original facts bị ảnh hưởng (views không thành samples độc lập); giữ native_unk_id/embedding hash. Cả D/E phải cùng receipt/token-ID hashes. Predicate OOV, empty-content field, thiếu native UNK ID hoặc OOV set khác pinned current audit là source/config mismatch phải review trước train, không silent fallback; không tự chọn ngưỡng OOV theo dev. Không mở thêm arm hay loại một subset để báo thắng riêng sau kết quả.

- `e_p` = pooled frozen tokens của literal predicate contains/inside/touching; `t_p = T_p e_p` (512→128), dùng ở F và common text query. Đây là relation descriptor, không caption future word hoặc GT class.
- Chỉ cho auxiliary loss/evaluation, `q_v = normalize(GELU/MLP([T_s e_subject ; T_p e_predicate ; T_o e_object]))`, mỗi projection512→128, text MLP384→128→128 với GELU. T_s và T_o riêng, không tie; same module ở D/E. Đây là role-aware query encoding thông thường, không novelty. Không thêm class text vào F/caption branch.
- Auxiliary text dùng đúng subject/predicate/object fields của từng equivalent view và tokenizer có provenance; query_views nguyên bản vẫn dùng cho caption diagnostic. Không parse hay thay caption references để tạo truth. GT class text chỉ là nhãn supervision, không input inference.

**Lý do không mean cả sentence:** cùng bag tokens sẽ không phân biệt subject/object hoán đổi khi predicate giữ nguyên. Role-aware concatenation tránh lỗi biểu diễn tất định đó, không bảo đảm mô hình học đúng semantics. Frozen token pooling yếu hơn contextual encoder; pilot không được quy mọi failure cho raw CLIP hoặc mọi success cho semantic disentanglement.

## 4. Common scores, BCE và MP — mẫu số chính xác

Cho original fact f với observed y_f∈{0,1}, hai views v=1,2; h_fv là output F128 **trước value projection ở cả D/E**. Dùng score chung:

`z_fv = 10 * dot(normalize(h_fv), q_fv) + b`, b là một scalar trainable common, init0; normalize epsilon1e-6. Không fit temperature trên held. `z_f = (z_f1 + z_f2)/2`.

`L_BCE = mean_f [ (BCEWithLogits(z_f1,y_f) + BCEWithLogits(z_f2,y_f))/2 ]` trên original facts sampled. Không gán label thứ hai độc lập cho inverse view, không cân theo số source_rows, không double-count duplicate observations. Sampling tự nhiên theo facts, không class weights/label balancing hay tăng weight cho rare predicate sau nhìn held.

Một lexical group g là đúng `lexical_group` canonical đã tạo trong CPU artifact, không tự recanonicalize theo detector order. Gọi P_g và N_g là **toàn bộ original fit facts** có cùng lexical group, tương ứng observed true và false. Hai tập này có thể ở các ảnh khác nhau. Chỉ58 groups có cả P,N được dùng cho MP. Loss một group:

`L_MP(g) = -(1/|P_g|) * sum_{f in P_g} log[ exp(z_f) / sum_{u in P_g UNION N_g} exp(z_u) ]`.

Dùng logsumexp ổn định. Đây là standard multi-positive softmax/ranking với mỗi observed positive có numerator riêng; các positive khác trong mẫu số vẫn là positive targets qua trung bình, không bị gán binary false. Lower bound khi negatives bị tách và positives bằng nhau là log|P_g|, không0; log raw loss và excess `loss-log|P_g|` để tránh đọc sai learning curve. Không gọi đây là CLIP đối xứng batch hoặc loss mới.

Mẫu số **chỉ** P_g∪N_g có nhãn thật trong fit. Không thêm ảnh khác query, batch unknown, unannotated predicate, mismatched proposal, absence trong reference, opposite orbit chưa quan sát hay arg-swap vào negatives. N_g negative chỉ cho query tương ứng, không negative chung của toàn ảnh. Không trộn held16 groups vào sampler hoặc normalization. Nếu artifact thực tế thiếu P/N của group thì fail receipt, không silent skip/replace bằng in-batch negatives.

Tính toàn group mỗi lần, không subsample negatives. Có thể microbatch exact scores nhưng không detach hoặc thay mẫu số khi nối; log IDs/P/N counts. H chỉ cần raw pooled visual features/predicted geometry/predicate và F, không cần chạy native caption decoder cho auxiliary images. Không load fit memories ở dev inference. Frozen raw fractional pooled features có thể cache tính toán, nhưng các projection trainable phải được recompute, không cache stale h/q.

Giám sát chung trước projection cố ý làm D mạnh. Nếu D học hai h gần bằng nhau thì hard sharing có thể dư thừa: đó là kết quả hợp lệ, không thêm một loss E-only hoặc ép keys khác nhau để cứu giả thuyết.

## 5. Caption objective, sampling và endpoint

**Một phase, không sweep:** 500 optimizer updates; global caption batch8 theo đúng ordered4000 rows CPU/C0, một lượt. Giữ native sampler/greedy/scoring/SCST policy normalization của C0. Reward gốc = CIDEr + object reward với coefficient .5 như run_concept.py; không thêm reward quan hệ hoặc fine-tune CE phase.

`L = L_SCST + 0.1 L_KL(C0 || student) + 0.1 L_BCE + 0.05 L_MP`.

KL dùng student sampled prefixes, frozen raw C0 logits, per-sequence mean trên active tokens rồi mean batch như C0. Không KL qua ảnh khác hoặc teacher có branch. Teacher C0 thay teacher A của recipe lịch sử là lựa chọn anchor riêng được chốt ở đây để neo đúng own initialization. Keep/negative concept-mass weights=0; không backprop extra reference CE. Fixed weights này là operational choice, không optimum đã chứng minh.

Mỗi update đồng thời:

1. Một caption batch8; data order/seed schedule giống hai arms. Sampled captions sau cập nhật được phép khác vì policy khác; không buộc replay D captions sang E.
2. Auxiliary BCE batch8 từ401 fit facts: deterministic seed42 shuffled cycle, mỗi fact một lần trước reshuffle; wrap sang cycle sau để đủ batch, không drop tail. 4000 fact presentations tổng cộng ≈10 epochs; báo chính xác exposures/family/truth counts, có nguy cơ overfit. Không pretrain auxiliary một phase riêng.
3. Một MP group từ58 fit groups theo deterministic independently seeded shuffled cycle; toàn bộ members, mean group loss. Tổng500 group presentations, không coi các members đó là BCE events bổ sung. Sampling streams độc lập caption để không chỉ học19 overlap images.

AdamW: native decoder last2 layers lr5e-6; toàn bộ new trainables lr1e-4; weight_decay.01, betas(.9,.999), eps1e-8, global clip norm1.0. Constant lr, no scheduler hoặc parameter-dependent tuning. Cùng optimizer groups/init/state0/AMP/backend/batch accumulation ở D/E. DDP global means phải chính xác; lặp group trên cả ranks không nhân đôi weight. Endpoint500 duy nhất; checkpoint trung gian chỉ để recovery/debug, không chọn theo held/dev. Các step1–3/25-multiples log raw/weighted losses, gradient norms, norm mean/std của h/V/context/Wo-context, per-family/MP exposures và finite checks. Chi phí gradient diagnostics chỉ ở vài điểm cố định, không từngstep.

Auxiliary losses train F/text path, không bảo đảm Wv/readout giữ semantic content. Wo0 làm caption gradient tới F/Wq/Wk/Wv bằng0 lúc bắt đầu là dự kiến; aux vẫn có thể cập nhật F. Không xem gradient0 ban đầu là hỏng, nhưng sau khi readout mở phải kiểm connectivity. Không reset Wo/E branch giữa train và evaluation.

## 6. Preflight và trần chi phí user-run

Đây là spec để root chuẩn bị notebook; **người dùng chạy**. Root pin mount/version, actual file SHA/payload/sample shape/finite và common init/data/token identities trước model. GPU parity/gradients chưa chạy. Không dùng API tự start Kaggle.

Preflight ngắn trên fixed fit IDs, không held labels để chỉnh config:

- C0 native logits/greedy parity với D/E ở Wo0, eval mode, cùng masks/tokenizer; null-only path parity. Param/scope/init equality; detector indices/orbit partner involution và không cross-instance averaging. Count slots<=91, no GT/class/truth path vào caption forward.
- FP32 E paired-value equality và key-only swap equality của context (atol1e-6/rtol1e-5); full logits atol1e-5/rtol1e-5. AMP chỉ kiểm thêm với atol/rtol1e-3, log actual maxima/greedy margins; FP32 là construction reference. Không sửa tolerance sau thấy lỗi. Key swap test phải có nonzero temporary readout hoặc kiểm context trực tiếp, tránh PASS rỗng do Wo0.
- Throwaway reference-CE backward với nonzero temporary Wo để kiểm finite nonzero reachable gradients tới F/Wq/Wk/Wv/Wo và decoder last2; aux loss gradient tới F/text, frozen native tensors không đổi. Đây chỉ connectivity test, không thêm CE training phase. Restore byte-identical init, optimizer/RNG states trước training.
- Time/memory benchmark20 optimizer-equivalent steps của **cả train objective và aux group**, fixed fit/caption IDs; reset đầy đủ sau benchmark và không dùng benchmark weights làm init. Không dùng toy caption-only tốc độ để forecast full run. Nếu OOM, numeric/parity/connectivity FAIL thì dừng artifact, chưa train; sửa implementation cùng spec rồi preflight lại, không đổi scientific endpoint silently.

Quota user báo gần20h/account, không mặc định walltime=quota hoặc T4x2 tính một giờ. Đặt **hard wall cap7200s/arm cho GPU preflight+train+GPU inference**, CPU scorer riêng<=1800s; thêm trần **<=5 quota-hours/arm** nếu biết cách nhà cung cấp hạch toán, luôn giữ tối thiểu15h/account theo số báo hiện tại. Đây là cap, không ETA/đảm bảo runtime. Root ghi accounting convention hoặc UNVERIFIED, không suy đổi2GPU tự động.

Trước bước train thật, benchmark phải forecast500updates + full dev generation + held diagnostics + overhead trong80% cap còn lại; chừa phần còn lại cho save/log/close. Nếu forecast vượt cap hoặc chưa pin mount/cache integrity thì STOP_PREFLIGHT_NO_TRAINING. Không tự rút500 thành200, tăng deadline hoặc mở thêm tài khoản để gọi completed. Cap timeout giữ partial artifacts/status INCOMPLETE; không so endpoint500 với partial arm. D/E có thể chạy mỗi arm một account theo user, cùng hardware/runtime pinned; root không tự hứa cả hai sẽ xong trong hạn.

## 7. Evaluation tối thiểu, stop/go và giới hạn

Không dùng held để train/calibrate/early-stop. Evaluate raw C0 và final D/E ở endpoint đã khóa. Dev1000 là reused development set, không pristine test; held99 chỉ held khỏi adaptation mới, C0 có thể đã thấy ảnh trong tiền huấn luyện trước. Không dùng expanded4000 hoặc tăng n để cứu pilot.

**A. Construction:** tất cả preflight identities/finite/connectivity PASS. Đây là math/implementation, không learned semantics. FAIL => stop implementation; không nhận method result.

**B. Observed semantics trên99 held facts/16groups:** báo BCE, AUROC theo contact/containment và macro-family, cùng query-conditioned concordance: với từng held group, fraction(z_true>z_false)+0.5ties qua toàn P×N; macro16groups. Không pool positives/negatives khác query để nhận query-controlled effect. Bootstrap theo image IDs để CI; báo groups mất một truth class trong resamples và số draws hợp lệ, không giấu sparse support.

Báo cả preprojection `z_f` và **effective-content diagnostic**: D r_fv=h_fv; E r_fv=(h_f1+h_f2)/2; thay normalize(h) bằng normalize(r) trong score mục4, dùng text projection đã học, không fit probe/head mới. Diagnostic này kiểm phần content sau averaging, chưa bảo đảm Wv/decoder dùng nó. Preprojection tốt nhưng effective score sụp thì không nhận semantic success. Log value norms/variance/nonzero rate; equality hoặc nonzero distance không phải truth evidence. Distinct-instance/opposite-orbit truth separation chỉ tính nơi có nhãn quan sát hỗ trợ; không đủ event thì ghi NOT_MEASURED, không tạo false label. Đặc biệt không nhận generic direction test PASS từ P3.

Screen vận hành được khóa trước model: E effective macro-family AUROC>=.60 và within-query concordance>=.60; report paired E−D và bootstrap CI, không gọi ngưỡng này là significant effect/power gate. D tốt tương đương là lý do projection có thể thừa. Nếu không qua hoặc collapse thì STOP current candidate; không scale/tune. Nếu uncertainty không phân biệt E/D, kết luận inconclusive, không equivalence/no-effect.

**C. Actual decoder/caption:** cùng fixed native greedy, full1000 dev C0/D/E; CIDEr, CHAIR hallucinated mentions/captions/counts/ratios, object recall, mean words, paired image-bootstrap CI và exact paired false-caption counts. CHAIR không đo relation correctness. Raw C0 full dev chỉ cần tính một lần cùng pinned runtime/scorer và dùng chung đã kiểm identity, không lấy scalar C0 từ cohort cũ khác. Kiểm C0 sentinels ở cả runtimes nếu chạy hai accounts.

Trên held99, thêm cheap teacher-forced diagnostic qua actual native decoder: mean active-token log P(query_view | image), mean hai equivalent views; tính same-query P×N concordance cho C0/D/E, không ép hai sentence probabilities bằng nhau. Branch vẫn chỉ enumerate P3 từ ảnh, không lấy query nouns/future tokens để chọn memory. Đây là supporting readout diagnostic; không thay free-caption result hoặc claim factual generation.

Trên fixed64 dev IDs chọn hash trước train, chạy reader-off và value-zero ở endpoint như **một intervention duy nhất** (Wo(context)=0, giữ adapted decoder); báo changed captions/logit KL/norm. Reader-off không phải raw C0, không retrain. Đây kiểm branch có ảnh hưởng, không bảo đảm ảnh hưởng hữu ích. Nếu cả1000 free captions E=D, không added caption utility dù semantic scores đẹp. Nếu chỉ diagnostics thay đổi nhưng free caption/metrics không có ích, không nhận thành công.

Free captions có thể không nói tới P3. Lưu C0/D/E captions trên99 held images và đối chiếu những relation assertions có thể bind rõ với observed query/label; unsupported/unmentioned/ambiguous giữ UNKNOWN, không tính silence là đúng. Nếu không đủ assertion được kiểm hoặc không có human-verified parsing, ghi relation-caption truth NOT_ESTABLISHED; teacher score/CHAIR không lấp chỗ trống đó. Không tự tạo annotation campaign để cứu pilot.

**Chỉ đề nghị bàn lượt xác nhận tiếp**, không auto-run, khi cả B đạt, branch có dùng, CIDEr E−D và E−C0 có point gain, false-caption count E<C0 và E<=D, recall không giảm quá.01 absolute và mean words trong[.95,1.05]×C0. Đây là pilot screen, không proof tăng CIDEr/giảm hallucination: phải báo CI dù cross0; không gọi thắng/đủ bar công bố. Gain chỉ vsC0 mà không vsD không ủng hộ projection. Hallucination/recall/length tradeoff, không added utility theo screen, hoặc sparse/inconclusive => không tự mở rộng/tuning, lưu kết quả và khuyến nghị dừng đầu tư instantiation hiện tại trừ khi user chủ động chọn mục tiêu khác. Không gọi CI cross0 là equivalence hoặc bác bỏ cả họ value-sharing. Nếu qua screen nhưng paired evidence yếu, nói rõ chỉ đáng thảo luận confirmatory budget; không xem thêm hạt giống/n lớn là việc mặc nhiên đã duyệt.

## 8. Bàn giao / chưa làm

Root triển khai đúng spec, kiểm metadata/static/mock cần thiết, pin code/data/config/common-init/objective/sampler/token hashes; người dùng tự chạy preflight/training. Hai output phải có endpoint identity hoặc INCOMPLETE, losses/exposures/gradient/norm logs, captions/semantic scores, receipt/runtime/timing/quota, model hash và review ZIP không tensors. Không xuất cả raw cache vào review ZIP.

Vòng này chỉ đọc protocol/source liên quan và compact audit của root, viết spec và static diff check. Không đọc lại ZIP/primary papers, không HDF5/checkpoint/model/Torch/autograd/Kaggle, không đo metric/runtime mới, không thêm nhánh. Root phụ trách commit/push/registry; không nhận đã phân phối từ file này.
