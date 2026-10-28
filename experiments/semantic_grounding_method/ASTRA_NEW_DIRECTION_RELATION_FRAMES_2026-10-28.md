# Hướng mới: học khung quan hệ từ phản chứng có nhãn, rồi đưa khung vào decoder

> DATA UPDATE28/10: exact image-hash/quota policy v1 không qua box-coverage gate; kết quả ấy giữ nguyên. [ASTRA_RELATION_DATA_V2_DECISION_2026-10-28.md](ASTRA_RELATION_DATA_V2_DECISION_2026-10-28.md) thay riêng phần chọn cohort và per-relation quota trước learning bằng pair-first/image-disjoint stratified evaluation. V2 vẫn UNVERIFIED cho tới CPU feasibility report; không tự coi data gate PASS hoặc launch từ thay đổi này.

28/10/2026, Asia/Bangkok. Đề xuất nghiên cứu để root phản biện và user chốt; **chưa code/GPU**. Thay deliverable brainstorm trước: T/Energy diagnostic và duration T không còn là hướng được đề xuất ở lượt này. Không sửa hay che kết quả T500.

## Quyết định và giả thuyết

**Chọn học biểu diễn khung quan hệ có thứ tự bằng các cặp ảnh có cùng hai loại đối tượng, cùng phát biểu nhưng nhãn đúng/sai khác nhau; sau đó cho caption decoder đọc trực tiếp các khung đã học.** Nguồn phản chứng đề xuất là Visual Spatial Reasoning (VSR), không phải UPT hoặc cặp winner/loser theo CIDEr.

Đây là thay đổi semantic unit, visual representation và nguồn giám sát so với T. Không có IG, contribution accounting hoặc loss hướng net gain. Khả năng là một phương pháp hữu ích có cơ sở hơn việc đổi threshold T; **originality và caption gain đều chưa được xác lập**. Tôi không nhận subject–object–union, contrastive learning hoặc relation memory là ý tưởng đầu tiên.

Failure muốn sửa: mô hình biết hai đối tượng xuất hiện nhưng sinh một quan hệ hợp ngữ cảnh thay vì quan hệ thực sự có trong ảnh. Trong hệ thống A, audit trước cho thấy một số lỗi vị trí/tư thế, trong khi kiểm danh từ không bắt được chúng. Đây là bằng chứng có lỗi loại này, **chưa chứng minh đây là lỗi chủ đạo hoặc nguyên nhân nằm riêng ở encoder/decoder**. Các thử nghiệm noun/name và attribution trước cũng chưa học từ nhãn phản chứng quan hệ sạch.

Giả thuyết mới: *biểu diễn được huấn luyện phải phân biệt hai ảnh có cùng entity concepts nhưng khác truth value của cùng quan hệ sẽ mang thông tin quan hệ tốt hơn biểu diễn chỉ học tái tạo mô tả dương; decoder được fine-tune đồng thời có thể chuyển thông tin đó thành caption.* Hai vế đều phải đo.

## 1. Hai hướng khác nhau thực chất

| Hướng | Can thiệp chính | Ưu điểm | Điểm yếu | Chọn |
|---|---|---|---|---|
| **Khung quan hệ học từ phản chứng** | Học visual pair representation, rồi thêm decoder reader cho khung quan hệ | Có supervision mới thật sự; giữ được thứ tự vai và bằng chứng tương tác; so positive-only với contrast dễ định nghĩa | Thêm data/feature preparation; detector lỗi; reader có thể bị bỏ qua; prior về relational captioning rất gần | **Có, pilot hữu hạn** |
| **Critic quan hệ làm reward cho caption tự sinh** | Giữ kiến trúc A, đổi mục tiêu SCST bằng grounded relation critic | Tác động trực tiếp free generation; không thêm inference memory | Reward hacking, bỏ quan hệ hoặc chuyển thành “near”; calibration từ template sang caption tự sinh; parser/coverage khó | Không chọn đợt này |

Hướng critic: train `d(I,q)` từ nhãn VSR; tách spatial assertions Q(y) trong caption, có thể dùng `R_rel=sum_unique_q (2 sigmoid(d(I,q))-1)` rồi `L=-E[(R_A+lambda R_rel-b)log p_theta(y|I)]+beta KL`. Unmapped/unknown assertions không được gán sai. Closest prior về tối ưu caption theo evaluator là DiCO: đã có evaluator-derived reward và neo mô hình; phần mới sẽ phải nằm ở granular truth supervision, không phải đổi reward. Đọc primary DiCO §3, Eq. (1)–(6): [BMVC 2024](https://bmva-archive.org.uk/bmvc/2024/papers/Paper_754/paper.pdf). Chưa có bằng chứng evaluator của DiCO thất bại trên A; điểm yếu liên quan là global matching không trực tiếp cung cấp truth labels cho từng quan hệ.

Minimum falsifier của hướng critic sẽ là frozen-critic evaluation trên spatial clauses tự sinh có chấm độc lập: nếu không phân biệt được sai quan hệ với diễn đạt hợp lệ/thiếu mô tả, không dùng reward. Sau đó tối thiểu cần A-tune và A-tune+critic, cùng SCST/KL. Dữ liệu/cache cho critic giống hướng chọn, nhưng cần thêm clause extraction và on-policy calibration. Không chạy vì thắng benchmark VSR chưa đủ chứng minh reward an toàn khi bị policy khai thác.

## 2. Ba tầng căn chỉnh và semantic unit

- **Định vị:** nối entity concepts của assertion với hai instance riêng. Box đúng tọa độ chưa bảo đảm đúng referent. Dùng COCO annotation chỉ để kiểm nhãn/match lúc train và evaluation, không làm input inference.
- **Ngữ nghĩa:** đơn vị là frame `(subject instance, spatial predicate, object instance)`, ví dụ `(cat, under, table)`. Không gán “under” vào một bbox; không coi union box không thứ tự là đủ.
- **Sinh câu:** representation của frame đi vào cross-attention thật của decoder và nhận gradient từ chính SCST caption objective. Phải chứng minh decoder đọc được thông tin mới, ngoài accuracy của auxiliary classifier.

Visual unit gồm ba nguồn: **vùng chủ thể, vùng đối tượng và patch interaction trong union**, cộng hình học có dấu. Một union vùng lớn có thể chứa nền không liên quan; đây là evidence candidate, không phải object-interaction certificate. Pilot tập trung **quan hệ không gian**, chưa nhận giải quyết holding/riding, attributes hoặc mọi quan hệ sự kiện.

## 3. Supervision mới có thể lấy từ đâu?

**VSR primary đã kiểm:** Liu, Emerson & Collier, TACL 2023; đọc §3.1–3.5 và schema repo. Dataset có 10,972 image–statement records sau validation. Khi tạo dữ liệu, hai ảnh đều chứa hai concepts; caption được chọn đúng ở một ảnh, sai ở ảnh kia. Nhóm tác giả lọc multi-instance ambiguity và dùng validation bổ sung. Release có image URL, relation, label và validator votes. Các nhãn vẫn có noise; không gọi gold tuyệt đối. [Paper](https://aclanthology.org/2023.tacl-1.37.pdf), [repo](https://github.com/cambridgeltl/visual-spatial-reasoning), [schema mẫu](https://raw.githubusercontent.com/cambridgeltl/visual-spatial-reasoning/master/data/splits/random/train.jsonl).

**Nguồn dương/âm:** chỉ lấy records có label đúng/sai đã được validate. Ghép cùng exact normalized statement, có cả một ảnh label1 và một ảnh label0; hai ảnh phải có cùng subject/object concepts. Không tự đảo relation/role để gán sai. Không dùng missing VG/UPT label làm negative; không lấy randomly mismatched caption làm factual negative.

**Chính sách pilot trước khi xem model scores:**

1. Dùng metadata validated, yêu cầu ít nhất ba validator cùng đồng ý với released label và không có validator phản đối. Nếu không đủ lượng, báo thiếu; không nới theo kết quả.
2. Bỏ relation phụ thuộc reference-frame trong pilot: left/right và các biến thể, in-front/behind, ahead/back, orientation facing. VSR cho phép đúng dưới intrinsic hoặc relative frame; không được thay bằng quy tắc center-x/center-y. Chọn từ tập định trước: `on, under, above, below, on top of, beneath, in, inside, within, outside, contains, touching, next to, near, far from, far away from`. Giữ các relation riêng, không tự gộp như synonyms.
3. Match concepts theo COCO80 categories và checked template span; tên nhiều từ được giữ nguyên. Recheck unique GT instance của mỗi concept. Dùng **predicted boxes** đã match class/IoU>=.5 với GT lúc chuẩn bị. Input cả train/inference là predicted boxes; detector miss thì bỏ toàn bộ supervision pair và ghi count, không biến thành false label.
4. Exclude mọi ảnh trong locked caption dev/val/test khỏi relational TRAIN. Chia relational images trước rồi mới ghép pairs; mọi record của một ảnh ở cùng split. Không dùng random record split như thể bảo đảm image-disjoint. Không trộn từ các split chính thức mà không ghi đây là derived training/evaluation cohort.
5. Khóa pool hash-first tối đa512train candidate pairs và128held-out candidate pairs trước feature extraction. Sau các data/detector guards, hash-select **256 training pairs + 64 image-disjoint held-out pairs**, ít nhất bốn relation, mỗi relation có >=16 train pairs và >=8 held-out pairs. Đây là pilot target, **chưa biết count sau joins**. Nếu pool cố định không đủ, dừng cấu hình dữ liệu này, không claim đã có đủ chỉ vì tổng dataset >10k. Không gọi split này zero-shot concepts.

**Đã có:** A, original caption caches/TRAIN2k/dev1000; detector cache trên phần dữ liệu cũ; COCO weak name/box bank. **Cần thêm:** VSR metadata, verified COCO instance mapping cho ảnh mới, ảnh subset còn thiếu, frozen CLIP patch features và frozen detector boxes cho những ảnh đó. Không phải tải/train một SGG mới hoặc toàn bộ COCO. Release có direct image URLs; lấy subset có hash/receipt. Feature extraction mới chỉ user-run trên Kaggle sau duyệt.

Quan trọng: phase học frame chỉ dùng CLIP features + predicted boxes, **không cần original semantic prompt của A cho mọi ảnh VSR mới**. Caption adaptation vẫn dùng caches chuẩn cũ. Nếu cần full-decoder probe trên VSR ngoài cache, phải chạy **đúng original prompt builder** cho 128 probe images; không thay bằng GT-name prompt. Prerequisite này phải ghi rõ nếu chưa resolve.

## 4. Input → module → decoder

### 4.1 Mã hóa khung có thứ tự

Frozen CLIP cung cấp grid V; detector frozen cung cấp tối đa12 boxes/class predictions. Không dùng score UPT làm quan hệ thật. Với mọi cặp có thứ tự i!=j:

`e_i = W_e ROIpool(V,b_i)`; `e_j = W_e ROIpool(V,b_j)` (d=256).

`u_ij = AttentionPool(query=[e_i;e_j], keys/values=V_union, two_role_masks=[M_i,M_j])`.

`r_ij = MLP([e_i; e_j; u_ij; g_signed(b_i,b_j); emb(class_i); emb(class_j)])`.

Hai masks/channels tách biệt; g có offset có dấu, scale/overlap, không suy semantic predicate từ geometry một cách cứng. Pair query giữ thứ tự. Phần này kế thừa subject/object/union modeling; không nhận là mới. Không append predicted relation sentence vào prompt.

### 4.2 Học representation và liên kết ngôn ngữ

Một GRU nhỏ dùng `r_ij` để tái tạo assertion đúng, dùng word embeddings và vocabulary10000 của A cố định. Chuỗi template là pretraining target, **không trộn thành COCO caption CE**. L_pos là trung bình ba CE theo lexical spans subject/predicate/object; không để các từ “the/is” chi phối loss. Whole statement OOV/invalid thì skip pair và count; không đổi vocab.

Một multi-label truth head cho `d_r(I,i,j)` — logit cho predicate r. Khi chỉ biết một assertion, chỉ head của predicate được annotate nhận BCE. Các predicate khác là unknown, không set0.

Với cùng statement q ở ảnh I+ và I-:

`L_truth = .5[BCEWithLogits(d_plus,1)+BCEWithLogits(d_minus,0)]`;

`L_pair = softplus(-(d_plus-d_minus))`;

`L_frame = L_pos + .5 L_truth + .5 L_pair`.

Không có attribution hoặc contrast theo tên đối tượng đơn. Hai ảnh đều có hai concepts, nên chỉ nhớ bag-of-object labels không đủ phân biệt nhãn của cùng q. Background/style shortcuts vẫn có thể tồn tại.

### 4.3 Đưa frame vào decoder đang sinh câu

Giữ memory gốc `[P_vis;Q]`. Ở decoder layer cuối, thêm **một reader riêng trên frame bank**:

`c_global = original cross-attention(h_t,[P_vis;Q])`;

`c_frame = Attn(h_t, keys=K(e_i,e_j), values=V(r_ij))`;

`c_t = c_global + W_frame c_frame`, rồi qua output/residual/FFN cũ.

W_frame khởi tạo0 chỉ để tái lập A lúc bắt đầu; đây là kỹ thuật initialization, không phải novelty/gate mới. Không cộng delta logits và không đóng băng decoder như T. R riêng biệt giúp không buộc frame cạnh tranh cùng softmax với toàn bộ memory gốc. Detector/CLIP frozen; original projection, Q-Former, prompt branch, decoder layer1 và vocabulary/word embeddings frozen; **layer2–3 decoder, frame reader/encoder và auxiliary heads trainable**. Cần static parameter count thật trước packaging; dự kiến khoảng vài triệu đến hơn8M trainable, không phải262k.

Caption objective giữ recipe chuẩn: `L_cap = SCST(R_A) + .1 KL(p_A||p_theta)` trên sampled prefixes. Replay một minibatch nhỏ VSR mỗi update với `.1 L_frame` để giữ semantics; auxiliary replay chỉ đi qua frame module/GRU/truth head, không cần A prompt. Decoder nhận gradient từ actual caption SCST, không chỉ từ một attention heatmap loss.

**Tại sao có thể giúp:** frame values đã học phân biệt counterimages thay vì chỉ chứa feature HOI thô; decoder được phép học cách đọc chúng dưới cùng mục tiêu chuỗi như A. Đây là lý do mới kiểm được để thử relational information sau UPT-memory/noun-head failures. Không giả định thêm thông tin tự động cải thiện caption.

## 5. Closest prior và claim có thể bảo vệ

| Primary đã đọc | Kế thừa, không nhận mới | Khác biệt ứng viên / giới hạn |
|---|---|---|
| **MTTSNet, CVPR2019**, §3.1–3.2, Eq.(1)–(2) | Separate subject/object/union features, geometry, relational caption/POS generation | Objective positive relational descriptions không tự cung cấp same-concepts true/false counterimages. Đề xuất học thêm distinction ấy, giữ frame làm input của full-image caption decoder. Chưa đo MTTSNet sai ở đâu trên cohort này. [PDF](https://arxiv.org/pdf/1903.05942) |
| **SGVL, EMNLP2023**, §3.2–3.4 và Appendix B.2 | Structure-aware visual tokens, graph supervision và hard negatives; B.2 đã giới hạn role swaps vào relations được tác giả đánh dấu asymmetric | Không được nói SGVL bỏ qua symmetry. B.2 còn dùng rule-based relation/attribute replacement; chưa thấy human validation từng counterexample trong quy trình đã đọc. Khác biệt ở provenance của explicit human-validated cross-image labels và chuyển frame vào caption decoder; **chưa đo negative error rate của SGVL**, không tuyên bố các negatives ấy sai. [PDF](https://aclanthology.org/2023.emnlp-main.870.pdf) |
| **Dai & Lin, NeurIPS2017**, phần Contrastive Learning | Contrastive caption optimization/reference model đã có | Mismatched-caption noise không tự xác nhận sai một relation khi objects giống nhau. Ở đây đối chiếu cùng statement với true/false images, ở regional frame representation; không nhận contrastive learning là mới. [PDF](https://arxiv.org/pdf/1710.02534) |

CSA-Net là tham khảo về semantic prompting, nhưng lượt này chỉ xác minh được author metadata/official repo, **chưa có full method text**; không dựa vào abstract để khẳng định CSA-Net thiếu một thành phần cụ thể. [Official repo](https://github.com/CrossmodalGroup/CSA-Net).

**Claim ứng viên vừa mức:** một scheme chuyển grounded relational counterevidence vào autoregressive captioning, với ordered frame và same-concept/statement cross-image supervision, hơn cùng architecture chỉ học positive descriptions. Chỉ có thể bảo vệ nếu lợi ích vượt positive-only và decoder-tuning controls, kèm evidence decoder dùng frame đúng. Chưa có cơ sở nói first/new SGG/new contrastive objective hoặc đã vượt mọi closest prior. Khác T về bản chất không đồng nghĩa mới tuyệt đối trong literature.

## 6. Shortcut và phản ví dụ phải giữ trong quyết định

- **Truth classifier giỏi nhưng decoder bỏ reader:** kiểm actual caption/probe ablation. Classifier accuracy một mình không phải thành công.
- **Đoán relation từ classes/background:** same-statement counterimages loại được object-label-only rule, không loại hết image-style bias. Split image-disjoint; metadata/annotator IDs không vào model. Paper cần unseen concept-pair test.
- **Geometry-only:** above/below có thể giải từ boxes; đó là cue hợp lệ nhưng chưa chứng minh đọc interaction appearance. So geometry-only baseline/appearance ablation trước claim rộng về visual interaction; nếu geometry đủ, thu hẹp claim.
- **Role symmetry/reference frames:** không giả đổi vai luôn sai; near/next-to có thể đối xứng. VSR ambiguity policy phải giữ, không relabel từ tọa độ. Pilot không học frame-sensitive relations nêu trên.
- **Global CLIP tokens đã trộn thông tin:** ROIpool không bảo đảm region-pure. Không gọi patch/union là causal local recognition; paper cần stronger local feature/crop control nếu muốn claim ấy.
- **Detector domain gap và thiếu pair:** train branch trên predicted boxes đã match, không teacher-GT boxes rồi giả inference tương đương. Report coverage; bỏ pair không đồng nghĩa ảnh/caption bị bỏ khỏi caption objective.
- **Template overfit / tăng câu dài / quan hệ dễ:** pretrain templates chỉ cho module; COCO adaptation vẫn SCST, giữ length/object recall/CHAIR guards. Một model chỉ thay mọi relation bằng “near” chưa đạt semantic goal.
- **Nhãn VSR vẫn có lỗi:** unanimity không phải guarantee. Audit nhỏ 32 records trước train, không cherry-pick theo score. Nếu có lỗi ontology/mapping có hệ thống thì dừng pipeline, không sửa riêng examples để đủ gate.

## 7. Pilot ít nhánh nhất và go/no-go

### Dữ liệu và pretraining, không phải một vòng tìm hướng khác

Gate data: đủ256train/64held-out paired statements theo policy; raw features/boxes/prompt-probe provenance resolve, 32-record audit không có lỗi mapping/reference-frame có hệ thống. CPU/network preparation cap90phút; freeze manifest trước learning. Thiếu thì báo **data-incomplete của đề xuất này**, không tự chuyển sang UPT negatives hoặc quay lại T.

Hai frame modules cùng initialization/data order, **200 pretraining updates**, batch16 assertion pairs, AdamW lr1e-4, wd.01:

- **RelPos:** L_pos trên positive descriptions; không truth/negative training.
- **RelContrast:** L_frame đầy đủ.

Đánh giá cùng GRU assertion likelihood `log G(q|r(I+))-log G(q|r(I-))`, không lấy accuracy của truth head chưa được train ở RelPos làm đối chứng. Pretrain gate exploratory: RelContrast correct-image ranking>=70%, hơn RelPos>=10 percentage points và paired CI lower(diff)>0 trên fixed64 pairs. Chấm checkpoint200 duy nhất. Nếu fail, dừng recipe, không chọn epoch tốt nhất.

### Ba caption arms — A chỉ là baseline frozen

1. **Tune:** A mở layer2–3, SCST+KL, không frame bank.
2. **RelPos:** cùng tuning, thêm reader + positive-only frame module/replay.
3. **RelContrast:** cùng tuning/architecture/initialization, thay bằng counterexample-trained module/replay.

Cả ba **1.000 caption updates**, same TRAIN2k/caption order/seed, global batch8, new modules lr1e-4, decoder lr5e-6; endpoint1000 duy nhất. Same original decoding/evaluator/reward receipts; không templates CE vào caption phase, không sweep. A parity phải verify trước updates. RelPos/RelContrast có cùng parameter count và new-data cohort; khác negative supervision, không khác detector/crop/resolution.

**Caption gate:** RelContrast-A>=+.01 CIDEr; paired95% CI lower(RelContrast-A), lower(RelContrast-Tune), lower(RelContrast-RelPos)>0; CHAIRs/i<=A; object recall>=A-.005; mean length không co hơn .4 từ. 10k paired image bootstrap chỉ là uncertainty conditional một seed/reused dev.

**Decoder-use probe:** 64 held-out VSR pairs, score normalized predicate+object tail likelihood từ cùng subject prefix trong actual caption decoder (cần original prompt builder cho probe images). Swap **frame VALUE của target pair** giữa I+/I-; giữ recipient global memory/keys/boxes, đúng entity classes. Không shuffle toàn bộ key-value rows vì attention bất biến theo row permutation. Correct-image likelihood gap phải tốt hơn RelPos và giảm khi đưa wrong counterpart value, paired CI lower cho hai contrasts>0. Đây là computational intervention trên reader, không pixel counterfactual. Nếu prompt path unavailable, mark probe unavailable và chưa nhận mechanism pass.

**Free-caption semantics:** thêm 64 hash-fixed dev images, chấm blind A/Tune/RelPos/RelContrast cho spatial claims: supported / contradicted / uncertain / no spatial claim; report counts và coverage, không coi omission là error. Require không tăng contradicted claims và số supported claims>=A. Người chấm độc lập chưa được bố trí; nếu chỉ assistant audit, ghi exploratory, chưa dùng làm independent human evidence cho paper. CIDEr gate vẫn đo free generation; likelihood probe không thay được nó.

Fail data/numerics/budget = incomplete, không bác giả thuyết chung. Accuracy tăng nhưng caption không vượt controls = không đạt mục tiêu captioning. Caption tăng mà RelPos/Tune tương đương hoặc reader không dùng = chưa xác lập đóng góp counterevidence. Dừng cấu hình, giữ A; không rescue bằng tăng updates hoặc đổi negatives trong cùng pilot.

## 8. Thời gian, compute và phạm vi paper

**Pilot có thể chuẩn bị trước31/10, không cam kết kết quả dương.** Phụ thuộc mới là subset download/frozen feature extraction, không chỉ existing caches. Pool preparation tối đa1.280 VSR images trước dedup, cohort học/đo cuối tối đa640, thêm probe prompts cho128 images; không full-dataset SGG training. Box matches/exclusions có thể làm chưa đủ lượng; không giả định đã đủ.

GPU caps, không ETA: feature/detector/probe preparation<=1 device-hour; hai frame pretrains tổng<=1; ba caption arms mỗi arm<=45phútT4x2, tổng<=4.5; bootstrap/numerical/archive overhead và reserve trong tổng **8 device-GPU-hours**. Current per-account balance chưa biết. Không ngoại suy speed262k-head cũ cho decoder fine-tuning; preflight10updates thuộc1.000 phải dự báo đáp ứng cap, nếu không thì stop computational-incomplete. Không đổi batch/updates để khớp thời gian sau khi xem score.

- **28/10:** root phản biện module/data joins/loss; nếu user chốt thì khóa manifest và triển khai preparation.
- **29/10:** subset/feature/pretrain gate; nếu pass chạy ba caption arms cố định.
- **30/10:** endpoint comparisons, decoder-use probe và blind caption audit; chỉ cân nhắc confirmation nếu có gain đúng controls và budget riêng.
- **31/10:** chốt pilot artifacts/claim đúng evidence. Không mở thêm kiến trúc vì deadline.

**Để thành paper:** cần originality review sâu hơn đúng scheme này; matched strong relational/SGVL-inspired baseline với annotation budget rõ; nhiều seed; unseen entity-pair composition; geometry/role/interaction ablations; independent factual relation assessment và final evaluation tách tuning. Ba ngày có thể cho một pilot quyết định đáng làm tiếp; chưa đủ để hứa hoàn thành toàn bộ bằng chứng hoặc được công bố.

Đây là lựa chọn mới tốt nhất của tôi trong lượt được yêu cầu. Nếu chỉ còn point về áp dụng dữ liệu VSR cho một memory có sẵn mà không chứng minh representation/generation advantage, phải gọi là adaptation, không thêm tên mới để nâng claim. Chưa code, stage dataset vào workspace/Kaggle hoặc launch GPU.
