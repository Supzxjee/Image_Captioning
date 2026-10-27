# Đặc tả method candidate — Astra, 2026-10-27

Nguồn: ASTRA_cum_vung_semantic_readout_2026-10-27.md; thread 01a15544-6a7a-7903-a6e1-23e7b9f6caa6, turn 01a17b9b-c1c7-7f31-a437-e98015131f28. Chỉ là thiết kế, chưa ML code/run.

ROOT CLARIFICATION: Proj_A trong spec phải được hiện thực bằng đúng vis_projection + ReLU của A ở eval trước Q-Former, không chỉ lấy linear output. Correction qua W_vocab không cộng fc_out.bias lần hai.

---

# Ứng viên: căn chỉnh cụm–vùng trên chính semantic readout dùng để sinh caption

27/10/2026. Đặc tả nghiên cứu; chưa code ML, notebook, inference hoặc train. Đây là quyết định thiết kế, chưa xác nhận novelty hay caption gain.

## 1. Giả thuyết và phạm vi

Giả thuyết: một readout trước Q-Former, được học đồng thời về hướng ngữ nghĩa của cụm danh từ và vùng hỗ trợ cụm ấy, sẽ hữu ích cho caption generation hơn readout cùng capacity chỉ học theo caption reward. Cùng một vector được căn chỉnh và được dùng để hiệu chỉnh logits; không dùng attention correctness làm đại diện duy nhất cho utility.

Đơn vị ngữ nghĩa là noun phrase gồm head và modifier nội tại, ví dụ `a red bus`, `a young man`. Pilot không chấm predicate hold/ride, không học từ V-COCO action=0, không tuyên bố sửa quan hệ hoặc scope coordination. Không ép mỗi từ/function word vào box. Không dùng atlas20 làm gold, train hay gate.

## 2. Nguồn giám sát và eligibility

Nguồn: human COCO training references và COCO instance boxes, trên TRAIN đã khóa. Dùng nguyên captions để xây supervision; không dùng generated captions hoặc CIDEr để chọn positive. Không dùng noisy VG region annotations trong pilot đầu. Theo audit root chuyển: labelbankv1.1 đã verify sáu payload hashes,4495images/5602regions/17904spans/71classes,weak_supervision_not_gold=true. Các raw spans chỉ là noun heads/multiword names, CHƯA có full adjective NP; các counts này không được gán cho supervision mở rộng.

Pilot dùng pairing của labelbankv1.1 làm anchor, không tự gán lại box qua nearest text. Muốn có modifier supervision phải tạo derivation riêng từ đúng caption ID/string/offsets: spaCy3.8.7+en_core_web_sm3.8.0, chạy Kaggle CPU khi được triển khai. Chỉ nối các direct prenominal amod vào head/name span khi tạo thành chuỗi liên tục cùng một noun phrase; giữ nguyên multiword head/name. Không mở rộng PP, relative clause, coordination, số lượng, negation/comparative hoặc modifier không xử lý được; các ca ấy không nhận expanded alignment target. Không có adjective hợp lệ thì giữ head/name target và báo rõ class-only. Ghi source hashes, head span, derived phrase span, prefix IDs và lý do skip; đếm số targets có modifier riêng. Attributes là weak supervision lấy từ lời human reference, không phải attribute GT suy từ COCO boxes. Không claim fine-grained phrase contribution nếu supervision thực tế chỉ còn bare heads.

Ghép phrase với box chỉ khi noun head có alias COCO không nhập nhằng và ảnh có đúng một instance hợp lệ, non-crowd của class đó. Skip plural/cardinality, generic/ambiguous alias, nested relational phrase, depiction/negation hoặc trường hợp không rõ entity linkage. Không suy gender/attribute từ class label: modifier chỉ được cung cấp bởi human reference. Đây vẫn là weak phrase–box join theo quy tắc, không được gọi thành manual grounding GT. Inventory cũ 5602 regions chỉ là nguồn đã biết, không tự là số eligible của policy này.

Positive là cặp (phrase, box) được nguồn trên hỗ trợ. Thiếu hoặc không chắc pairing thì không có alignment target; ảnh/caption vẫn dùng trong caption objective. Không có semantic negative trong pilot: không coi reference khác, word foil, ảnh khác hoặc object không được mention là false.

CLIP text encoder đóng băng cung cấp vector 512 chiều đã normalize của `a photo of {phrase}`, cache một lần trên Kaggle. Nó biểu diễn meaning của text, không phải visual verifier hoặc chứng nhận phrase đúng trong ảnh. Pin checkpoint/tokenizer/template/truncation. Skip phrase bị truncate hoặc head không biểu diễn được trong A vocabulary. Vocabulary A giữ nguyên.

## 3. Giải quyết cùng-prefix/nhiều-next-phrase

Với mỗi image và exact A token-ID prefix trước content token đầu tiên của noun phrase, gom toàn bộ eligible (phrase, box) từ tất cả năm references có cùng prefix thành tập S(I,p). Deduplicate theo phrase/instance ID; thứ tự reference không làm tăng trọng số một cặp.

Forward không nhận phrase tương lai hoặc box. Nếu hai references có cùng prefix nhưng có thể chọn hai entity khác nhau, không tạo hai one-hot target xung đột cho cùng một attention. Loss marginalize trên S. Chỉ các prefix có cặp eligible mới nhận alignment loss. Nên một prefix có thể được căn chỉnh với bất kỳ lựa chọn đã được annotation hỗ trợ; không claim nó định vị đúng một next word chưa biết.

Tập S chỉ là tập positive đã quan sát, không được gọi là đầy đủ mọi caption continuation đúng. Marginal objective vẫn có thể ưu tiên một mode/generic phrase; không bảo đảm coverage, cần caption/recall endpoint. A-logit path vẫn có thể chọn một word không tương ứng feasible intent được alignment chọn, hoặc Wo có thể gần zero. Shared readout tạo đường ảnh hưởng, không chứng minh decoder dùng đúng semantic intent. Cần đánh giá grounding/factuality của phrases THỰC SỰ sinh ra riêng; không thay bằng loss trên reference prefixes.

## 4. Module và tensors

Giữ A hoàn toàn frozen, eval-mode; CLIP197x768, original projection, Q32x512, prompt20x512, decoder và vocabulary head không đổi weights. Inference dùng ảnh và original prompt như A.

Z=Proj_A(V_patch) có shape196x512: projected patch tokens trước Q-Former, bỏ CLS. h_t512 là đúng state A đưa vào vocabulary head sau prefix hiện tại. LN0 là LayerNorm không affine.

q_t=LN0(h_t) Wq; K=LN0(Z) Wk; V=LN0(Z) Wv.

Wq,Wk:512x128; Wv:512x512. Không bias/dropout. a_t=softmax(q_t K^T/sqrt(128)); r_t=sum_j a_tj V_j; s_t=r_t/max(norm(r_t),epsilon).

logits_t=logits_A,t + W_vocab,A (s_t Wo).

Wo:512x512, không bias, khởi tạo zero; ba ma trận còn lại nonzero initialization. Tổng trainable655360 weights. Dùng chính normalized s_t cho cả semantic loss và correction, tránh một free learned semantic-head/nullspace hoặc norm channel tách khỏi readout sinh. Không thêm gate, gamma, top-K, orthogonality, semantic prototype bank hoặc decoder unfreeze. Zero Wo bảo toàn logits chỉ ở initialization; không bảo đảm phân phối sau train giữ nguyên.

In inference, tính a_t,s_t từ prefix sinh và patch tokens, rồi dự đoán token từ actual corrected logits. Không parser, reference, phrase target, GT box hay text teacher ở inference. Forward giữa train và inference giống nhau; teacher forcing chỉ là cách cung cấp reference prefix để tính auxiliary loss.

## 5. Explicit joint alignment objective

M_Bj trong[0,1] là diện tích giao box B với cell j chia diện tích cell, theo đúng square224 transform của A. Không dùng native-image box trực tiếp lên grid14x14. Với box có support không rỗng, m_B(a_t)=sum_j a_tj M_Bj. Giới hạn grid làm target coarse; nó không là segmentation/pixel-local truth. Box nhỏ có maximum attainable mass thấp; phải báo coverage/size, không hứa localization chính xác.

e_phi=normalized frozen text embedding512. Với tau=.1:

L_align(I,p) = -log mean_(phi,B in S(I,p)) [m_B(a_t) exp(-(1-cos(s_t,e_phi))/tau)].

Tính ổn định bằng logsumexp; mean trên các valid image/prefix groups, không tính group rỗng như negative. Semantic similarity và support mass phải thuộc cùng một cặp (phi,B), không được chọn phrase của entity1 và box của entity2 độc lập. Đây là positive set likelihood/surrogate, không nhận là calibrated probability hoặc thuật toán MIL mới.

Khác L_box đơn thuần: value s_t thật sự dùng sinh caption phải hướng về meaning của whole phrase. Khác semantic cosine đơn thuần: cặp được thưởng phải có attention support tại region được ghép. Không tuyên bố hai điều đó loại hết language shortcut hoặc global-context leakage từ CLIP patches.

Main objective:

L_main = -mean_i stopgrad(R_A(y_s)-R_A(y_g)) sum_(t<=firstEOS) log p_theta(y_s,t|prefix,I,P) +.1 L_KL.

R_A là chính pinned CIDEr-D+.5 object_count_v2 recipe đã có; không thêm reward ngữ nghĩa mới. y_s multinomial temperature1, y_g greedy của actual corrected policy. L_KL=mean valid-token KL(p_A||p_theta) trên cùng sampled prefixes; A/prefix decisions detach. Loss/log-probs đến first EOS, kế thừa BOS/PAD/decode policy hiện có.

L_T=L_main+.05 L_align. Không CE restricted-vocabulary/foil hoặc CE warm-up trong pilot. Alignment tính trên reference prefixes, caption objective trên generated prefixes. Hai distribution khác nhau là limitation thật; generation endpoint sẽ kiểm nó. Không inference-time GT query, không reward interval, không arbitrary random-image negative.

## 6. Matched pilot và quyết định

Control C và treatment T có cùng module655360 params, source Z, initialization, optimizer, data/order, seed và update count. C chỉ L_main; T thêm .05 L_align. A là baseline gốc. Đây là minimum control để đo đóng góp của explicit alignment. Không dùng P/Q khác capacity làm đối chứng thay thế. Q-source để sau nếu có gain và cần phân tích.

Starting protocol: 2000 ảnh hash-first từ4495images của verified labelbankv1.1 thuộc TRAIN10k, disjoint caption dev/val/test; khóa IDs trước modifier expansion, không chọn theo attributes hay model outputs. Tất cả ảnh vẫn học L_main. Hai epochs, globalbatch8=500updates/arm; AdamWlr1e-4,wd.01,clip1,seed42,end-only. Mỗi image/epoch dùng một reference theo rotation được pin cho forward phụ; target set S vẫn được dựng từ cả năm refs cùng prefix. C/T dùng cùng rotation. Missing eligible target chỉ tắt L_align. Chỉ dùng spans trong cửa sổ tokenizer/max-length của A đã khóa; lệch offset/truncation thì skip, không đoán. Counts eligible cần báo khi preparation; không mở rộng ontology hoặc chọn ảnh theo outputs để cứu density.

Text features, base reference hidden states tại target positions và factual patch/memory có thể cache vì A frozen; caching là user-run Kaggle, không localML. Actual free-generation prefixes không được thay bằng cached reference states. Không giả định cache có sẵn hoặc runtime3–5phút. Cap đề xuất: common preparation30phút wall; mỗi arm train+end45phút T4x2; đo throughput sớm và dừng nếu vượt cap, không âm thầm đổi recipe. Chưa launch.

Primary dev1000 exploratory gate: T-A>=+.01CIDEr; paired95CI lower(T-A)>0 và lower(T-C)>0; CHAIRs/i không xấu; recall>=A-.005; length diagnostic. Dùng fixed evaluator/full IDF như A. Dev đã reuse không là fresh confirmation. Nếu caption không gain thì không dùng alignment loss/heatmap đẹp để gọi thành thành công.

Giữ một hash-first eligible alignment-dev cohort, disjoint training, từ dev hiện có; report joint pair alignment loss, support mass và semantic cosine riêng. Không chọn cohort theo output hoặc giả held-out source sạch tuyệt đối. Nếu chỉ localization tăng mà phrase cosine không tăng thì chưa chứng minh semantic part. Nếu caption gain mà alignment không tăng, chưa xác nhận cơ chế semantic alignment. Ngoài ra báo từ vựng, attributes và generic-head-only target share; semantic teacher có thể chủ yếu học noun class.

Sau positive pilot mới làm independent case-level check về phrase/attribute correctness trên captions sinh. Không dùng atlas20 hoặc cùng training scorer làm độc lập xác nhận factuality. Không gọi permutation/OOD sensitivity là semantic causality.

## 7. So với lịch sử dự án

- R: decoder/reference-to-Q-to-patch box loss; ở đây alignment đặt trực tiếp trên semantic value readout trước Q và same value sửa logits. Box mass một mình không đạt claim.
- G/S: noun-presence classifier/logit head; ở đây target là phrase embedding+grounded region, không BCE presence/absence hoặc class list verifier. Nếu data chỉ còn bare nouns và gain chỉ ở counts thì đóng góp thực chất vẫn rất gần noun head, cần nói rõ.
- VG G: region phrase CE; ở đây không dạy decoder tái tạo caption vùng độc lập. Reference-prefix readout được căn chỉnh với semantic pair rồi dùng trong whole-caption policy. Không dùng nguồn VG noisy ở pilot đầu.
- CE/CROSS21: template/preference learning; ở đây không template bank, không false counterfactual captions, và actual autoregressive corrected policy là endpoint.
- CSA-like: không noun/verb prototype banks hoặc cascade gating. Frozen text teacher và multi-positive phrase targets vẫn kế thừa rộng semantic alignment, không tự mới.

## 8. Prior và đóng góp ứng viên

Zhou etal CVPR2020 đã có attention distillation/POS-SCAN và SCST; phần method3 đã được đọc từ arXiv2004.00390. Không nhận supervised alignment+caption objective là mới. https://arxiv.org/pdf/2004.00390

Mavroudi/Vidal CVPR2022 chỉ rõ limitation word-agnostic attention và dùng latent word-region alignments. Lượt này xác minh abstract chính thức/supplement, chưa đọc toàn main PDF vì fetch403. Candidate này không nhận prefix-only a_t là grounding chính xác của nextword; chỉ prospective positive-set evidence alignment. https://openaccess.thecvf.com/content/CVPR2022/html/Mavroudi_Weakly-Supervised_Generation_and_Grounding_of_Visual_Descriptions_With_Conditional_Generative_CVPR_2022_paper.html

Show,ControlandTell CVPR2019 là tiền lệ phrase/chunk-region captioning. Candidate không dùng region-control input tại inference. https://openaccess.thecvf.com/content_CVPR_2019/html/Cornia_Show_Control_and_Tell_A_Framework_for_Generating_Controllable_and_CVPR_2019_paper.html

Phần ứng viên cần đánh giá: joint set supervision của semantic direction và support region ngay trên evidence value được dùng để sửa logits của frozen captioner; xử lý nhiều continuation cùng prefix mà không cung cấp future phrase cho forward. Đây là thiết kế kết hợp có thể nghiên cứu, CHƯA xác minh originality của component/loss. Novelty search chưa hoàn tất. Có gain và grounding improvement cũng chưa tự chứng minh đủ novelty để publish.

Quyết định: chọn đây là method candidate có explicit learned semantic alignment để root phản biện/hoàn thiện; không launch hoặc hứa gain/novelty. Nếu matched pilot fail, dừng cấu hình này, không kết luận mọi alignment bất khả thi.
