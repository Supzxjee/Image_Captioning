# Astra — chốt ứng viên phương pháp sau C/T

26/10/2026. Thảo luận thiết kế, chưa viết code/chạy model. Dựa trên các kết quả đã audit, báo cáo V-COCO và source liên quan; không đọc lại ZIP. Phần CEV-PG dựa trên bản công thức/bằng chứng cộng sự triển khai đã chuyển, không tuyên bố đọc lại file Argon đầy đủ.

## 1. Hai ứng viên và lựa chọn

| Ứng viên | Thay đổi thực chất | Ưu điểm | Rủi ro quyết định |
|---|---|---|---|
| A. Lọc semantic prompt kiểu CEV-PG | Học độ tin cậy prompt từ text và visual read, điều biến memory | Dùng pipeline hiện có, có thể giảm ảnh hưởng cue sai | Chưa có negative labels đủ chắc; gần verifier/gate đã có; decoder có thể đã bỏ cue sai; signed residual không bảo đảm giảm noun logits |
| B. Học caption bằng cải thiện mệnh đề có căn cứ actor–role, giữ unknown dưới dạng khoảng | Đổi semantic representation của tín hiệu học và policy objective; inference giữ giao diện A | Dùng nhãn vai thật; cập nhật trực tiếp caption sinh tự do; không cần teacher attention hay ảnh che | Parser/binding có thể sai, nhãn thưa có thể làm mọi khoảng đều bất định; tính mới chưa xác minh; reward có thể bị khai thác |

**Tôi chọn B làm ứng viên phương pháp để chuẩn bị một pilot có giới hạn. Không chọn A theo nhãn/loss hiện tại của Argon.** Đây là lựa chọn thiết kế có điều kiện dữ liệu, chưa là Go train hoặc lời hứa caption gain.

Tôi thay ưu tiên C0 vì yêu cầu hiện tại là chọn một phương pháp thực nghiệm. C0 CE+KL vẫn có giá trị chẩn đoán, nhưng không cần chạy trước một regime SCST khác có matched continuation control riêng. C0 không được dùng làm method paper hoặc bằng chứng phủ định mọi adaptation.

## 2. Nút thắt và dự đoán có thể bị bác bỏ

Các phép thử đã cho thấy định vị proxy và teacher-forced score không bảo đảm utility; recipe CE+auxiliary sau checkpoint SCST đã regress. Chưa có bằng chứng quy regression cho riêng CE, stale loss hoặc frozen decoder.

Giả thuyết B: **một tín hiệu học dựa trên phần cải thiện ngữ nghĩa mà annotation thực sự xác nhận, giữ chung identity của các vai trong câu, có thể cải thiện policy caption trên nền A hơn cùng SCST không có tín hiệu ấy.**

Dự đoán: T phải hơn cả A và matched SCST-control về caption tự do; đồng thời tăng các mệnh đề quan hệ được xác nhận trên tập role-development đã khóa mà không giảm coverage/parser acceptance hoặc mất nội dung. Nếu chỉ reward tăng trên train, hoặc CIDEr tăng nhưng relation outcome không tăng, chưa xác nhận semantic–visual alignment.

Không giả định actor/object phải được caption gọi bằng ID hay mỗi từ cần một bbox. Caption `a person riding a horse` thường là mệnh đề tồn tại: cần một binding hợp lệ trong ảnh. Với nhiều mệnh đề cùng một người, binding phải dùng **cùng một actor**, không chọn một người thuận lợi cho từng verb. Đây là alignment ở mức mệnh đề/cấu trúc vai; không claim selected-instance grounding khi text không định danh instance.

## 3. Representation và tensor flow

### Đường inference

Ảnh → CLIP197×768 → projection512 → Q32×512 và prompt≤20×512 → memory≤52×512 → decoder3layers → vocabulary logits.

Giữ cấu trúc A. Chỉ mở **toàn lớp decoder cuối**, khoảng4,2M tham số theo kiến trúc512/FFN2048 hiện tại; encoder, hai lớp decoder đầu, embeddings và vocabulary head đóng băng. Đây là một lựa chọn capacity mới cần kiểm chứng, không suy từ kết quả trước rằng nó chắc tốt hơn29M. Cả control/treatment có cùng trainable set.

Inference ảnh mới dùng đúng ảnh và prompt của pipeline A, không có GT role, box, graph, parser hoặc reranker. Không đổi caption selection để gọi thành cải thiện decoder.

### Module chỉ dùng khi train

1. **Caption parser cố định:** caption → entity nodes có mention/coreference IDs + directed role predicates. Không chỉ tìm từ hold và object ở đâu đó trong câu. Giữ negation, quantifier và identity constraints; construction không xử lý chắc chắn trả unknown. Pilot chỉ chủ động chấm hold/obj và ride/instr theo nguồn đã audit; không tự gộp carry vào hold hoặc `on` vào ride.
2. **Visual annotation graph:** nodes là COCO instance IDs/classes; edges là role links V-COCO; actor/action labels có trạng thái1/0/−1. Không lấy raw UPT score làm truth.
3. **Grounding matcher:** đối chiếu mỗi connected proposition của caption với các assignment tương thích class/identity trong graph. Hai verbs chung một text subject phải dùng cùng GT actor; hai entity được text xác định khác nhau không được tùy tiện nhập làm một.
4. **Interval evaluator:** mệnh đề được chứng minh bởi nhãn, bị phản bác bởi nhãn rõ, hoặc còn bất định. Nó trả khoảng utility thay vì ép mọi thiếu link thành0/false.
5. **Policy loss:** kiểm tra trực tiếp khoảng utility của sampled caption so với greedy caption hiện tại; chỉ cập nhật role term khi khoảng này xác định được dấu. Đây là pairwise policy-gradient surrogate, không phải SCST của một fixed scalar role reward.

Ví dụ: GT có actor1 hold bat và actor2 ride horse. Câu `a person holding a bat and riding a horse` không được nhận hai điểm nhờ đổi actor giữa các cạnh. Nếu GT explicit action labels phủ đủ các actor tương thích và phản bác mọi assignment chung, mệnh đề bị refute; nếu còn một assignment chưa rõ thì unknown. Câu nói rõ hai người khác nhau có thể đúng. Không dùng bbox center/overlap làm nhãn predicate.

## 4. Supervision và khoảng utility

Nguồn là V-COCO role/actor labels chính thức đã được ghép COCO instance IDs, không phải caption absence. Kho `grounded_links_v1` hiện có2.026FITimages,1.330positive links trên1.014images cho hai actions; phải giữ cả ảnh không có positive link. Role thiếu ID đích vẫn unknown. [Nguồn tác giả V-COCO](https://github.com/s-gupta/v-coco).

Đã biết giao FIT với mainTRAIN10k chỉ175images/85positiveimages/116links. Vì vậy không tuyên bố main10k có1.000positiveimages. Pilot dùng **toàn FIT hợp lệ làm một phần training pool**, không nhập old calibration/dev vào FIT chỉ vì chúng là Karpathy train.

Export grounded_links chỉ chứa annotated actors/linked endpoints, không tự là danh sách mọi người trong ảnh. Muốn chứng minh phủ actor cho một refutation phải join nguồn COCO instances đầy đủ đã dùng để tạo export; nếu chưa có phép join đó thì giữ unknown, không refute. Phương án này dùng cached prompt embeddings gốc, không cần original prompt JSON để dựng P0 hay audit gate Argon.

Cho một projected proposition π, đặt weight wπ theo số role edges của nó (cap3). Chỉ đánh giá những thuộc tính/role thật sự thuộc ontology; không nhận việc GT class person xác nhận gender, màu áo hoặc toàn bộ câu.

- Supported: tồn tại assignment nhất quán, mọi modeled edge được known-positive role link xác nhận → [lπ,uπ]=[wπ,wπ].
- Refuted: mọi assignment còn hợp lệ đều có mâu thuẫn explicit đã chú giải → [−wπ,−wπ]. Trong pilot, chỉ dùng actor/action=0 khi binding actor không nhập nhằng và các actor tương thích đã được source bao phủ; nếu còn actor chưa rõ, role target thiếu hoặc ambiguity thì **không** refute. Không dùng “role không ghi” làm negative.
- Unknown: [−wπ,wπ].

Các bound chỉ đúng **có điều kiện theo ontology, parser và annotation đã chấp nhận**, không phải chứng nhận chân lý vật lý. Noise annotation vẫn có thể làm sai; phải audit policy chấm.

Caption có thể đúng dù bỏ một fact. Penalty cho việc mất known-supported fact là lựa chọn utility về bảo toàn nội dung, không phải nhãn “caption còn lại sai”. Predicate ngoài scope được phân biệt với predicate có thể thuộc scope nhưng parser không xử lý: trường hợp sau giữ một opaque/unknown unit, không âm thầm xóa khỏi graph.

### Vì sao không ép unknown thành negative?

So trực tiếp caption sampled y_s với greedy y_g của current policy. Cả generation decisions và bound evaluation được detach. Với canonical propositions thêm A_s và bỏ D_s trong y_s so với y_g:

LΔ(y_s,y_g)=Σ_(π∈A_s) lπ − Σ_(π∈D_s) uπ.

UΔ(y_s,y_g)=Σ_(π∈A_s) uπ − Σ_(π∈D_s) lπ.

Các mệnh đề giống nhau triệt tiêu; duplicate paraphrases đã xác nhận cùng meaning không được cộng lặp. Không merge mọi `person` thành một entity trước khi kiểm vai.

Đặt K=3 cố định và role advantage:

a_G(y_s,y_g)= clip(LΔ/K,−1,1) nếu LΔ>0;

a_G(y_s,y_g)= clip(UΔ/K,−1,1) nếu UΔ<0;

a_G(y_s,y_g)=0 nếu khoảng chứa0.

**Đính chính26/10:** bản trước chấm riêng r_safe(y_s) và r_safe(y_g) rồi trừ nhau. Nó không bảo đảm abstention ở policy update: unknown sample có bonus0 vẫn nhận advantage âm nếu greedy có bonus dương. Công thức hiện tại thay thế bản đó. Claim zero role signal chỉ áp khi **khoảng utility của chính cặp** chứa0; không áp tự động vì một caption riêng lẻ có nhãn unknown.

Không chuẩn hóa bằng số parsed claims vì model có thể thay denominator. Không thưởng việc biến một claim sai thành một verb mà parser không hiểu: added opaque unit có khoảng bất định, nên không tạo positive sign chỉ nhờ biến mất khỏi parser.

Ví dụ theo đơn vị weight1:

- Thêm known-supported fact, phần khác giữ nguyên: Δ=[1,1] → bonus.
- Bỏ known-supported fact: Δ=[−1,−1] → negative utility advantage.
- Thay known-false bằng opaque unknown: Δ=[0,2] → không có bonus.
- Thêm một true fact đồng thời thêm một unknown: Δ=[0,2] → chưa thưởng vì unknown có thể bù mất cải thiện.

Đây là policy conservatism về sign của cải thiện theo nhãn đã có; không gọi toàn bộ câu unknown là false. Nó có thể rất ít signal — đó là rủi ro phải kiểm trước train.

## 5. Loss và gradient chính xác

R_B là **đúng reward CIDEr-D +0,5 object_count_v2 đã có trong recipe SCST của dự án**, giữ source/targets được pin; không nhận reward này là verifier chân lý mới.

Với y_s sampled từ current policy bằng multinomial temperature1, y_g là greedy cùng policy, đặt ℓ_s=Σ_(t≤firstEOS) log pθ(y_s,t|y_s,<t,M_i) và ΔR_B=R_B(y_s)−R_B(y_g):

L_C = L_CE(reference) −0,05 mean_i stopgrad(ΔR_B) ℓ_s.

L_T = L_C −0,01 mean_i stopgrad[a_G(y_s,y_g)] ℓ_s.

Hệ số0,01 giữ nguyên scale0,05×0,2 đã đề xuất. Chỉ base term là SCST thông thường. Role term là **masked pairwise policy-gradient surrogate**; không nhận nó là unbiased gradient của expected fixed scalar role reward. Nếu khoảng cặp chứa0 thì gradient bổ sung từ role term đúng bằng0; CE và base SCST vẫn có thể cập nhật caption đó.

CE lấy mean valid token qua5references/ảnh như pipeline hiện có. Parser, matching, bounds và generation decisions đều detach. Chỉ rescored log-probabilities/CE cập nhật decoder cuối. Không backprop vào image encoder, graph parser hoặc annotation. Claim conservative ordering chỉ có điều kiện theo parser/annotation và soundness của bound; không phải bảo đảm chất lượng policy sau một optimizer step.

Không có loss mask image, attention mass, BCE gate, giả ordering từ ảnh che hoặc reference lựa theo CIDEr. Khác phrase CE/CROSS: training là **full-caption policy**, role labels đến từ source chính thức, binding ở toàn mệnh đề, không phải preference giữa hai template trên21groups. Không thêm six evidence tokens/KVLoRA đã có trong phrase_model.py.

## 6. Pilot tối thiểu: A + hai nhánh

**C_SCST:** L_C trên.

**T_role:** giống C_SCST mọi mặt, thêm pairwise role term −0,01 stopgrad(a_G)ℓ_s.

A là checkpoint gốc, cùng decode/evaluator. Hai nhánh cùng A initialization, decoder cuối trainable, optimizer/seed/data order và training budget. Đây là control continuation của **regime SCST đang dùng cho method**, nên C0 CE+KL không cần chạy trước hoặc thành arm thứ ba. Nó không được thay bằng A đơn lẻ.

Training pool: hash-first tối đa2.000FITimages sau khi loại currentdev1000/val/test và giữ old V-COCO calibration/dev ngoài training. Thêm cùng số replay images từ factual I1/P1 của stale-cache2k, disjoint FIT và các role-eval IDs. Chọn n cân bằng, làm tròn bội4; không chọn theo model score. Mọi ảnh đều có human references. Replay không có graph hợp lệ nhận a_G=0, không gán negative.

Budget: hai epochs trên2nimages, globalbatch8 → n/2updates, tối đa1.000updates; LR5e-6 AdamW(wd.01,betas.9/.999,eps1e-8),clip1,eval-mode có gradient. Một seed, end-only, không sweep hoặc thêm arms trước gain. Đây là starting protocol có thể bị bác bỏ, không phải recipe đã tối ưu.

### Chuẩn bị nhỏ nhất trước khi duyệt train

Một packet duy nhất, không phải chuỗi ablation:

1. Inventory FIT eligibility/source/cache/GT statuses và disjointness; ngưỡng thực dụng n≥500images. Missing source/coverage phải báo, không mở calibration/dev hoặc trộn test labels để đủ số.
2. Trên128FITimages hash-first đã khóa, dùng saved A captions nếu đúng IDs/checkpoint/decode; nếu chưa có thì người dùng chạy một frozenA greedy + một multinomial sample/ảnh trên Kaggle. Không localML. Parse/bind/evaluate bounds và ghi rõ nguồn derivation.
3. Audit30quyết định chấm theo sampling rule đã khóa, kiểm role attachment/unknown handling với annotations; ghi rõ AI-assisted nếu có. Yêu cầu≥27/30đúng derivation và không có unknown→false do thiếu link. Không dùng đây làm nhãn winner–loser cho model.
4. Cần ít nhất16/128cases có a_G(sample,greedy) khác0 và ít nhất8cases mà interval/identity handling khác cách bag/unknown=0. Đây là gate density/operative-component cho ngân sách, không power estimate. Nếu chỉ thành ordinary relation reward hoặc gần như mọi interval chứa0, dừng ứng viên này; không tự tăng số samples/ontology để cứu. Unit cases phải có unknown sample so với known-positive greedy và chiều ngược lại: chấm riêng rồi trừ có thể khác0, nhưng nếu direct pair interval chứa0 thì a_G phải bằng0.

Các số trên là tiêu chí đề xuất trước khi thấy packet, cần khóa trước thực thi. Hiện chưa có đủ dữ liệu xác nhận chúng sẽ đạt. Việc đầu tiên tôi chọn là **packet feasibility của chính phương pháp B**, không C0 và không viết notebook train ngay.

## 7. Shortcut, alias escape và phép thử cơ chế

- Ngôn ngữ/object-class priors có thể đoán đúng nhiều actions. Cần một role-development cohort cố định, có ảnh cùng object classes nhưng khác annotated actions; output raw captions và known-supported/contradicted/unknown counts, không chỉ lexical cooccurrence.
- Parser không chắc coreference/negation/passive/number thì unknown. Không làm số parsed claims biến thành training reward, không bonus paraphrase ngoài parser. Vẫn không bảo đảm chặn mọi parser gaming; coverage và human reading phải được báo.
- Novelty về binding chỉ tồn tại ở các ca cần chung actor/object identity. Nếu những ca đó không xuất hiện/không thay đổi scorer, không claim instance binding đã được kiểm chứng.
- Positive role không tự xác nhận mọi adjective hoặc selected actor mà caption không định danh. Đối chiếu existential proposition được phép; không quảng cáo bbox grounding cho từng từ.
- T có thể chỉ tăng những phrases đúng annotation mà ít liên quan với nội dung caption. Primary CIDEr/CHAIR/recall và bonus cap ngăn một phần nhưng không bảo đảm. Sau gain mới làm blinded30case relation review trên outputs nguyên bản; không xem cùng training scorer là độc lập chứng minh correctness.

Primary gate trên currentdev1000: T−A≥+.01CIDEr với paired95CI lower>0; T−C_SCST lower>0; CHAIRs/i không xấu; recall≥A−.005. Length diagnostic. Secondary role-dev đã xem: supported propositions tăng trước control, contradiction/unknown/coverage không bị che hoặc denominator đổi. Nếu chỉ primary thắng mà role outcome không có tín hiệu, chưa claim alignment.

Không gọi currentdev hoặc oldrole-dev là fresh unseen. Không dùng annotation/test role input vào inference. Early stop cho hash/binding lỗi, nonfinite, frozen-state đổi, OOM hoặc vượt budget; không interim dev để chọn checkpoint.

## 8. Chi phí và cache

Encoder A đóng băng cho phép cache **memory≤52×512** thay vì stage lại7,1GB raw main tensors. Với≤4.000train+1.000dev, FP32 memory tối đa khoảng0,53GB, thêm masks/IDs. Inference vẫn chạy encoder A bình thường; đây chỉ là cache cho training.

Replay có factual cache đã lưu. FIT mới cần kiểm original visual/text coverage; nếu thiếu visual cache mà có pixels/prompt source đúng thì encode factual một lần trên Kaggle, không tải crop/ảnh che mới. Nếu không có đủ đúng source thì không dựng input giả. Memory-cache phải được giữ và kiểm digest/parity trước hai arm.

Không suy runtime SCST từ3phút CE của C/T: có sample+greedy rollouts và CPU parsing/matching. Giới hạn đề xuất: common preparation/cache≤30phút wall-clock T4×2; mỗi arm train+end-eval≤60phút. Tương đương tối đa5T4-device-hours tính riêng hai phần này; setup/I/O khác phải ghi riêng và quota còn lại cần xác nhận trước run. Đây là cap, không ETA. Dùng throughput50updates đầu để dừng nếu projection vượt cap; không âm thầm đổi batch/epochs. Không dùng judge VLM/LLM lớn.

## 9. Prior gần và ranh giới tính mới

Đã đọc phần method liên quan của hai nguồn chính thức:

- **SC-Captioner, ICCV2025:** reward dựa trên thay đổi object/attribute/relation giữa caption ban đầu và self-corrected, đối chiếu reference; có thưởng/phạt. Do đó graph decomposition, reward cho semantic edits và RL không mới. [Method3.2–3.3](https://arxiv.org/html/2508.06125v1).
- **FoCUS, arXiv2026:** scene-graph component scores, prompt-conditioned signed rewards và GRPO; dùng verification judge cho relation/attribute. Không thể nhận việc đem scene-graph annotations vào reward là mới. [Method3.1–3.3](https://arxiv.org/html/2609.00709v1).

Phần có khả năng là đóng góp: **giữ unknown dưới dạng khoảng cải thiện và chỉ cho role signal khi sign được xác định theo annotation, với entity-consistent grounding thay vì ghép class-level tuples độc lập**. Đây là hypothesis novelty, chưa xác minh đủ prior về conservative RL/partial labels. Không claim thuật toán bound/SCST nói chung mới.

SPICE là tiền lệ semantic propositions; đã đọc method. CCCaption(CVPR2026) cũng là prior rất liên quan về completeness/correctness rewards; lượt này mới đọc primaryabstract, chưa full method nên không nhận đã loại trừ overlap. [CCCaption](https://arxiv.org/abs/2602.21655).

## 10. Phản biện CEV-PG của Argon, trong một phạm vi

Tôi không chọn bản hiện tại vì label construction chưa hợp lệ cho claim verifier: không có class trong instances+references vẫn chưa là verified absence; cross-batch absence cũng vậy. Cần một policy negatives được kiểm chứng, không chỉ BCE/ranking có vẻ hợp lý.

Ngoài ra signed residual không có monotonicity tới word logits: alpha có thể âm, Wmod đổi dấu, decoder Jacobian tự do. Alpha0 chỉ identity lúc initialization nếu đường base không đổi; sau khi unfreeze, alpha0 không trả về A. Gate inversion/permutation là sensitivity/OOD test, không tự là correctness proof.

Wp/Wh/Wmod đã có393.216weights trước MLP/LN, nên claim .20M sai;9Mbase cũng phải count thực. Contextual text embeddings có thể mang shortcut; so khác ảnh không loại mọi text/position priors. Four-arm proposal cộng reference-selection mới sẽ mở nhiều confounds trước khi có label policy.

Phần đáng giữ là metadata association audit prompt-spurious classes và A hallucinations, nhưng nó không đủ chứng minh causal channel hoặc filter sẽ giúp; tôi không biến audit ấy thành một method thay cho lựa chọn B. C0 cũng chỉ kiểm regime continuation mới, không phủ định mọi CE/SCST adaptation.

**Chốt:** chọn ứng viên B, chuẩn bị đúng một packet feasibility gắn với method. Nếu packet fail thì nói ứng viên hiện không khả thi trong ngân sách, không dùng tên loss mới để bảo đảm paper. Chưa viết code hoặc launch GPU.
