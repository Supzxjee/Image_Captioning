# Đề xuất: học phụ thuộc vùng–từ bằng hoán đổi bằng chứng có nhãn bộ phận

01/11/2026. Codex2, base63c9b12. Brainstorm độc lập theo yêu cầu mới của user; chưa chọn triển khai, chưa code/train/GPU. Áp dụng ARS3.24.0 theo phạm vi research-question/method design, source verification và tự phản biện; làm inline, không giả nhận có hội đồng/subagent độc lập. Không sửa báo cáo tuần của Codex1 hoặc lịch sử X018.

## Quyết định và câu hỏi nghiên cứu

**Ưu tiên một hướng: giám sát tác động của bằng chứng vùng qua interchange interventions tại nhánh đang tham gia sinh token.** Kế thừa Interchange Intervention Training (IIT), không nhận phát minh phép hoán đổi hoặc causal alignment. Cải tiến dự kiến là một cách xây dựng và sử dụng **counterfactual targets chỉ được xác định một phần** cho caption ảnh thật: chỉ giám sát các noun/attribute facts suy ra hợp lệ từ vùng donor và các vùng còn nguyên; để UNKNOWN cho phần caption không suy ra được.

RQ: Với cùng captioner, ROI evidence và positive labels, ràng buộc phản ứng của token dưới hoán đổi một vùng có giúp học correspondence thực sự được dùng để sinh caption, hơn grounding supervision và factual token CE thông thường không?

Ba hướng đã cân nhắc: (i) tiếp tục giữ router distributions giống nhau trong X018; (ii) region-constrained inference/correction; (iii) **học một quan hệ phụ thuộc vùng–từ bằng internal interventions có nhãn**. Chọn(iii) vì có thể bác bỏ ngay việc attention/auxiliary đúng nhưng output bỏ qua evidence, và có đường dùng positive object labels thay vì suy absence. Đây là lựa chọn nghiên cứu, không chứng minh utility/novelty.

## Vì sao sửa khuyến nghị X018

Với a,a′ là mass router trên các anchors, JS(a,a′)=0 khi cảhai cùng sai. Nếu independent grounding supervision đã đạt a=a′=onehot đúng thì JS cũng tự bằng0. Ví dụ đối tượng đúng là index0:

| a | a′ | JS, nats | Grounding NLL hai ảnh |
|---|---|---:|---:|
| (1,0,0) | (1,0,0) | 0 | 0 |
| (.1,.8,.1) | (.1,.8,.1) | 0 | 4.605170 |
| (.9,.05,.05) | (.6,.2,.2) | .063288 | .616186 |

Hai dòng đầu cho thấy consistency không tự thêm semantic truth; dòng cuối cho thấy nó phạt cả confidence khác nhau khi vẫn chọn đúng. JS vẫn có thể regularize hữu ích khi dữ liệu hữu hạn, nhưng chưa có lý do lấy nó làm contribution chính. Số ở đây là đại số được tính bằng Python chuẩn, không kết quả mô hình.

Một phản ví dụ khác: π chọn đúng box nhưng output projection của branch W_o=0. Grounding tốt trong khi p(token) hoàn toàn bằng C0. Điều này chỉ ra một khả năng cấu trúc; không quy D/E trước đây là branch hoàn toàn unused hoặc chứng minh đó là nguyên nhân thất bại của chúng. Response15 đã cho thấy ảnh có tác động, chưa cho biết vùng nào tạo tác động.

## Forward: correspondence đi vào đâu?

Giữ C0: CLIP→Q-Former→decoder cùng original prompts. C0 cung cấp hidden h_t∈R512 trước fc_out. Toàn C0, detector và CLIP frozen. Thêm một **regional evidence adapter** trước chính fc_out; không thêm GRU/ngôn ngữ mới hoặc K vocabulary heads như X018.

Predicted proposal bank K≤6, cùng detector/NMS policy cho mọi arm. Box/confidence được dùng, predicted class name không là input. **Local value lấy từ crop RGB encode độc lập bằng cùng pinned CLIP**, z_i∈R768; không coi pooled contextual patches của toàn ảnh là đã isolated object evidence. Crop vẫn có thể chứa background/đối tượng chồng lấn; không có causal certificate. Phải so source dimensions/crop preprocessing thật, không giả cache cũ chứa các features này.

~~~text
q_t = W_q h_t
k_i = W_k z_i + W_b geometry_i
π_t = softmax( additive_score(q_t,k_0), ..., additive_score(q_t,k_K) )
v_i = W_v z_i ; v_0 = 0
r_t = Σ_i π_ti v_i
logits_t = W_B (h_t + W_o r_t) + b_B
p_θ(y_t | I,prompt,y_<t) = softmax(logits_t)
~~~

W_B,b_B là fc_out của C0 giữ frozen; q,k,v dùng dimension128, W_o:128→512. Trainable chỉ router/projections W_q,W_k,W_b,W_v,W_o và scorer. Đây là region attention/residual quen thuộc, không phần mới. W_o=0 lúc init cho exact C0 forward nếu mọi phần còn lại thực sự frozen; gradient đầu vào W_o vẫn có thể khác0, sau đó mới truyền qua các projection phía trước. Không đặt một hard gate bằng0 khiến toàn nhánh hết gradient. Khi bank rỗng, residual=0.

**Cái gì căn chỉnh:** content token/fact↔predicted region được match với positive box, và **ảnh hưởng của value tại region ấy↔thay đổi token phải xảy ra**. h_t có global image/prompt information; không đổi tên nó thành text-only hoặc claim đã loại bypass. Chính vì bypass còn tồn tại, training intervention giữ h_t cố định để buộc đường local có tác dụng thích hợp. Có residual không đảm bảo tác dụng đúng hoặc giảm hallucination.

## Intervention và nhãn bộ phận

Cho base b và donor s, có một predicted region được match hợp lệ ở mỗi bên. Ở lượt intervention, giữ nguyên h_t^b, keys/router π_t^b và tất cả values khác; chỉ thay v_k^b←v_l^s. Các tensor đó được giữ cùng giá trị giữa factual/patched pass; không cần detach gradients của router hoặc shared projections. Đây là can thiệp ở mô hình, **không phải ảnh phản thực mới hay caption gold cho ảnh gốc**.

Với mỗi fact context c_j, lập target record:

| Trạng thái | Khi nào được gán | Target sau intervention |
|---|---|---|
| Updated | Local variable được thay và value/label donor đã biết | Donor label đúng tại content token/window tương thích |
| Protected | Một fact khác gắn vùng không đổi và độc lập với phép thay trong partial semantic model đã định nghĩa | Base label vẫn đúng |
| UNKNOWN | Quan hệ/count, attributе chưa biết của donor, context đã nêu identity cũ, vùng overlap hoặc dependency không rõ | Không tạo loss/nhãn đúng-sai |

**Hai loại data để tiếp cận đúng mục tiêu:**

- Object identity: lấy donor class từ một positive COCO-matched region thật. Ví dụ thay value của region bus bằng value của region truck, target local noun có thể là truck; không suy bus vắng khỏi toàn ảnh hoặc thay tất cả chữ bus trong caption. Context cho intervention phải nằm trước noun và tương thích với cảhai labels; contexts như “the bus is...” không hợp lệ để đổi class bus→truck. Pilot dùng single-token aliases/grammar-compatible contexts và unique GT-class bindings; ambiguous instances giữ UNKNOWN.
- Attributes: cặp cùng entity có old/new color đã kiểm như MED15 giúp đặt target thay màu, giữ class noun và fact của vùng khác khi hợp lệ.15cũ vẫn là development diagnostic; không biến thành held test. Không có attribute label thì không tự suy từ class box hoặc reference omission.

Ví dụ base có bus đỏ, car trắng; donor là bus xanh cùng entity: dưới intervention chỉ value bus đổi, local bus-color target là blue, fact car-white được bảo vệ. **Toàn câu mới, salience, quan hệ không gian và những chi tiết chưa biết không có target**. Không giả định được đầy đủ causal graph của ảnh tự nhiên. Cho object donor khác ảnh, partial semantic model mô tả virtual evidence bank; không khẳng định ảnh base thật nay chứa donor object. Đánh giá caption thực vẫn dùng ảnh thực và gold thích hợp.

Positive sets có thể nhiều proposals. Factual grounding dùng tổng mass trên mọi proposal phù hợp. **Pilot interchange ban đầu chỉ dùng bindings có một correspondence proposal rõ sau NMS/matching**, khóa eligibility trước scores. Chưa giải quyết transport nhiều-to-nhiều/group entities; trường hợp đó skip intervention loss, không bỏ full-caption evaluation. Gold class/box chỉ để tạo loss/intervention target khi train; không prune memory theo target hoặc đưa gold box vào inference.

## Loss và lý do nó khác auxiliary alignment

~~~text
L_common = L_real_caption(p_θ)
         + λ_fact L_verified_fact_CE(p_θ)
         + λ_g [−log Σ_{i∈G_j} π_ti]

L_partial_IIT = mean_over_valid(b,s,k,j)
               CE( p_θ^{b←s,k}(· | c_j), target_{b,s,k,j} )
~~~

CE lấy tại đúng content token/window, qua fc_out đang dùng khi decode. Multiword candidates cần teacher forcing riêng và target mask rõ; không lấy normalized tail odds làm calibrated probability. Không có full-caption CE trên một caption tự ghép hoặc nhãn âm từ missing annotation. Protected dùng correct-token CE, **không ép equality của probability** để khỏi phạt cải thiện đúng. UNKNOWN không bằng zero-valued negative.

Trong intervention pass, global C0 branch không đổi nên nó không thể tự giải thích một target token phải đổi. Với một changed entry k và single-token contrast a/b:

~~~text
Δ[log p(a) − log p(b)]
  = (W_B[a] − W_B[b]) W_o π_tk (v_l^s − v_k^b)
~~~

Đẳng thức từ linear readout khi h_t/π cố định, không theorem mới và không áp trực tiếp cho mean nhiều token. Nó làm rõ phần cần học: đường local value phải điều khiển đúng token contrast. Nếu W_o=0 hoặc value không được dùng, patched prediction giữ nguyên; không đạt updated targets xung đột đã gán hợp lệ. L_ground một mình không kiểm điều này. Ngược lại loss có thể bị giảm qua overfit labels, tăng residual quá mạnh hoặc artifacts donor; source-held và natural-caption outcomes vẫn bắt buộc.

**Một delta chính:** huấn luyện actual caption readout bằng các intervention có **partial semantic targets** thay vì chỉ matching attention/embeddings hoặc input–output correlation. Xây dựng target/mask hợp lệ cho ảnh thật và buộc một channel vùng→token có tác dụng là phần muốn thử; generic IIT, partial-label masking và cross-attention đều kế thừa. Nếu full-supervision/architecture đã khiến intervention đúng thì term này có thể dư thừa; không claim nó luôn cung cấp thông tin mới.

## Prior sát và bằng chứng thực đọc

| Nguồn | Phần xác minh | Ranh giới với candidate |
|---|---|---|
| **Geiger et al. (2022), Inducing Causal Structure for Interpretable Neural Networks, ICML/PMLR162** | Tự lấy publisher PDF, đọc §3 Eq1–4, §4 Eq6–8, §5 Eq9–10 và xem render p3. IIT huấn luyện neural model bằng gold outputs sau aligned internal swaps; typed interventions cũng đã có. ReaSCAN đã là vision-language grounding. | Không nhận internal swaps, causal alignment, type-specific swaps hoặc áp dụng chung vào vision-language là mới. Candidate cần làm rõ partial targets cho open-ended caption và hiệu quả của sửa đổi đó; đổi task sang caption riêng không đủ novelty. |
| **Neural Baby Talk (Lu et al., 2018), CVPR** | Đã tự đọc §3.1–3.4/Eq3–14 ở X009. Pointer, sentinel, multi-positive supervision, vùng→lexical output có prior. | Factual region supervision/readout là baseline kế thừa, không contribution; cần khớp labels/budget khi so. |
| **How Many Counterfactuals Does It Take? (Gupta et al., 2026)** | Lượt này đọc author README và code CounterFactualGenerator, CRMLoss, ExogenousVariableModule, trainer.train_step tại pinned commit; **chưa full paper**. Code CRM được đọc là factual BCE reweighted bởi clipped exp(logp_factual−logp_cf), weights detach. | Activation patching/log-prob influence không mới. Chưa thấy partial per-region donor-label objective trong các hàm đã đọc, nhưng không kết luận vắng trong paper/toàn repo. Không chạy code hoặc xác nhận reported results. |

Nguồn kiểm lại:

- IIT: [publisher PDF](https://raw.githubusercontent.com/mlresearch/v162/gh-pages/geiger22a/geiger22a.pdf), [paper landing](https://proceedings.mlr.press/v162/geiger22a.html). SHA256 f97e0d0d4b3285af9a2727b26442131f906cc6ff358755b39906d652f4ef665d. ARS pdf_read_preflight: PASS15/15/15, không repair warnings; structural PASS không thay đọc hiểu. Không mượn zero-loss causal-abstraction theorem cho dataset hữu hạn/partial labels của ta.
- NBT: [primary](https://openaccess.thecvf.com/content_cvpr_2018/html/Lu_Neural_Baby_Talk_CVPR_2018_paper.html); phạm vi PDF đọc đã lưu X009.
- Gupta et al.: [paper lead](https://arxiv.org/abs/2606.08777), [CRM source](https://github.com/dsgiitr/mechanistic-counterfactual-grounding/blob/4f2b277b4bab948d8f885fa2057e82a7b7e579ae/training/crm_loss.py), [trainer](https://github.com/dsgiitr/mechanistic-counterfactual-grounding/blob/4f2b277b4bab948d8f885fa2057e82a7b7e579ae/training/trainer.py#L205). Code/README evidence không là full-method clearance.

Related leads: [VisualFLIP](https://github.com/didizhu-judy/VisualFLIP/tree/fa677e35036e4dc423911ec4eb788ae587dc1f8f) README mô tả benchmark paired flips + irrelevant edits, chưa đọc full paper/không tải data; không claim flip testing là mới. [Grounding-MLLM](https://github.com/bupt-mmai/Grounding-MLLM/tree/e1abf08a768639b9ea0b7aabe83ad544187c4b72) README về counterfactual text queries/nonexistent objects, chưa method-depth. [MED](https://github.com/Relaxed-System-Lab/hallu_med/tree/f05a6883e4b9e2c6ddde80c9a84edf5983b1bed2) README hiện chỉ dẫn generation scripts; không xác minh exact feature consistency từ README. CTM+Stay/COMO vẫn phải so trước claim, không loại vì khác task hoặc vì nguồn chưa tải được.

Search hôm01/11: primary author GitHub/references, queries grounding+consistency+captioning; counterfactual+grounding; phrase+equivariance; cycle+phrase grounding; counterfactual+attention+captioning; Inducing Causal Structure; Human Importance-aware. Không có scholarly-search connector; Github repository search có coverage hạn chế, không chứng minh absence khi0results. Public publisher copy của IIT là full text mới đọc, không chỉ đổi tên hypothesis cũ. Receipts/raw texts ở scratch ngoài repo; không đưa copyrighted PDFs vào Git.

## Pilot nhỏ, đối chứng và điểm dừng

**Hai train arms, cùng regional adapter/crops/box và token labels:**

- Control: L_common + factual CE replay trên base/donor với các target labels tương ứng.
- Candidate: L_common + partial-IIT CE cho cùng số valid target events được chuyển sang intervention contexts. Protected/updated labels đều có ở cảhai; khác nơi áp supervision, không cho candidate nhiều nhãn hoặc nhiều cập nhật hơn. Báo forward/scoring budget thật; không giả số ms bằng nhau chỉ vì parameters như nhau.

FrozenC0 là reference. Không train một loss/critic rồi chỉ dùng nó chấm đầu ra; cùng fc_out sinh captions. Khóa mask, λ, LR, data/order, update budget, decoding và source-group split trước run; chưa chốt GPU quota/ETA hoặc viết trainer.

1. **Feasibility trước đầu tư nhiều nhãn:** audit metadata tối đa6base–donor pairs từ positive COCO boxes có class/alias rõ và predicates không tham gia; đối chiếu crop/proposal visibility. Dùng tối đa vài MED development pairs cho attribute sanity nếu box correspondence đã có; chưa có thì đánh dấu thiếu, không tự gán gold. Mục tiêu xem label/mask có đặt hợp lệ không, không lựa chọn bằng model scores.
2. Pilot sau duyệt cần fit/held source images tách nhau và chưa dùng15diagnostic làm test. Donor lẫn base đều phải obey split; thêm split held class-pair combinations nếu có đủ data. Sample size/schedule khóa sau biết coverage; chưa có căn cứ hứa một số pairs sẽ đủ học hoặc đạt significance.
3. Endpoints cơ chế: factual grounding, held interchange updated correctness và protected correctness/joint, UNKNOWN coverage, residual norms và can thiệp value cùng weights. Identity/self-swap không đổi output; joint permutation của records invariant. Gold anchors chỉ evaluator/loss, inference dùng predicted banks.
4. Endpoints thật: free captions trên ảnh tự nhiên, CIDEr/CHAIRs/i/false mentions+captions/recall/length so ownC0 và control. Caption không nói tới fact là coverage, không tự chấm đúng/sai; màu không thay object-CHAIR. Chỉ đẹp interchange/probe nhưng caption không tốt hơn thì chưa đạt mục tiêu.
5. Chỉ mở confirmation nếu candidate thêm utility so control, có tác động đúng ở unseen swaps, và có tín hiệu cùng tăngCIDEr/giảmobject hallucination không nhờ rút ngắn/bỏ nội dung. Nếu control đã đạt ngang, rút luận điểm cần intervention training. Không sửa weights/epochs trên cohort đã xem để cứu hypothesis; claim paper cần independent test và uncertainty.

**Chi phí:** adapter dimension128 có quy mô cỡ vài trăm nghìn weights theo các projection trên; chưa parameter-count trên implementation. Native vocab head chỉ tính một distribution/token, nhưng thêm Kcrop encodings/ảnh và patched readout events. Train có thể cache crop features vì CLIP frozen; inference vẫn trả chi phí crop. Chưa benchmark; không dùng quota T4 lịch sử làm số hiện tại.

## Tự phản biện và giới hạn đóng góp

- Nguồn label độc lập chưa đủ để cho toàn caption sau swap; partial mask giảm lỗi gán nhãn chứ không tạo oracle. Relation/count/overlap/ambiguous referent cần UNKNOWN thực sự. Nếu không xây được valid masks với lượng annotation vừa sức thì không nên chạy.
- Crop cùng class có thể thiếu chi tiết, chứa background hoặc tạo shortcut theo dataset/style; donor crop resolution/dimensions có thể là dấu hiệu. Phải có held donors, multiple classes/directions và matching preprocessing. Nội bộ hoán đổi vùng không chứng minh causal process ngoài thế giới.
- Model có thể tăng W_o quá mạnh để thắng virtual tasks rồi gây lỗi caption tự nhiên. Factual CE/clean inputs và protected facts là controls, không guarantee. Không đề xuất ép residual/mass/orthogonalization để gọi thành đóng góp thứhai.
- Nếu chỉ thay JS bằng generic IIT trên một task khác, chưa đủ innovation. Phần cần bảo vệ là **một cơ chế tạo partial targets hợp lệ từ positive regional labels và dùng chúng để học channel vùng→token của captioner**, cùng giá trị thực nghiệm ngoài data augmentation/grounding. Partial-label learning tự nó cũng có prior; exact novelty chưa xác minh.
- Tôi ưu tiên hướng này hơn X018 hiện tại vì failure “grounding tốt nhưng output bỏ qua evidence” có một phép thử/gradient trực tiếp trong training. Đây vẫn là hypothesis về hiệu quả; không quy nó thành nguyên nhân đã đo của C0/D/E hoặc hứa hai metrics sẽ tăng.

**Kết luận hiện tại:** có một phương án phương pháp cụ thể để Codex1/user phản biện, chưa có chứng nhận mới. Blockers là full-method comparison với mechanistic/interventional grounded generation và feasibility partial-label construction; không phải thiếu thêm tên module. Hai full texts cần bổ sung nếu nguồn tiếp tục bị chặn là [Gupta et al.,2026](https://arxiv.org/abs/2606.08777) và [MED](https://arxiv.org/abs/2506.07227). IIT đã tải/đọc được, không cần yêu cầu gửi lại.

Kiểm tra lượt này: đọc source/PDF có preflight, đại số JS và toy2token về local effect bằng Python chuẩn. Không import Torch/autograd, chạy mô hình/notebook/scorer, stage datasets/crops, thay experiment code hoặc sheet. Không có model result mới.
