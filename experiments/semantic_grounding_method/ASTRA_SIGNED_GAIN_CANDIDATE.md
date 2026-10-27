# Hai ứng viên research gap sau phản biện joint phrase–region loss

27/10/2026. Brainstorm method-level; chưa code ML/notebook/run. Không dùng atlas20 làm gold hoặc chờ4–6manualpairs. Prefix inventory root đã verify:95,86%singleton; multi-box0,534%. Vì vậy joint-set xử lý ambiguity là correctness guard, không đủ làm contribution chính.

## A. Ưu tiên: căn chỉnh phần cải thiện xác suất của cụm với nguồn bằng chứng có dấu

### Vấn đề khác spatial/semantic auxiliary loss

Một attention map hoặc importance ranking đúng vùng không xác định được (i) correction thật sự làm probability của linguistic claim tăng hay giảm so với A; (ii) bao nhiêu gain đến từ vùng được ghép; (iii) positive/negative contributions có triệt tiêu với biên độ rất lớn không. Không nhận các câu hỏi này là hoàn toàn chưa từng được nghiên cứu: HINT, LRP-IFT, CDEP và IG là closest priors trực tiếp. Candidate gap là constrained adaptation của strong frozen captioner theo attributable probability gain, không phải attention hoặc attribution nói chung.

Ví dụ: region contribution+5 và complement−4,95 cho netgain+.05 có thể đẹp theo ranking/positive-map nhưng có cancellation lớn. Targetregion đứng cao nhất cũng có thể đi với netgain0. Các regularizer cần phân biệt hai trường hợp này với improvement nhỏ nhưng có căn cứ. Đây là vấn đề trên computation graph, không tự là causal/factual truth ở pixels.

### Semantic unit, source, visual evidence

Semantic unit: actual reference head/name phrase gồm một hoặc nhiều A tokens, với đúng prefix và phrase–box pair của verifiedlabelbankv1.1(4495images/5602regions/17904spans/71classes). Không ngầm có adjective labels; không cần CLIP text teacher hoặc modifier expansion để định nghĩa candidate này. Pairing là weak supervision, không manualgold. Missing/ambiguous pairing không nhận region objective. Không word-foil negatives, random-imagefalse hoặc VCOCOride0=>Englishfalse.

Evidence units: projected patch tokens trước Q, Z196x512. Chỉ học branch mới; A hoàn toàn frozen. h_t là last hidden state A đưa vào vocabulary head, đã image+prefix-conditioned. q_t=LN0(h_t)Wq; k_j=LN0(Z_j)Wk; v_j=normalize(LN0(Z_j)Wv). Wq/Wk/Wv:512x128,Wo:128x512,no bias,Wo0;262144weights. a_t=softmax(qK^T/sqrt128). Logit correction từng patch:

b_tj(w)=a_tj [W_vocab,A(v_j Wo)]_w.

actual logits=logits_A+sum_j b_tj. Có thể chặn normWo để giới hạn correction; nếu dùng phải cùng constraint ở mọi control. No gates/orthogonality/topK. No inference GT/ref/parser. Đây là additive evidence branch, không open-vocab.

### Signed contribution có completeness đối với log probability

Với fixed prefix và token w, định nghĩa path chỉ scale branch: p_alpha=softmax(logits_A+alpha sum_j b_tj). Prefix, A memory và attention không thay đổi dọc path. Baseline alpha0 chính là A trên CÙNG prefix; không zero/mask input image.

c_tj(w)=integral_0^1 [b_tj(w)−sum_u p_alpha(u)b_tj(u)] dalpha.

Khi tích phân đúng: sum_j c_tj(w)=log p_theta(w|prefix)−log p_A(w|prefix). Đây là IG/Aumann–Shapley completeness được kế thừa, không là theorem mới. Nó invariant với common logit offsets và tính cả softmax competition; không dùng raw target-logit margin làm probability gain.

Với singleton phrase phi, cộng c_tj(y_t) qua đúng token span trên reference prefixes để được c_phi,j và Delta_phi=logp_theta(phi|prefix)−logp_A(phi|prefix). Không nhận Delta_phi là change của caption greedy hoàn chỉnh vì hai policy có thể sinh prefix khác nhau.

Multi-reference guard: khi cùng prefix có các phrase alternatives, không đòi mọi alternative đều tăng probability độc lập. Đặt F(alpha)=log P_alpha(S|prefix), probability mass của tập valid observed phrase-prefix events; deduplicate và xử lý prefix-overlapping events bằng trie, không doublecount. F1−F0=Delta_S. Pair-specific attribution:

c_phi,j=integral_0^1 pi_alpha(phi) sum_(t in phi) [b_phi,t,j(y_t)−sum_u p_phi,t,alpha(u)b_phi,t,j(u)] dalpha,

pi_alpha(phi)=P_alpha(phi|prefix)/P_alpha(S|prefix). Khi prefix events disjoint, sum_phi,j c_phi,j=Delta_S. Với overlapping events, dùng disjoint trie frontier và region target set tương ứng hoặc skip group nếu không giữ được semantic pairing; không cộng xác suất lặp. Multi-reference handling là normalization guard, không novelty claim.

Cho M_Bj fractional box/cell overlap theo đúng preprocessing. C_G=sum_phi,j M_Bphi,j c_phi,j; C_U=Delta_S−C_G. Giữ phrase và region cùng pair index. Đặt K=|C_G|+|C_U| để đo cancellation giữa licensed-support và complement groups; không nhận rằng nó kiểm mọi cancellation bên trong từng group. Signed contributions của các valid alternatives trong cùng group được cộng trước trị tuyệt đối, tránh phạt riêng redistribution giữa valid phrases như thể đó là semantic mistakes.

Không nên materialize196x10000logit contributions: sau khi có path-averaged probability vector, nhân W_vocab^T với vector đó để lấy contrast direction512 rồi dot với perpatch hidden correction512. Numerical quadrature chỉ thêm softmax/vocab operations trên head, không thêm imageencoder forward. Chưa đo runtime; completeness chỉ approximate ở implementation, phải report residual và numerical tolerance thay vì nói exact theo lý thuyết cho mọi quadrature.

### Constrained objective đề xuất để root phản biện

m_S=min(.05,.1[-logP_A(S|prefix)]): targetgain dương nhỏ, có xét headroom; P_A(S)=1 thì không đòi thêm probability mass. rho=.8,kappa=2 là starting design constants, chưa optimized.

L_gain=[m_S−Delta_S]_+.

L_cancel=[K−kappa max(Delta_S,0)]_+.

L_groundgain=[rho max(Delta_S,0)−C_G]_+.

L=L_main+lambda_E(L_gain+lambda_c L_cancel+L_groundgain).

L_main là pinnedSCST reward của dự án+.1KL(A||student), tính trên actual corrected policy. Tham số lambda chưa khóa vì đây là methodbrainstorm, không runnableprotocol. Mục tiêu là boost observed positive phrase mass và quy phần lớn netgain cho đúng support. Không bắt unmentioned facts phải false; complement-region constraint là prior về nguồn computational support, không là annotation absence. C_G có thể vẫn phản ánh context được CLIP truyền toàn ảnh; không claim pixel-level causal grounding hoặc factual certificate.

L_gain một mình là reference-based preference/NLL-like training, không novelty. Cần giữ nó trong controls để không gọi extra reference exposure thành benefit của alignment. Giữ cùng L_cancel trong control để isolate directional region term. L_cancel dùng cùng region partition nhưng đối xứng giữa hai groups: control này KHÔNG region-blind hoặc hoàn toàn không nhận annotation information. Nó là đối chứng mạnh hơn, tách directional licensed-source requirement khỏi reference exposure và symmetric cancellation regularization. Không chọn positive-onlymaps rồi bỏ sign/completeness trong numericalscorer.

### Controls và falsification

Tối thiểu mọi arm cùng262144paramhead,source/data/optimizer/budget:

1 C: L_main+lambda_E(L_gain+lambda_c L_cancel), không licensed-region term.
2 H: C cộng HINT-style importance-ranking supervision trên cùng phrase–region pairs; phải ghi adaptedbaseline, không fulloriginal reproduction. Dùng per-decision scores, không word-agnostic mass nếu muốn so closestprior nghiêm túc.
3 T: C cộng L_groundgain.

A là frozen reference, không train. T cần hơn A,C và adaptedH về free-generationCIDEr cùng guardsCHAIR/recall, đồng thời có attributablegroundgain trên heldout eligible pairs. Gain so C mà không so H chỉ chứng minh region guidance hữu ích, chưa ủng hộ claimedmethod advantage over prior. Không thêm manualpairs mới. No end-to-end factual guarantee; reference-prefix gains và generated caption grounding phải báo tách biệt.

Các phản ví dụ đại số cần giải trước codeML: common-offsetlogits không thay đổi Delta/c; gradient/attention đẹp nhưng Delta0; C_G rất lớn+C_U âm gần bù; cùngprefix alternatives đã chiếm toànprobmass; integratedsum khớp actual logprob change. Đây là kiểm tra định nghĩa method, không một frozen-model audit mới.

## B. Backup: căn chỉnh event scope và identity, không chỉ chọn scene subgraph

Semanticunit là caption-specific event plan: entityIDs, predicate, argumentIDs và explicit scope/shared-predicate structure. Ví dụ hold(person,toothbrush) và nearby(cat,person) không được serialize thành hold(person,toothbrush AND cat). Visualevidence gồm entity slots+pair interaction support; không một unionbox cho toànsentence.

Supervision từ conservative parses của human references; chỉ endpoint có phrase–box pair hợp lệ mới nhận visualbindingloss. Captionplan mô tả những gì một câu chọn nói, không scenegraphđầyđủ. Absence một edge trong plan không được dùng làm label rằng edge đó false trong image. Parseruncertainroles skip/latent; không VCOCO0negatives hoặc synthesizedfalse captions.

Một thiết kế cụ thể: posteriorq(G|I,reference) và imagepriorp(G|I); entityslots predict boxes/semanticIDs từ patches; eventnodes có typedargument pointers; lexicalizer nhận eventstate và chỉ hợp nounconjunction khi plan cho phép shared predicate. Loss=E_q[-logp(caption|I,G)]+betaKL(q||p)+lambda_endpoint L_box+lambda_role L_partialrole. L_partialrole chỉ áp known positive caption roles, không unmatchededge BCEzero. Inference dùng p(G|I), predictedslots và scope-aware lexicalizer; không reference/GT/posteriorinput.

Đối chứng quyết định giữ cùng planner/data/loss nhưng lexicalizer dùng graphfeatures không scope constraint so với lexicalizer giữ argument/scope; đồng thời so eventplan có typedroleIDs với bag-of-entities nếu muốn xác nhận scope mechanism. Plan-to-text consistency đẹp mà plan sai ảnh không đủ: captiontruth/quality và relationoutcomes phải đo riêng.

Novelty candidate là bảo toàn argument/scope khi chuyển evidence plan thành surface sentence, không graphcaptioning/GNN/CVAE/modulecontroller nói chung. Chưa loại trừ overlap với structured/controllablecaptioning. Rủi ro cao: plan hallucinations, parse noise, exposure gap và ít doublygrounded references. Chỉ từlabelbank5602regions/4495images, sốimages có>=2 bankregions không thể vượt1107; đây là upperbound, KHÔNG sốeventtrainingexamples đã có. Vì vậy không chọn B làm implementationcandidate trong budget hiện tại.

## Closest priors đã kiểm tra

- HINT,ICCV2019: đã đọc4.2–4.3, importance=sumchannelgradient của GT outputscore, rankingloss so humanimportance, cộngtaskloss. Importance có sign; không được claim HINT chỉ dùng magnitude. https://arxiv.org/pdf/1902.03751
- LRP-IFT,InformationFusion2022: đã đọcV.C.2/Eqs28–32/Algorithm2 trong arXiv2001.01037v5. Signed relevance reweights context+hidden features rồifine-tuneCE/SCST; không được claim use signedexplanations để cải thiệncaption là mới. https://arxiv.org/pdf/2001.01037
- CDEP,ICML2020: đã đọc3.2–3.3, CDdecomposeslogits vào featuregroup/rest, tổngquát explanationpenalization+taskloss. Generalframework bao trùm nhiều regularizer của candidateA; candidate cần phân biệt bài toán/properties cụ thể, không nhận inventexplanationregularization. https://proceedings.mlr.press/v119/rieger20a/rieger20a.pdf
- IntegratedGradients,ICML2017: baseline-path attribution và completeness là inherited. https://proceedings.mlr.press/v70/sundararajan17a.html
- Sub-GC,ECCV2020: đã đọc3.2–3.4; chọn subgraph bằng noun-nodeIoUlabels rồiattentionLSTMdecode. Không nhận semantic subgraphselection là mới. https://arxiv.org/pdf/2007.11731
- CNM,ICCV2019: đã đọc phần module/controller từprimarypaper; OBJECT/ATTRIBUTE/RELATION/FUNCTION vàlinguisticmoduleloss có trước. Không nhận tách semantic modules là mới. https://arxiv.org/abs/1904.08608

## Quyết định

Ưu tiên A cho vòng hoàn thiện method và closest-prior equivalence check; B là backup dài hơi. A khác previouscosine+massloss ở constraint về actual probability gain có accounting và cancellation, không dựa vào tỷ lệambiguousprefix. Noveltypotential chưa xác lập: nếu equivalent với attributionregularization hiện có chỉ khác notation/baseline thì loại, không cứu bằng tênmới. ChưaGOcodeML/train/notebook; chưa hứa paper hoặc gain. Không mở thêm genericdataaudit.
