# Hypothesis: key giữ hướng diễn đạt, value chia sẻ fact đã ground

30/10/2026. **Một corechange để root/user đánh giá; chưa xác nhận novelty, chưa chọn triển khai/code/GPU.** Shared-referent reranking chỉ còn dự phòng sau phản hồi user. Không quay về attribution/T, generic latent planner, scalar veto hoặc mass-envelope.

**Amendment 30/10, sau đoạn lịch sử trên:** user đã đồng ý giữ K/V làm ứng viên chính, hoàn tất nearest-method check rồi chuẩn bị pilot D/E nhỏ. [Nearest-method closure](KV_NEAREST_METHOD_CLOSURE_2026-10-30.md) chọn (a): có cơ sở pilot như cải tiến kế thừa, chưa xác nhận tính mới hoặc hiệu quả. Multi-key/shared-value memory và group averaging đều đã có prior; E còn tương đương logsumexp group addressing với một value/fact, không phải attention operator mới. Đúng một core change giữ nguyên. Ontology, nguồn nhãn, common-loss activation và runtime budget còn phải khóa trong protocol; không suy quyền chạy GPU từ amendment này.

## Gap → mechanism, viết thẳng

**CSA-Net dùng object/relation word prototypes để chọn/fuse semantic prompts vào region features; cùng nguồn biểu diễn cung cấp cả key lẫn value của cross-attention. Nhưng các cách diễn đạt khác nhau của cùng một fact không được ràng buộc phải mang cùng semantic value. Ta thay independent directional values bằng một phép chia sẻ value giữa đúng hai cách biểu đạt tương đương của một instance-grounded fact, trong khi giữ keys riêng theo hướng truy vấn.** Giả thuyết: dưới supervision hữu hạn, việc ngăn lựa chọn lexical key làm đổi nội dung fact có thể giúp captioner dùng bằng chứng quan hệ nhất quán hơn mà không ép xác suất/ngữ pháp hai câu bằng nhau.

Đây là giả thuyết inductivebias, **không phải đã xác định nguyên nhân lỗi A/C0**. R/T/W chỉ cho biết tăng box-attention chưa đủ trong recipe đã thử. Chưa đo được lexical-key leakage gây bao nhiêu lỗi caption; không dùng số cũ làm bằng chứng nhân quả cho cơ chế này.

## 1. Fact là gì? Điều nào được và không được đồng nhất?

Fact ở đây là **một assertion trên các instance boxes cụ thể**, không phải canonicalpredicateID hay tập mọi pairs cùngpredicate:

\[
f=[(I,i,p,j)]_\sim,
\quad (I,i,p,j)\sim(I,j,p^{-1},i).
\]

Ví dụ dog_i BELOW table_j và table_j ABOVE dog_i cùngfact. Dog_i ABOVE table_j là orbit khác; dog_k BELOW table_j cũng là fact khác. Proposalindices chỉ định danh vùng trong ảnh, không GTinstance ởinference. Sắp canonicalID chỉ phục vụ bookkeeping; không phải contribution.

Chỉ có inverse map đã xác nhận về nghĩa và frame; p^{-1}=p cho quan hệ đối xứng đúng theo định nghĩa đang dùng. Không blanket-map near/beside/on/above, không man-wear-shirt↔shirt-on-man. Left/right chỉ dùng cùng fixedreferenceframe rõ; unsupported/ambiguous giữUNKNOWN. Truthlabel y của q và q_inverse được transport theo equivalence, **không thành hai quan sát độc lập**. Arg-swap giữp không tự là false.

## 2. Đúng một corechange trong decoder read

Giữ upstream CLIP/cache/nativeprompt/Q-Former củaC0. Pair features: ROI_i,ROI_j,union fractionalpool từ rawCLIPgrid768→sharedprojection128; signedgeometry8; frozenrelationprompt512→projection128. Không input GTclass/reference vào forward; relationqueryp được enumerate từ ontology nhỏ đã khóa, không lấy futuretoken từcaption.

F là một shared nonlinearMLP, output h_ijp∈R128. Input gồm [e_i,e_j,u_ij,g_ij,t_p] và một interaction thông thường (W_g g_ij)⊙(W_t t_p). **CùngF/interaction ở cả haiarms**; không nhận interaction là novelty. Nó tránh giới hạn rõ của additiveF, nhưng không bảo đảm đã giữ semantic direction.

```text
predicted ROI_i/ROI_j/union + signedgeometry + query p
    → shared F → h_ijp(128) ──→ Wk → key_ijp(64)   [giữ riêng]
                     ↘
paired h_ji,inverse(p) → mean trong đúng fact-orbit → Wv → value_f(128)
native h_t(512) → Wq(64) → attention trên keys → weighted values(128)
    → Wo(128→512) + native h_t → native vocabulary head → actual word distribution
```

**D, strong control:** K_ijp=Wk h_ijp, V^D_ijp=Wv h_ijp.

**E, proposed:** K_ijp nhưD,
\[
V^E_{ijp}=W_v\frac{h_{ijp}+h_{ji,p^{-1}}}{2}
=V^E_{ji,p^{-1}}.
\]

Giữ cả hai slots mỗiorbit ởD/E, cùngmemorylength, params, F, key/valueprojections vàcaptionpath. Đây là group-average/projection đã biết; delta được đề nghị là **directional-key / instance-fact-value factorization trong learnedcaptionalignment**, không operator toán mới. Không thêm semantic planner, GNN, verifier reranker, coverage hoặc outputprobabilitycontroller.

Viết v_+=(v1+v2)/2,v_-=(v1−v2)/2 cho hai directedvalues trong mộtorbit. Contribution củaorbit ởD là (α1+α2)v_+ +(α1−α2)v_-; E bỏ đúngterm (α1−α2)v_-. **Câu hỏi nghiên cứu là liệu phần value thayđổi theo preference giữa hai equivalentkeys có chứa nuisance hay information cầnthiết.** Không mặcđịnh nó lànuisance. Mean cũngcóthể làmnhỏnorm/readout; phải báonorm và mứcbranchđượcdùng, khôngnhậngain giảmstrength thànhsemanticproof. Reader-off sau khi mởdecoderlayers vẫn giữadapteddecoder, khôngđược gọi làrawC0.

Inference: lấyK≤6predictedboxes theo rulepinned; giữ toàn bộ K(K−1)|P|slots cho inverse-closed typedP, cộng mộtnullvalue0. Không truth-threshold/GTpruning; false/unknown hypotheses không bị coi là facts true. Nativeattention phải học chọn evidence; vì thế branch vẫn có thể sai hoặc bịignore. NếuP có5types thì≤150slots+null; countP thực chưa khóa. Emptybank→null-only. GT chỉ loss/evaluation; không GTsubstitution.

Trainable: F/projections/Wq/Wk/Wv/Wo và hai decoderlayerscuối; nativeencoder/Q-Former/prompt/embeddings/vocabularyhead giữfrozen nhưmatchedscope đã biết. C0 raw làreference, D/E cùng init/scope/schedule/data. Wo có thểinit0 để parity banđầu; không đồngnghĩa mọigradient/utility đãđượcchứngminh. Không có truthhead riêng thay đường tới actualcaption: captionCE/SCST đi qua chínhkeys/values/readout ởtrên.

## 3. Giới hạn kiến trúc phải kiểm trước claim

Nếu F additive: h_ijp=Ae_i+Be_j+Ct_p+Dg_ij, thì averagevớireverse có thểlàm **V_E(i,j,p)=V_E(i,j,p_inverse)**, xóa cảphysicaldirection. Đây là counterexample đại số do root/Astra chỉ ra; kháckeys không phục hồi thôngtin đãmất ởvalue. Nonlinear interaction làm haiorbits có thể phânbiệt, không chứngminh chúng sẽphânbiệt sauhọc. E còn có thể xóa perspective/rolefeatures hữ́uích màdecoder cần. Entityidentity phải còn trongvectorfact128, không chỉ mộttruthscalar haybagofclasses.

Không épP(sentence)=P(inverse-paraphrase). Prefix/nativeLM còn phânbiệt subject, syntax, salience vàlength; cùngfactvalue không phải mộtlatent syntax space mới. Nếu sharedvalue làmdecoder nhầm dog/table role thì hypothesis fail.

## 4. Supervision và strong matchedcontrol

Nguồn đề nghị: COCOcaptionrefs; GQA/VGpositive groundedtriples; VSRobservedbinarytruth trên referentsunique và predicted-boxmatches. GQA mới chỉ đọc schema, chưa stage/map. Chỉ KarpathyTRAIN/restval, image-groupedfit/held, loại val/test/fixeddev/oldVSRheld; identity mapping theoCOCOIDs. Không suy unannotatedpredicate/unmatchedproposal/khôngnhắctrongreference làfalse. Multiplepredicates cùngpair có thể cùngtrue. Ambiguous/missinglocalbinding→masklocal loss/reportcoverage.

D được **cùng canonicalfacts, cùng hai verbalizations tươngđương, cùngsame-pair labels và standardmulti-positive alignment** nhưE; không dùng mộtbaseline chỉ thấy1view. CaptionCE/SCST gốc và anchor nếu dùng phảigiốngnhau. Auxiliary loss dùng h của **cảD/E** trướcvalueprojection:

- Standardmulti-positive contrastive alignment tới Q_equiv của observedtruefact; negative chỉ khi có contradiction observation cho đúngimage/pair/query, không argswap mặcđịnh haybatchcaptionkhácảnh mặcđịnhfalse. Không đủpositive/negative choevent thì skipcontrast/reportcounts.
- Observedbinarymatching loss cho true/falseVSR hoặc positiveGQA, chỉpredicate được quan sát; equivalentform có cùngy, mean theooriginalfact đểkhôngdoublecount.
- Actualcaptiongeneration loss qua nativewordhead; không thêm một loss mới riêngchoE đểcứu novelty. Hệsố/schedule/budget chưa đượcchốt, không nhận MP/BCE/SCST là mới.

**E−D chỉ kiểm hardvalue-sharing bêntrên standardsemantic/equivalence supervision.** NếuD đãhọc h1≈h2 vàcùngcaptionbehavior, projection dưthừa; dừngdelta, khôngthêmloss hayđổitênđểcứu. Việcdata có thêmparaphrases không được nhận thànhgain củaE.

## 5. Minimum experiment, ba mức bằng chứng

Sau khiuserchọn hypothesis/noveltyscope mới bànbudget cho đúng **D/E matchedpilot vàownC0**, cùngfixeddev đãdùng, khôngsweep. Chưa chuẩn bịcode hoặc chạy bấtkỳmức nào.

1. **Construction, khôngsemanticproof:** swap chỉhaiKEYS trong mộtorbit, giữvalues vàotherslots. Tổngsoftmaxmassorbit giữ; Econtext/distribution/caption invariant trongtolerance, D cóthểđổi. Đây là identity củaattention, cóthểkiểmNumPy/static, không chứngminh learnedgrounding, physicalcausality hoặc usefuldecision. Nativeotherpaths giữnguyên.
2. **Nontrivial semanticselectivity:** trênimage-held facts cóobservedlabels, kiểm true/falseimage discrimination cùngquery, opposite-fact orbit separation khi cónhãn hỗtrợ, vàdifferentinstancefact retrieval. Giữ UNKNOWN khi chỉ biết hai captions khôngtươngđương; không tựgán mộtcâu false. Phải báo both correctequivalence và **khôngcollapse** direction/instances, khôngchỉembeddingdistancegiảm. Testconstruction PASS màtruthselectivityfail→dừng.
3. **Actualcaption:** E phải cóaddedutility vsstrongD và tínhiệu tăngCIDEr/giảmCHAIR falsecounts vsownC0, recall/lengthguards, báopaireduncertainty. CHAIRkhôngđo relationtruth; semanticclaims cầnnhãnobserved riêng. Branchignored, captionchangechỉnuisance, E≈D, hoặc quality/factualitytradeoff khôngđạt→khôngscale. Không bằng chứng thì không hứaaccepted hoặc tựxinGPUtiếp.

Một testpredictedbehavior cụthể: trên cùngảnh dogbelowtable, thay preference giữa hai equivalentkeys mà giữevidencefact. E không được đổi supported-vs-contradicted predicate choice chỉvìretrievalwording; D cóthểđổi. Tuy nhiên Eunchanged **chỉlàconstruction**; giátrị học chỉ có nếu nguyênbản đãselectđúngfact và phânbiệt được ảnh/fact đối chứng cónhãn, rồi captionthựctốt hơn. Không dùngtestnày mộtmình làmGO.

## 6. Đã có / delta / chưa xác minh

**Nearestcaptionmethod:** [CSA-Net, DOI10.1109/TCSVT.2023.3343520](https://doi.org/10.1109/TCSVT.2023.3343520), đãđọc localprimary§III-A/B Eq2–12 và§III-C Eq13–17. Object/verbprototypealignment vàsemanticcrossattention đãcó. Eq10 dùng cùngsemanticfeatures làmnguồnK/V; chưathấysharedfactvalue constraint trongphầnđãđọc. Không khẳngđịnhCSA thựcnghiệmmắcđúnglỗikeyleakage.

**Nearestoperator:** [Group Equivariant Stand-Alone Self-Attention, ICLR2021](https://arxiv.org/pdf/2010.00977), đọc§3,§4.1–4.3,§5overview; chưađọcmọiproofs/Eq14–16implementation. Group-equivariantattention/averaging/invariantrepresentations đều đãcó. §5 còncảnhbáoinvariancequámạnh làm mấtorientation—phùhợprisk ở§3. Khôngclaim groupattention mới hoặc guaranteeexpressivity từprojection.

**Nearestmeaning/form:** [DSS-VAE,ACL2019](https://aclanthology.org/P19-1602.pdf), đọc§3.1,§3.2.1–3.2.2 và§3.2.3intro. Separate semantic/syntaxlatents vàautoregressivegeneration đãcó, khôngclaimfirstmeaning/formdisentanglement. Candidatekhôngtạo2VAElatents; chia directionalretrievalkey vàsharedgroundedfactvalue trongcrossattention.

Collisionchecksđãđọc: [MR-Net,AAAI2019](https://ojs.aaai.org/index.php/AAAI/article/download/4819/4692), MutualRelationLearningEq1–6: vp≈−vp_reverse, noncollapse,untiedclassifier,semanticranking. Vìvậy anti-symmetricrelationvector khôngnew; nócũngkhôngtựđảm bảoactualpredicateinverseequivariance hoặcsharedfactvalues. [Structure-CLIP,AAAI2024](https://arxiv.org/pdf/2305.06152), §3.1Eq1–8/§3.2Eq9–12: scenegraphnegative/structuredtripleembedding đãcó; Eq2argswap cópotentialfalsenegative vớisymmetricrelation, nhưng chưakiểmcodefilter/frequency/causalimpact nênkhôngnhậnbảnđềxuấtđãchứngminh lỗi thựcnghiệm của họ. C²LIP/multi-positiveconceptalignment trongledgercũ làprior, khôngđọc lại/nhậnMPnew.

**Delta đề nghị** làtask-specificK/Vfactorization+hardvaluecoupling cho semanticfact-equivalence trêninstancepairs, đối chiếu vớistandardmulti-positive/canonicalsupervision cócùngactualcaptionpath. Nó là cơchế cụthể hơn ontologyID/knownrerankertrêncheckpointkhác, nhưng **chưa đủ để nhận đãmới hoặc cóích**. Genericgroupoperator, content/form separation vàmultipositivelearning đều làkếthừa. Chưa kiểmhết prior equivariantcrossattention/KVdisentanglement đã làmđúngcấu trúcnày; focusedpapersởtrên khôngchứngminhfirstness. Userđược đánhgiábar contribution trước triểnkhai; chưakhuyênGPU.

Tôi rút future-objectchainrule/persistentplan khỏicandidate: AR model khôngbịbấtkhảthivìchưaemitobject; Shen2020segment-recordmarginalization/Human-likeCIC2021 đã cópriorrấtsát trongreviewcũ. Khônggọi nónew bằngđổitên.

## Receipt / chưa làm

### Root review sau khi đọc bản hoàn chỉnh

Root đã đọc notes này, không đọc lại các PDF mà Astra đã ghi scope. Đây là một core mechanism đủ cụ thể để trình người dùng phản biện: loại đúng antisymmetric value component của hai views cùng fact trong actual cross-attention, trong khi cùng nguồn upstream, supervision và trainable scope. Chưa nhận đủ mới, đã đúng nghĩa hoặc có hiệu quả. Không thêm một architecture thứ hai vào proposal.

Các điểm cần giữ rõ khi trình:

- Keys **được phép khác nhau**, chứ chưa được bảo đảm đã học direction hoặc lexical form hữu ích. Shared h còn chịu MP/caption supervision; nếu h hai views đã giống nhau, cả K và V có thể giống và phép chiếu không còn ích lợi. Không sửa bằng cách tự thêm loss sau khi nhìn kết quả.
- Giá trị đối tượng/quan hệ không được chứng minh bằng tên “fact code”. Construction invariance là identity; semantic separation/truth và actual caption phải được kiểm riêng như ba mức ở trên. Dataset/ontology cần phân biệt image fact với relation label chung, và không giả tất cả đảo đối số là nhãn âm.
- Counterexample additive-F vẫn áp dụng; nonlinear interaction làm tránh collapse trở thành điều có thể học, không một bảo đảm. E còn thay norm/readout strength, nên caption gain đơn lẻ chưa xác định được nguyên nhân là semantics.
- Group projection và meaning/form separation có prior. Focused comparison với CSA/GSA/DSS-VAE chưa hoàn tất kiểm tra toàn bộ nearest equivariant K/V caption-alignment literature; chưa gọi “phương pháp mới được xác nhận”. Một kết quả tốt cũng không tự chứng minh novelty.

Khuyến nghị root: giữ một hypothesis này để đánh giá contribution và nguồn dữ liệu trước khi bàn protocol/GPU. Nó có delta kiến trúc rõ hơn bản đồ thị cũ, nhưng người dùng chưa chốt mức contribution; không tự triển khai hoặc hứa chấp nhận bài báo.

Đãđọc source sectionsđãghi, phân tíchđạisố vàđịnhnghĩamodule/control. Chưacode/model/neural/GPU/extraction/staticNumPytest, chưađoC0failuremode/KVgap/sourcecoverage, chưacóperformance hoặcvenueassurance. KhôngrereadZIP. Rootphânphốisauphảnbiện, filekhôngnhậnđãpush/sheet.

Cacheprimary mới: `C:/Users/leduc/OneDrive/Tài liệu/ChatGPT/Đề cương/tmp/pdfs/gap_mechanism_20261030/`:

```text
group_attention.pdf dd39fa749d14637016014ed9d4e676c4637a5fd0dce340510882dfd685f35f51
syntax_semantics.pdf ae417ba705f81b6578b16ddec3fde64a94ad13f2df6b5d0dd9e42557d9b57076
mrnet.pdf 059153cbfaa72905c361e79b9084f5e8740cbac34301fa1e595600620a17276a
structure_clip.pdf bd760a846210664d8d6db559f80fca627e6ca9784af1c05f20c8eff71c559392
```

VTransE§3.1Eq1–3/§3.2 cũngđọcđểchặnsimpletranslationnovelty; downloadedplanningpapers khôngtínhfullread, dùngnotesđãverifiedtrướccủaShen/CIC. MR-NetAAAIfetchbanđầuconnection/gzip lỗi, cuốiđãlấyđúngPDFvớiUserAgent/Accept-Encodingidentity; không dùngbrokenHTMLlàmnguồn. Web503; OpenAlex/SemanticScholar chỉdiscovermetadata, cơchếtừprimaryPDF/localCSAtext.

CSAprimarytextlocal: `C:/Users/leduc/OneDrive/Tài liệu/ChatGPT/Đề cương/tmp/pdfs/research_review_20261020/Cascade-Semantic-Prompt-Alignment-Network-for-Image-Captioning.txt`, SHA256 `531b16550734126e5da84905164494b6a478dfe2eee07cb27acba8a168ea3124`. Đây làhashtext extraction cũ, khônggiảlàPDFhash.
