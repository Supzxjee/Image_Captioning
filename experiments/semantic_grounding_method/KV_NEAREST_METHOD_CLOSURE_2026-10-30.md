# Đóng nearest-method check cho K/V hypothesis

30/10/2026. Scope: [hypothesis đã khóa](RELATION_KEY_FACT_VALUE_HYPOTHESIS_2026-10-30.md), đúng một thay đổi Evalueaverage vsD; không thêmcandidate/module/loss. User giữ K/V làmứngviênchính và muốn nearestcheck rồipilotnhỏ, **chưa authorizeGPU/neural/extraction**.

**Kết luận (a): có cơ sở chuẩn bị pilot như một cải tiến kế thừa với delta xác định được.** Chưa tìm thấy exactfull-method collision trong những primarysections đãkiểm; điều đó khôngchứngminhfirstness. K/Vseparation, nhiềukeys→mộtvalue và groupaveraging đều cóprior trực tiếp; khôngđược trình chúng thành phátminhmới. Literaturecheck không xácnhận hiệuquả hoặc đủbar côngbố. Scope khảdĩ là **typedrelation-equivalence constraint chỉ trên learnedvalues trongactualcaptioncrossattention**, kiểm addedutility ngoài standardmulti-positive supervision đã matched ởD.

## 1. Ba primarymethods mới, readscope và collision

| Method / đã đọc | Đã có | Exactdelta còn lại / không được nhận |
|---|---|---|
| [Milleretal., Key-Value Memory Networks for Directly Reading Documents, EMNLP2016](https://aclanthology.org/D16-1147.pdf). §3.1 addressing/value-reading equations (không đánh số); §3.2 KBTriple,Window,Window+Title | Keys dùngmatchquery, values dùngresponse; key vàvalue featuremaps khác nhau. KB cóforward/reverserelation entries. Nhiềuwindows cóthểcùng trảcenterentity hoặcdocumenttitle, tức manykeys/sharedcanonicalvalue đãcó | Khôngclaim Klexical/Vsemantic hoặc nhiềuđịa chỉ/mộtvalue là mới. Candidatehọcvisualpairfactvalue rồiproject chỉtrongequivalentrelationorbit; khônglookupKBanswerentity/title. Chưa thấy inverse/symmetric groundedpair-value projection vànativecaptionpath trong phần đãđọc |
| [Punyetal., Frame Averaging for Invariant and Equivariant Network Design, ICLR2022](https://arxiv.org/pdf/2110.03336). IntroEq1; §2.1Def1/Eq2–3/Theorem1 vàcomments; chưafullproofs/apps | Group/Reynoldsaveraging làmnetworkfunction hoặcphầnnó invariant/equivariant; averageoverframe làcải tiếncomputational của họ | **Emean chính là trường hợpgroup2phầntử củaoperator đãbiết**, không newtheorem/newoperator. Candidatechọnsemanticrelationaction vàápdụngchỉV, giữK riêng trongcrossattention. Khôngclaim improvedexpressivity từprojection; nó hạnchếfunctionclass |
| [Lietal., Equivariant and Invariant Grounding for VideoQA, ACM MM2022](https://arxiv.org/pdf/2207.12783). §3.2/§4.1–4.4Eq3–20 | Grounding có desiredinvariance/equivariance, twoattentions+Gumbelsplit causal/environment, mixupinterventions, contrastiveanswerrepresentations vàERM | Cóprior invariantgrounding, khôngclaim ýtưởnggiữmeaningundertransformation mới. Transform của họ thaycausalscenes/questions hoặcenvironment, khác equivalentverbalization/argument-frame action trên mộtgroundedassertion. Chưa thấy value-only orbitprojection hoặccaptiondecoding trong cácphầnđãđọc. KhôngmượnSCMclaim của họ làm causalproof choE |

Reuseđãread: CSA-Net§III-A/B/C Eq2–17 (wordpromptaggregation/alignment vàcùngsemanticfeatureschoK/V); GSA2021§3–5overview (groupattention/invariancevsorientation risk); DSS-VAE2019§3.1–3.2 (meaning/formdisentanglement); MR-Net2019Eq1–6 (mutual/antisymmetriclatentpairrepresentations); Structure-CLIP2024§3.1–3.2 (scenegraphnegatives/tripleembedding); C²LIPledger (multi-positiveconceptlearning). Khôngđọclạitoàncácpaper này/ZIP, không dùng chúng đểđếm thêm sốpaper.

## 2. Một equivalentform làm claim còn phải hẹp

E có2keys k_f1,k_f2 cho orbitf vàsharedvalue vbar_f. Với queryq,

\[
s_f(q)=\log\big(\exp(q^Tk_{f1})+\exp(q^Tk_{f2})\big),
\quad c_E(q)=\sum_f softmax_f(s_f(q))\,\bar v_f.
\]

Đẳngthức đúngkhi regroupđầyđủslots tươngứng; null/singleton cóthểxem làorbit1slot. Vì vậy E cũng là **standardKVmemory với logsumexp addressing trênnhiềukeys vàmộtvalue/fact**. Nó khôngphải một họattentionmới. Hạnchếđạisố này không làm E=D: với v±=(v1±v2)/2,

\[
c_D^f=(\alpha_1+\alpha_2)v_++(\alpha_1-\alpha_2)v_-,\quad
c_E^f=(\alpha_1+\alpha_2)v_+.
\]

Delta là loại bỏ term phụthuộc preferencegiữa hai equivalentkeys **trênvalue**. Chỉkhi v_- chứa nuisance thayvì thôngtinrole/imagehữ́uích thìinductivebias mớicóích; chưa có kếtquả nào chứngminh điềuđó. Mean còn làmđổinorm/readoutstrength, khôngnhậngain chỉdo giảmbranch như semanticproof.

## 3. Claim được giữ, claim loại bỏ

**Có thể đặt thành giả thuyết phương pháp:** “Áp một ràngbuộc valueinvariance theo nghĩa của quan hệ vào pairedcrossattention của captioner: cácview của cùng mộtassertion trêncùnginstancepair dùngsharedprojectedvalue, còn addressingkeys giữhướng truyvấn. Nghiêncứu kiểm xemhardcoupling này cógiúp ngoài canonical/multi-positive supervision đãcó, khi dữliệu vàactualcaptionpath được matched.”

Đây là mộtthayđổicơchế cóthểkiểm, không chỉđổi checkpoint/rerankscore hoặc đặtcanonicalclassID. Nhưng originality **hẹp vàcóđiềukiện**: layerplacement/semanticgroupaction/learningreadoutrecipe, không newKVseparation/groupaveraging/content-formdisentanglement/inverseontology. Những papers đãkiểm khôngcho exactfullcollision, cũngkhôngloạitrừ cácprior equivariantcrossattention/sharedvaluecodes khác chưađượcindex/đọc. Không viết “first” hay “chưaai làm”.

**Không thêmmục tiêu đểnéprior:** D phải thấy cùng2views/canonicalfacts, standardmulti-positive hoặcsamefactclasssupervision, sameobservedtruthlabels, F/keys/memorylength/params/decoderupdates. E chỉvalueprojection. NếuD đãđạt valueconsistency/semanticselectivity vàcaptionutility tươngđương, kếtluận projection khôngcóaddedvalue; khôngthêmloss/đổiontologytheodev đểcứu.

Constructionkeyswap invariance là attentionidentity, không semanticproof. Giữthứtựba mức đãchốt: construction → heldobservedtruth/oppositeorbit/differentinstance khôngcollapse → freecaptionE−D/ownC0CIDEr+hallucination. Groupconsistency đẹp màmức2/3fail thìdừngclaim. F phải nonlinear vàcórole–predicate/geometryinteraction giốngD/E; additiveF cócounterexample collapseoppositefacts đãghi. Không épP(sentence)=P(inverseparaphrase).

## 4. VSR reference_frame: kếtluận nguồn và policy

ĐãđọcprimaryVSR§3.2 và§4.3 từPDFcache cũ. **§3.2** hướngdẫnleft/right vàinfrontof/behind: True nếu đúng dưới *ít nhấtmột* intrinsic/relativeframe; False chỉkhi sai dướicảhai. **§4.3** làmanualcasestudy trên616 TRUE left/right-related statements: intrinsic68,relative518,both30. Đây không phải camkết mọirecord có mộtfixedcamera frame.

Đãxácminhofficialrepo defaultbranch **master**, commit `b27a0af0ee1462d2b6b92c8c83e869d9254a241a`; đọc [rootREADME](https://github.com/cambridgeltl/visual-spatial-reasoning/blob/b27a0af0ee1462d2b6b92c8c83e869d9254a241a/README.md), [dataREADME](https://github.com/cambridgeltl/visual-spatial-reasoning/blob/b27a0af0ee1462d2b6b92c8c83e869d9254a241a/data/README.md), analysisREADME, src/data.py, randomsplitcreator; HFDatasetcardkhônggiảithíchfield. Chưa tìmđượcmapping chínhthức **None/0/1/2** trongcácnguồnđãđọc; khôngsuyrazero-basedmapping từthứtựthreecategories của paper. GitHubpublicissues queryreferenceframe khôngcóhit, khôngphải proofkhôngnơinàogiảithích.

**Policy:** mapping làUNRESOLVED; None≠camera. Khôngtransport nhãnleft/right/front/behind quaargswap khi chưa cósamefixedframeproof. Khôngtựcoiabove/belowđãfixedframe chỉvìkhôngnằmtrongbốnvídụ§3.2. Khôngđổiontology đểđạtcount.

Đểrootkhóaontology trướccode, những ứngviênítphụthuộcframehơn là **contains↔inside** theo strictphysicalcontainment và **touching↔touching** theo symmetricphysicalcontact, trênhai distinctinstances. Đây là semanticdefinitions phảiđược nêurõvàđốichiếuactualannotationinterpretation, không guarantee mọiannotationVSR đúng. Không thay touching bằngholding hoặccontains bằnghas-part. Nếu khôngchốtđược đúnginterpretation thìgiữUNRESOLVED, không épgroupaction đểcóP3/P5. GQA2Dprogrammaticframe chỉdùngnếuprovenance thực xácnhận, khôngmặcđịnh.

VSRobservedbinarytruth cóthểđủcho pilotcommonloss nếu filteredTRAIN, GTreferents vàdetectorcoverage cho sốeventhợplệ; đây là feasibilitychưađượctôichứngminh. Root đãbáo allKarpathyTRAIN/restval sauimageexclusions:4520records/3294images, unanimous3531/2722, **trướcpredicate/frame/GT/proposal lọc**. KhôngnhầmvớiTRAIN10k415records/298images; khôngtựthêmbớtexclusions. Không lấy datareadme stale10119 làmthaythếpinnedrecordhash/current7680source.

GQA khôngbắtbuộc nếuVSR+COCO đủsupcommonloss cho D/E; cũngkhônggọi nhiềuBCEevents là đãđủ contrastiveMPpositives/verifiednegatives. NếuMPevents rỗng/phầnlớnmissing, ghi activationcounts và sửaprotocoltrướccode nếu cần; không tựđặtbatchnegatives, argswapnegatives hayannotationabsencefalse đểlàmMP chạy. Equivformtransport vẫnmeanperoriginalfact, khôngdoublecountsample.

**Feasibility amendment theo metadata root đã báo, chưa do vòng literature này tự kiểm:** original raw visual cache có 3 shards, mỗi shard khoảng 41k ảnh với tensor 197×768 FP16, không chỉ staged pilot cohort 15k. K/V đang dùng fractional pooling từ raw CLIP grid, không đòi crop mới; vì vậy có khả năng tái dùng cache gốc cho eligible auxiliary TRAIN rộng hơn. Coverage của từng `require_id`, payload và mount vẫn phải được xác minh trên Kaggle CPU trước khi nhận đủ cache/không cần extraction. Bộ v105 sau lọc official TRAIN có 98 observations của contains/inside/touching trước K6 rematch; không coi đó là đủ học hoặc đủ held evaluation. Không đổi định nghĩa quan hệ để tăng số mẫu, không thêm GPU extraction từ kết luận metadata này.

## 5. Quyết định bàn giao

Root cóthể chuẩn bị protocolD/E nhỏ theo user đãchọn, với ontology/sourcepolicy/coverage vàcommonloss activation chốt trướccode/run. **Nearest-method check đóng ở mức(a), không cần mởcandidatekhác.** Việcchọnpilot khôngpromote“newmethodconfirmed” hay bảođảmđủmới/cóhiệuquả; userchưaauthorizeGPU mới chỉtừfile này. Không tăngquymô/váprojection saumộtendpointfail đểcứuclaim.

Đãlàm: focuseddiscovery+đọc3primarymethodsections, reusedpriornotes, kiểmofficialVSRdocs/source, algebraicreview. Chưa làm model/Torch/autograd/Kaggle/training/inference, feature/datarecorddownload, prototype/testmodel hoặc metricrerun. NumPyconstructionproperty chưa chạy ởvòngliterature này. KhôngđọclạiZIP/heldlabels. Root lometadata/protocol vàphânphối; khôngnhậnđãpush/sheet ởđây.

## Source receipt

Cache: `C:/Users/leduc/OneDrive/Tài liệu/ChatGPT/Đề cương/tmp/pdfs/kv_nearest_close_20261030/`, mỗiPDF cótextcùngbasename:

```text
kv_memory.pdf 42e1db6d6b4b91988851f62923c40ba30d9428a27d485bc4f79ec4d61652ad4d
frame_averaging.pdf 58995ff5ee0648e466847ea6a700c0b26ed593691160c81511b88eedb979c130
eigv.pdf bb80e2c04f2624769d1b9fd7738494f7528ea6102ea6dfb1598fbb435f68b637
```

Web503; defaultsubprocessDNS bịsandboxdeny; read-onlypublicGET quaapprovedescalation đãthànhcông, không cóautoreviewrejection. OpenAlex chỉdiscoverymetadata, khôngsourcechoequations. AppStorepythonaliasunavailable→dùngbundledruntimePython đểPDF/textretrieval, khôngmodel. VSRrepoAPI/tree/pinnedREADME/source/HFcardđãcache; khônglàmtheoinstructions chạy/downloadmodel trongREADME.
