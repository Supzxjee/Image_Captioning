# Hướng chọn: dùng grounding có chung referent để chọn caption hoàn chỉnh

30/10/2026. **Astra chọn hướng này để phát triển và để root/user phản biện, chưa duyệt triển khai.** Không quay về pointer/veto. Claim là một cải tiến có kế thừa cho đường quyết định caption, không phát minh graph matching/MAP. Nearest-method/readscope/hash: [reading notes](SHARED_REFERENT_READING_2026-10-30.md).

## 1. Ba việc và ba hướng đã cân nhắc

Tìm vùng phù hợp chỉ cung cấp ứng viên bằng chứng. Hiểu ngữ nghĩa còn phải biết vật nào mang thuộc tính/tham gia quan hệ nào. Để caption dùng hiểu biết ấy, cần một đường quyết định thực: ở hướng chọn, grounding trực tiếp chọn **cả caption**, trước khi trả câu cho người dùng. Không trông chờ auxiliary loss hoặc heatmap tự chuyển thành chất lượng sinh.

| Hướng thực chất | Semantic unit / evidence / đường tới caption | Supervision và unknown | Prior, delta có thể kiểm | Điểm yếu / pilot tối thiểu / vì sao chưa bị refute |
|---|---|---|---|---|
| 1. Học lại biểu diễn thị giác cục bộ, coadapt decoder | Grounded noun phrase↔dense region tokens từ DINO+CLIP; decoder đọc tokens đã được học và cùng cập nhật | COCO/VG positive box–phrase; không lấy thiếu annotation làm âm | COMM đã fuse multi-level CLIP/DINO; delta phải là grounded phrase learning trên host nhỏ, không chỉ nối hai backbone | Tốn feature/model adaptation, dễ chỉ là backbone upgrade; so cùng coadapt/data với/không local alignment. H/L frozen CE không bác hết hướng này, nhưng chưa có delta đủ thuyết phục nên **không chọn** |
| 2. Tối ưu caption tự do bằng reward ngữ nghĩa có ảnh | Claim trong rollout↔image/pair evidence; reward đi qua SCST/PPO thật | VSR có observed true/false cho một assertion; mọi claim khác unknown | MOCHa dùng reference-NLI+BERTScore+KL; delta là image-grounded claim reward thay proxy chỉ text | Cần verifier đáng tin trước; reward dễ bị né predicate/đổi câu generic. So cùng reward nhưng bỏ image/geometry signal, giữ adequacy. C/G và I/J không bác mọi reward, nhưng lịch sử cho thấy không nên bắt đầu một vòng RL khi verifier chưa vững. **Không chọn** |
| **3. Chọn caption bằng grounding có shared referent** | Cụm/nhóm mệnh đề↔một assignment các vùng có thứ tự, dùng chung node cho referent thật sự được share; chọn caption từ cùng C0 pool | Positive grounded nodes/edges của GQA, observed binary VSR; unobserved edges/proposal misses unknown | Johnson2015 đã joint MAP; MAttNet đã bound referent+context; PICL đã listener decoding. Delta ở đây là **adaptation structured grounded decision cho native C0 candidates**, có typed local evidence và explicit referent consistency qua nhiều claims; không phải một factorization mới | Parse sai, ít caption nhiều claims, scorer/geometry priors hoặc candidate pool không có câu tốt. Fit một verifier, so BIND/IND cùng weights/pool plus C0. Old CIDEr-ranker thiếu truth/grounded labels; J pooled hidden contexts khác chọn một câu có shared witness. **Chọn**, với mức originality hẹp được nói rõ |

Tôi ưu tiên hướng3 vì có một giả thuyết ngắn và một phép đối chứng trực tiếp; không vì đã chứng minh nó sẽ thắng. Khác late veto: chọn một câu khác có thể thay cả phần “holding a…”, không cố xóa một danh từ sau khi đã commit prefix.

## 2. Hypothesis và ví dụ tách khỏi bài toán chỉ tìm đúng class

**Hypothesis:** chấm riêng từng mệnh đề có thể chấp nhận các bằng chứng thuộc những instance khác nhau. Nếu caption nói cùng một referent ở nhiều mệnh đề, buộc tất cả dùng một assignment vùng chung sẽ giảm lựa chọn câu ghép sai mà vẫn giữ các câu đúng do native captioner sinh.

Toy score table dưới đây là **phản ví dụ cấu trúc, chưa phải ảnh hoặc kết quả model đã kiểm**. Một ảnh giả định có dog1 dưới table nhưng xa chair; dog2 cạnh chair nhưng không dưới table. Caption “A dog is under a table and next to a chair” đòi **tồn tại một dog** thỏa cả hai. Không cần biết trước nó là dog1 hay dog2.

| Compatibility | dog1 | dog2 |
|---|---:|---:|
| dog unary | .99 | .99 |
| under(dog,table) | .95 | .05 |
| next_to(dog,chair) | .05 | .95 |

Table/chair unary đều.99. IND chọn tốt nhất từng cạnh, average5terms=(3×.99+2×.95)/5=.974. BIND buộc cùng dog, tốt nhất=(3×.99+.95+.05)/5=.794. **Cả hai cạnh IND đã có thể lọc đúng subject/object class**; khác biệt còn lại là share instance qua hai cạnh. Một caption nói rõ hai dogs khác nhau không chịu cùng constraint. Gap chỉ biểu thị inconsistency theo scorer, không chứng minh ảnh thật sai.

Nguồn có thể kiểm binding thật: GQA cung cấp object IDs/boxes và positive edges để xác nhận cùng annotated referent qua nhiều claims. VSR cố ý loại multi-instance, nên không dùng nó làm bằng chứng đã kiểm ambiguous binding. Ảnh minh họa thật và độ phủ multi-claim của C0 là missing facts cần kiểm trong pilot được chọn; không tự coi toy là verified image/gold. GQA thiếu edge không được suy thành false conjunction.

## 3. Module và actual inference

**Generator:** C0 frozen đúng checkpoint5d92cd5…; A vẫn reference. Dùng native vocab/max length/preprocessing. Pool tối đa6 câu = greedyC0 + beam5C0, dedup theo rule cố định; mọi policy thấy **cùng strings và C0 scores**. Không dùng oracle references hoặc GT lúc chọn. Whole-caption selection đầu tiên đơn giản hơn incremental phrase search và không đổi candidate support giữa arms.

**Evidence:** tối đa6predicted proposals/ảnh từ một source pinned, không GT box ở forward. Đề nghị encode riêng ROI crops và union crops bằng frozen CLIP ViT-B/16 cùng version/preprocess đã pin, không coi mean của global-context patch là crop độc lập. Có≤6ROI+15uniqueunion crops/ảnh; union appearance dùng chung hai hướng, geometry phân biệt thứ tự. Đây là feature extraction mới phải được user duyệt, chưa chạy; không hứa runtime/quota. Tất cả controls dùng cùng crops. Thất bại H/L không chứng minh crop giúp; ở đây không nhận crop gain vào công shared binding.

**Một verifier, hai loại factors:**

- Appearance ROI head cho class compatibility a(c,i), lớp ngoài mapping là OTHER chỉ khi GT thực sự cho class ngoài mapping; unmatched proposals không tự là background. Pilot dùng COCO-class noun aliases rõ nghĩa để giảm phạm vi, nhưng parser được đọc **cả câu đã hoàn thành**, nên compounds nhận diện từ cả cụm, không suy leading token tương lai.
- Ordered pair head r(p,c_s,c_o,i,j), từ ROI_i/ROI_j/union appearance và signed geometry; candidate class/predicate là query hợp lệ, không phải detector class labels. Cùng query lexical có thể đúng/sai theo ảnh; tên lớp không phải nhãn truth. Đây là classifier compatibility, không phải GRU assertion likelihood. Relation vocabulary nhỏ lấy từ explicit TRAIN ontology/mapping giao nhau GQA–VSR, khóa trước evaluation; không tự đồng nhất mọi synonym/verb với một spatial relation.

Đưa output factors về [-1,1] bằng2softmaxprob−1 /2sigmoid−1. Chúng là **bounded compatibilities**, chưa gọi là calibrated physical-truth probabilities. Không thêm GNN, OT solver, coverage, scalar veto hay probability-envelope.

**Semantic parse:** dependency graph tối đa3object nodes và các supported spatial edges. Chỉ share node khi explicit syntax cho cùng antecedent/subject, chẳng hạn conjunction dùng chung subject hoặc relative clause có antecedent rõ. Không merge theo class/name; “another”, plural, unresolved pronouns và ambiguous PP attachments ngoài grammar được đánh dấu unsupported/fallback, không “incorrect”. Duplicate assertions dedup; không ép mỗi ROI chỉ dùng một lần. Hai NP khác nhau không tự là cùng instance.

Nếu graph không parse được hoặc không có evidence bank, grounded score0, giữ base score; ghi coverage. Một unknown predicate không nhận nhãn false; partial graph có thể dùng các supported factors và phải báo phần bỏ qua. Low scorer support vẫn có thể gây lựa chọn sai khi detector bỏ sót; đây là risk của soft score, không certificate absence. Không GT substitution.

## 4. Score hoàn chỉnh và matched comparison

Với F(G) là số supported unary+edge factors, a/r đã bounded, đặt

\[
G_{bind}(I,G)=\max_z\frac{\sum_v a(c_v,z_v)+\sum_{(u,p,v)}r(p,c_u,c_v,z_u,z_v)}{F(G)},
\]

z gán ROI cho mỗi **semantic node**, mọi xuất hiện của node dùng cùng z. Spatial pairs là ordered/nonself khi semantic relation đòi hai entities. Với≤3nodes,K≤6, có≤216assignments nên có thể enumerate, không cần graph solver mới.

\[
G_{ind}(I,G)=\frac{\sum_v\max_i a(c_v,i)+\sum_{(u,p,v)}\max_{i,j}r(p,c_u,c_v,i,j)}{F(G)}.
\]

IND giữ nguyên typed pair head, nouns/classes, query, appearance, geometry, data, weights; chỉ bỏ equality constraint giữa những occurrences của cùng semantic node. Vì feasible set, Gbind≤Gind. Đây là đại số kế thừa, không theorem novelty. Nếu conditional edge heads đã khiến IND có cùng witness thì không còn headroom và phải dừng claim.

Chọn câu bằng

\[
S_X(C,I)=|C|^{-1}\log p_{C0}(C\mid I)+.1G_X(I,parse(C)),\quad X\in\{ind,bind\}.
\]

Mean logprob gồmEOS, giữ cùng length convention; hệ số.1 là đề nghị khóa trước pilot, không sweep theo dev. Ground score average không tự ưu tiên rawsum nhiềuclaims hay phạt rawlogprob câu dài; **không bảo đảm hết graph-size/extreme-value bias**. Báo phân phối node/edge/length, candidate changes và missing-factor rates. Có thể viết Sbind=Sind−.1(Gind−Gbind), tức chỉ khác coherence penalty dùng cùng evidence.

Thêm pool-base policy chỉ dùng C0 score để phân biệt beam/search gain; raw C0greedy là ownbase thực dụng. Không nhận mọi BIND−greedy là alignment gain. Không reweight scores sau xem endpoint.

## 5. Loss, supervision thực có và split

Fit verifier **một lần**, không train một bộ head tốt hơn riêng cho BIND. Generator không cập nhật. Objective ban đầu:

\[
L=L_{ROI-CE}+.5L_{rel,VSR-BCE}+.1L_{rel,GQA-positive}.
\]

Các hệ số là proposal để prelock, không optimized. ROI-CE chỉ GT-matched predicted proposals; relation losses chỉ đúng cặp được match và predicate có observation. GQA-positive là-logsigmoid r cho annotated positive edges; không đặt các predicates/pairs còn lại bằng0. VSR dùng observed binary truth trên referents unique được nguồn xác định và COCO boxes match; không biến whole-image truth thành box-pair truth khi mapping ambiguous/missing. Mask event đó và báo denominator. Same statement true/false pairs chỉ coi là paired nếu release/manifest thực sự có cả hai; không tự tạo bằng relation swap hoặc mismatched images.

Nguồn mới: GQA normalized scene graphs có objectIDs/boxes/relations, nguồn gốcVG; **chưa stage/map/cache**. Chỉ GQA officialTRAIN, map qua COCO IDs, giao Karpathy train/restval; không fuzzyfilename guesses. VSR officialTRAIN/nhãn đã có có thể bổ sung source mới nếu lọc đúng;7,680 là số instances của original randomTRAIN, không số ảnh/nội dung chắc chắn đủ điều kiện. Regroup/exclude theo imageID và loại mọi image của officialVSRdev/test, oldVSRheld128, Karpathy val/test và fixeddev. Nếu dùng existing256TRAINpairs phải ghi đúng phạm vi, không nâng thànhfullVSR.

Khóa tối đa4,000unique eligible TRAINimages theo hash, chừa một image-grouped TRAINholdout cho grounding checks trước fit. Số retained sau ontology/split/proposal matching chưa biết; không coi tất cả GQA đã humanverified hoàn hảo. COCO2017/2014/VG identity mapping cần exact IDs và original dimensions. Không đưa old32semantic panel/mixedAI labels vào TRAIN hoặc gọi là independent confirmation. Không mở/tuning expanded4000 đãdừng.

## 6. Minimum experiment để quyết định đầu tư

Đề nghị một pilot có **một lầnfit verifier** (500updates,batch8,seed42, endcheckpoint; budget chưa duyệt) và bốn inference policies trên cùngfixeddev1000 đã dùng: rawC0greedy; pool-base; IND; BIND. A là reference không trainarm. Fit/eval feature extraction là phần của pilot khi được chọn, không tự mở auditphase/model/extraction bây giờ.

Không mở một chuỗi pretrain diagnostics rồi hy vọng sau đó cócaption: endpoint đầu tiên gồmcaptionselection/CIDEr/fullCHAIR80. Own C0 beam oracle chỉ tính chẩn đoán từ đúng pool mới khi làm pilot, không đưa vào inference; oldoracle1.513905 của checkpoint khác không chứng minh headroomC0. Nếu pool không có câu cải thiện có ích, không thêmbeam/rerun để cứu endpoint.

**Một falsifier binding tối thiểu:** trên GQA adaptation-heldout positive subgraphs có same-class competing instances và một shared node qua≥2supported edges, giữ unaries/ROI bank/geometry và maxima của từng edge table, nhưng hoán vị độc lập các hàng subject của **một** edge table giữa same-class ROIs. IND invariant vì per-edge maxima giữ nguyên; BIND mất correspondence dùng chung. Đo score/rank của known positive graph và caption-selection changes. GT chỉ chọn/đánhgiácanthiệp, không forward thường. Đây là representation intervention, không physical counterfactual hoặc truthproof; nếu head không phân biệt đúng witness ban đầu thì test uninformative. Báo toàn mẫu đủ điều kiện bằng rule khóa trước, không chọn chỉ cases BIND thắng. Nếu không có eligible multi-claim graphs, không gọi đã kiểm mechanism.

Các checks nhẹ: joint permutation của tất cả records invariant trong tolerance/argmaxmargin; role reversal không mặc định false nếu nhãn không có; no-image/name-only discrimination phải được nhìn qua same-text VSR pairing/held strata. Geometry là evidence hợp lệ cho spatial relation; không claim visual interaction appearance vượt geometry nếu chưa có matched test riêng. Không mở thêmtrainarm chỉ để có nhiềuablation.

**GO sang xác nhận** chỉ khi BIND có tín hiệu tăngCIDEr và giảmfalsecounts vsownC0, recall khônggiảmquá.005abs, có added utility vs IND/pool-base, và held binding check đúngchiều. PairedCIs/report exploratory nếu rộng; pilot không phải accepted-method certificate. BIND>IND ởsemanticbinding/CIDEr nhưngCHAIR khônghơnIND vẫn cóthể tách vai tròcoupling vàrecognition, nhưng dualgoal tổngmethod soC0 vẫn bắtbuộc. Không lấy CHAIR làm relation-truth metric. Length/graphcoverage cho biết có néunknown/genericcaption haykhông.

NO-GO nếu không hơnownbase hai mục tiêu, IND đãđủ, parser/candidate coverage quáthấp để có materialutility, hoặc coherence penalty chỉ làmcaptionngắn/đổi câu đúng thànhsai. Không tăngλ/steps/beam đểcứu. Những cutoffmateriality cụthể phải chốt với user trướcrun, không bịa150changedcaptions là thốngkêbắtbuộc. Không dựbáoquota/runtime hoặc đủthínghiệmtrước31/10.

## 7. Claim và khuyến nghị riêng

**Đóng góp dự kiến vừa đủ, có điều kiện:** “Một cách thích nghi grounded caption selection cho captioner đã huấn luyện: học các compatibility đối tượng–quan hệ từ positive grounding và nhãn quan hệ được quan sát, rồi buộc các mệnh đề cùng referent sử dụng chung assignment vùng khi chọn câu. So với chấm từng mệnh đề độc lập trên cùng candidate pool và cùng verifier, nghiên cứu kiểm tra lợi ích của tính nhất quán referent đối với caption quality, object hallucination và grounding.”

Kế thừa: scene graphs, unary/binary potentials/MAP (đặc biệt Johnson2015Eq2 rất gần), modulargrounding, region/classification/BCE, CLIP crops, beam/listener selection. **Không nhận công thức mới, graphbinding mới, first structuredcaptionranking hay causalproof.** CORA cũng đã dùng captiongraph và entity-based reranking; không được nói mọiprior chỉ biếtbagofwords. Phần có thể đóng góp là task-specific adaptation/learning-dataflow và ích lợi giữreferent consistency trongactualcaptionselection trên strongC0, được tách khỏi thêmdata/features/search bằng matchedcomparison. Nếu chỉ mang scorerJohnson/CORA sang một checkpoint mà không có addedutility/giải thích phạm vi, chưa đủ đểnhận cải tiếnphươngpháp.

Đây là originality hẹp, không một hướng có novelty toàncầumạnhđãxácnhận. Tôi vẫn chọn nó trước haihướngcòn lại vì có một thayđổi quyếtđịnh rõ, khôngnhậnabsencelabels giả, sửa được cảcaption commitment, và kiểm được bằng mộtmatchedcomparison. User có thể cânnhắc mức contribution này cho bài trong nước; không hứachấpnhận. Chưa có performance result đểkhẳngđịnh đángscale. Nếu user đòi một cơ chế nền chưa có trongprior, vòng đọc này **chưa tìm được** cơ chế nhưvậy.

Missingfacts khi chuẩn bị protocol sau userchọn: C0samepoolheadroom; parser/sharedclaimcoverage; GQA–COCO/VSRsplit/ontology/boxcoverage và sourceavailability; frozenCLIP/checkpoint/hash/disk/quota; actual verifier relation semantics. Không tìmsourcecounts trênlockedvalerrors đểđổilexicon. Chưa code/neural/GPU/dataextraction; đã đọc primary methods và ghi hypothesis đểphảnbiện.

## 8. Root review trước khi trình người dùng

Root đã đọc proposal và reading notes, không tải/đọc lại toàn bộ các PDF. Có thể trình đây là một **ứng viên cải tiến có kế thừa**, với đường quyết định và đối chứng cụ thể hơn CDR-AEV; chưa nhận cơ chế căn chỉnh mới đã được xác lập. Học compatibility từ nguồn nhãn mới và đưa known graph assignment vào caption pool không tự đủ làm novelty. Added utility là bằng chứng về hiệu quả; novelty còn cần delta phương pháp cụ thể so nearest prior và tiêu chí của venue.

Ba rủi ro quyết định còn mở:

1. C0 có thể ít câu nhiều mệnh đề dùng chung referent; khi đó BIND gần IND và contribution chỉ có phạm vi hẹp. Không thay thế bằng ví dụ toy hoặc old-checkpoint oracle. Chỉ kiểm thông tin này trong phạm vi dữ liệu/pilot sau khi user chọn, chưa tự mở extraction.
2. GQA positive multi-instance grounding và VSR observed binary truth giữ vai trò khác nhau. Chưa có nhãn false cho toàn bộ các conjunction nhiều instances; giữ unknown policy, không quảng cáo held positive-graph intervention là đánh giá đầy đủ đúng/sai của free captions. CHAIR không đo hết semantic binding.
3. Crop/union features là chi phí mới, không phải chỉ dùng lại cache UPT/CLIP cũ. Upper bound21crops/image là số lần encode cần lập kế hoạch, không phải runtime đã đo. Pipeline thêm crop features không được coi là lợi ích riêng của shared assignment; matched IND đã giữ cùng features để tách việc đó.

Khuyến nghị trình user đúng mức: đây là một hướng adaptation có giả thuyết rõ để cân nhắc, không lời hứa tìm được công thức mới hoặc đạt chất lượng tốt. Chưa có quyền code/GPU/budget. Không đổi metric goal hoặc tái tuning expanded4000.
