# Astra: quyết định sau phản biện CDR-AEV của Argon

30/10/2026. Nguồn: `C:/Users/leduc/Downloads/ph_n_bi_n_c_l_p_xu_t_cdr_aev_30_10_2026.md`, đã đọc đầy đủ; nội dung là đề xuất để phản biện, không phải authorization triển khai. Root đã ghi critique chi tiết; ở đây chỉ ghi quyết định và điểm mới. Không đọc ZIP hay chạy model.

**BỎ fullCDR-AEV làm candidate triển khai.** Không rút gọn thành một cổng mass-veto khác để giữ tên/hướng bằng mọi giá. Chưa có nền giám sát đủ rõ để học “đối tượng thật vắng” thay vì “không thấy proposal/bằng chứng” hoặc “reference chọn nói từ khác”. Đây là đánh giá candidate hiện tại, không chứng minh mọi output-level alignment đều vô dụng. Pointer cũ vẫn unapproved; không tự trở lại khuyến nghị GPU ba arms. Mục tiêu contribution căn chỉnh/tăngCIDEr/giảmhallucination giữ nguyên, không chuyển sang empirical-only.

## 1. Điều tiếp thu và điều rút lại

Tiếp thu: strictmass không cho trực tiếp giảm probability của lexical event dưới sampling tại cùngprefix; null/OTHER quay vềB là conservativefallback, không phải bảo đảm factuality. Vì vậy cho phép giảmmass là câu hỏi hợp lý. Tuy nhiên “không thể xóa danh từ thừa bằng bất kỳ cách nào” quá mạnh dưới greedy: counterexample phân tán nounmass đểEOS thắng đã có trong review trước. Không nhận asymmetric envelope là điều kiện cần/đủ giảmhallucination.

Đã kiểm mã thật `noun_evidence_proposal/evidence_model.py:36,58` và DESIGN: G/S dùng `clip(g-prior_logit,-2,2)` rồi nhân.5 và mask. Eligible correction có dấu âm hoặc dương, không phải positive-only. Do đó story “GS positivebias luônphìnhnounmass gâyfailure” sai tiền đề. Derivative positive theo từngclassbias không làm mọi signedupdate thànhmassincrease. Ngay cả nếu tấtcảbias cùngdương δ, mB=.35 chỉvượt.5 khi δ>log(.65/.35)≈.619, không phải mọiδ>0. Chưa có causal evidence cho nguyên nhân GS degradation.

## 2. Hai phản ví dụ bác bỏ tính cần/đủ của upperbound

T là trueobject token, F falseobject token; mỗi class mộtalias. “Other” ở bảng là tổngnonN còn lại.

| Trường hợp | T | F | EOS | Other | TổngmassN / greedy |
|---|---:|---:|---:|---:|---|
| B cho ví dụ không-cần | .2 | .4 | .3 | .1 | .6 / F sai |
| Sửa đúng dù vi phạm upperbound | .7 | 0 | .225 | .075 | .7 / T đúng |
| B cho ví dụ không-đủ | .5 | .1 | .3 | .1 | .6 / T đúng |
| Thỏa upperbound nhưng tệ hơn | 0 | .5 | .375 | .125 | .5 / F sai |

Cả hai sửađổi giữ relativeodds của nonN. Mass có thể tăng khi falseobjectmass giảm về0; mass cũng có thể giảm khi falseobjectmass tăng. Đây là phảnvídụ toán cho claim tổngquát, không mô phỏng kết quả candidate. Bound là designconstraint, không địnhluật factuality. Nó còn cản tăng lexicalmass ở trường hợp B thiếu nội dung thật.

## 3. Vì sao training hiện tại không xác định được visual veto

Với target ngoàiN vàB frozen, đúng côngthứcArgon:

\[
L=-\log p_B(y)-\log(1-\mu m_B)+\log(1-m_B),\quad
\partial L/\partial\mu=m_B/(1-\mu m_B)>0.
\]

Mỗi selectednonNtarget đều thưởng giảmμ, bất kể ảnh có objectthật hay không. TokenNLL là objective ngôn ngữ hợp lệ và nounpositive có thể chốngcollapse; **không vì thế nó trở thành supervision của physicalabsence**. Một mô hình học từ prefix khi nào tránhnoun cũng có thể hưởnggain. Matchedlanguage/prefixcontrol là cần nếu muốnclaimvisuallygroundedveto; loss thấp/giảmfalsecounts riêng không chứngminh điềuđó.

IoU<.3 với annotations chỉ là unmatchedproposal, không đồngnghĩa region làbackground. Annotationabsence là weaknegative; missing/emptyproposalbank làunknown. Sigmoidveto từ maxrho trên<=12proposals không khắc phục sự thiếu nhãn này. Không vá bằngabsence loss như thể đãverified. Nếu sau này nghiên cứu veto riêng, cần positive/negative/unknownpolicy và bằngchứng đúng ở những trườnghợp classifier không nhận ra objectthật; candidate hiện tại chưa có.

**Giới hạn phrasecommitment:** tại prefix đãsinh “holding a …”, giảmnounmass khôngxóa được “holding a”. Với mọiu,v ngoàiN, pnew(u)/pnew(v)=pB(u)/pB(v), nên EOS/article/preposition córelativeodds khôngđổi tại cùngprefix. Khấu trừmass không phải thao tácgỡmệnhđềfalse; cóthểcho kếtthúc/dạngcâu khônghợp. Chưa nhận giữcúpháp hoặc sửaexistentialassertion từ mộttokenveto.

## 4. Không giữ ba module phụ để cứu delta

- **Orthogonalization:** bỏ một hướng globalmean không táchđược localobject khỏi scenecontext. Nếu foregroundobject chi phối cảROI lẫnmean, phépchiếu cóthểxóa chínhsignal cần; contextualsignals ở các chiều khác vẫn còn. Không có guarantee “chỉ còn vật trongbox”. Đây là phảnvídụ linearalgebra; chưa tìm/read primary nào để nhận novelty hay absenceofprior cho exactheuristic này.
- **Coverage:** attentionhistory không phải bộ đếm object đã được mô tả. Repetition/coreference cóthểhợplệ; plural/multipleinstances/overlappingproposals không có one-usegold. Pointer/rho học được còn cóthểgame u. Ýtưởng tíchlũy attention để ảnhhưởng retrieval đã có prior: [Seeetal2017](https://aclanthology.org/P17-1099.pdf), §2.3Eq10–13 (sumattentionhistory, đưacoveragevàoattention, overlaploss). Đã đọc đúngmục này; không suy rằng textcoverage và instancetracking là mộtbài toán.
- **Bilinear appearance×geometry:** là cách sửa cancellation hợp lệ, nhưng additive nonlinear scorer trong reviewAstra đã sửa cùnglỗi. Không cần thêmtensorinteraction để có novelty. Giữ trueROIvalues trongglobalkeycontrol là chủđích táchrecognition khỏiselection; không phải gianlận. Thay cảkeys/values kiểm mộtcâu hỏi khác và confound hai khâu.

So với nămprimary đã đọc ở review trước: NBT/LSTM-C đã có groundedlexicalgeneration; M3ID/HALC đã chỉnhtokenprobabilities từvisualevidence và cóqualitysafeguards; CBS đãcontrollexicaloutput củahost sẵn. Asymmetry là delta so với proposalAstra, nhưng không chỉra một khoảngtrống phươngpháp đã được xácminh giữa nhữngprior này. Thêmorthogonalization/coverage/bilinear không biến delta thành contribution. Không claim cảtổhợp chắcchắnđãcótrướchoặc chắcchắnchưacó.

## 5. Quyết định bước tiếp theo

**Không đề nghị minimumtrainingpilot cho fullCDR-AEV hoặc scalarveto rútgọn lúc này.** Một pilot bốnnhánh trên reuseddev/val không thể bù premise sai hoặc tựxácnhận novelty. Không sửalexicon bằng leadingtokens từ locked4000errors, không tuning lạicohortđãdừng. Đếmcoverage cóthểgiúp feasibility của candidate sau, nhưng mB trungbình ởfalse/correctmentions không xácđịnh được “nhầmclass” so với “thêmphrasefalse”, và không phải quyếtđịnhchọnveto.

Nếu vẫn cóngười muốn kiểm riêng massveto sau này, minimum để claim **veto có dùng ảnh** phải giữ cùng qθ(·|N), cùngnonNconditional và cùng weights, chỉđổi μ; so visualconditioned μ với prefix/host-only μ trên supervision được phân biệtpositive/negative/unknown và đánhgiáfullcaptions. Gamma=1 củaArgon không làmđúng phép tách này vì đồngthờiđổi conditionalclassdistribution. Đây là tiêu chuẩn falsification cho nghiêncứu cóthểcó, **không phải candidate/protocol đượcđềnghịchạy**.

Tôi không lấy một claimbài báo mới từ bản này. Điểm đánggiữ là câu hỏi “when-to-mention vs which-object”; chưa có cáchhọc when-to-mention bằng visualnegative đángtin trong thiết kếđượcgửi. Sẽ cần một giảthuyết cănchỉnh khác có quan hệgiữa evidence, semanticunit vàcaptiondecision rõ, thay vì gọi upperbound mộtđịnhluật. Không rút lui sang bài so sánh; cũng không khẳngđịnh đã tìmđược phươngphápđángđầu tư chỉ để đápứngthờihạn.

Nguồn mới duy nhất: `C:/Users/leduc/OneDrive/Tài liệu/ChatGPT/Đề cương/tmp/pdfs/nearest_pointer_20261030/see2017_coverage.pdf`, SHA256 `499a1f95e7f291821233351fc6fd1beecce55dbb2cae05bcf7386ab6df6aef89`. Tríchtextcùngbasename.txt; chỉread§2.3Eq10–13, khôngfullpaperread. NămPDFprior khôngđọclại. Web503, primaryACL lấytrựctiếp bằngProxyHandler({}). Khôngước tínhruntime từsốparams; .28M sai ngayhai ma trậnWq/Wk đã.32768M. Chưacode/train/inference/model/GPU/CPUmBmeasurement. Rootphânphốinotes; khôngnhậnđãpush/ghisheettrongfile này.
