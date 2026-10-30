# Phản biện lại region–lexical candidate trước triển khai

30/10/2026. **Quyết định của Astra: SỬA ứng viên; chưa qua cổng chốt phương pháp/code/GPU.** Không giữ luận điểm bảo toàn lexical mass là đóng góp chính. Không kết luận toàn bộ ứng viên đã bị công bố giống hệt; cũng chưa có bằng chứng đủ để nhận một phương pháp mới. Mục tiêu contribution căn chỉnh, tăng CIDEr và giảm hallucination giữ nguyên; không đổi sang bài empirical-only.

## 1. Nguồn và độ sâu đọc thực tế

Áp dụng nature-academic-search/multi-source-search, sau đó PDF skill để kiểm ký hiệu. Web search/open lỗi503; urllib mặc định bị proxy refuse và IAB navigation timeout. Đã lấy trực tiếp năm primaryPDF bằng urllib ProxyHandler({}), không đổi thiết lập hệ thống, không chạy model. PDF được trích text bằng PyMuPDF. Hai trang công thức NBT/LSTM-C đã render và xem vì text extraction làm mất dấu bổ sung tập hợp. Đây là focused nearest-method comparison, không exhaustive novelty search.

Cache local ở `C:/Users/leduc/OneDrive/Tài liệu/ChatGPT/Đề cương/tmp/pdfs/nearest_pointer_20261030/`, ngoài repo nghiên cứu. Tên `dcc.pdf` trong cache thực ra là **LSTM-C**, không phải Deep Compositional Captioning/DCC; dùng title và hash để phân biệt. Các PDF/text là cache nguồn, không phải output thực nghiệm.

| Primary method | Phạm vi đã đọc trong vòng này | Cơ chế có liên quan |
|---|---|---|
| [Neural Baby Talk, CVPR2018](https://arxiv.org/pdf/1803.09845) | §3.1–3.4; Eq4–13 và đoạn visual-word extraction; xem trangPDF5 để kiểm Eq13/regionfeatures | Content/state-based pointer, sentinel/text branch, lexicalization của slot vùng, supervision trên nhiều positive proposals. Region features gồm appearance, vị trí và embedding nhãn detector. |
| [Incorporating Copying Mechanism in Image Captioning, CVPR2017 — LSTM-C](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yao_Incorporating_Copying_Mechanism_CVPR_2017_paper.pdf) | §3.1–3.4, Eq6–10; xem trangPDF5 kiểm Eq8 | Điểm copy phụ thuộc hidden state và object-classification score; trộn generate/copy, normalization chung trên vocabulary hợp. Đây không phải DCC; DCC là baseline khác. |
| [Guided Open Vocabulary Image Captioning with Constrained Beam Search](https://arxiv.org/pdf/1612.00576) | §3.1–3.3; Eq1–2, Eq11–12; implementation của FSM/constraint | Ràng buộc tập từ hoặc chuỗi phải xuất hiện thông qua nhiều beams theo trạng thái FSM; có thể dùng captioner sẵn có và image tags. Không học prefix→ROI correspondence. |
| [Multi-Modal Hallucination Control by Visual Information Grounding / M3ID, CVPR2024](https://arxiv.org/pdf/2403.14003) | §4.1 Eq1–4, đoạn contextual pressure và đầu§4.2 | Sửa log-prob bằng chênh có/không ảnh, hệ số theo thời gian và confidence threshold. Đã nhận diện rủi ro can thiệp cả từ hiển nhiên/từ chức năng; không học ROI→word pointer. |
| [HALC, ICML2024](https://arxiv.org/pdf/2403.00425) | §4.1–4.4; Eq2–4 và đoạn matching-based beam search | Nhận dạng token cần xử lý, grounding bằng detector, lấy nhiều FOV, chọn bằng JSD và contrast logits hai chiều; global image–text matching chọn beam. Đã có local visual evidence tác động trực tiếp token probabilities. |

Không nhận đã đọc toàn bộ experiments/appendices hoặc chứng minh các bài không có bất kỳ biến thể liên quan nào. Không cần thêm abstract papers để đủ số năm.

SHA256 của primaryPDF đã đọc:

```text
nbt.pdf  2e7fe67918f5051703426a3dd36f4d8ba75986527d7c4eae13075ade652ce268
dcc.pdf  ab9a212d80720ea552f33ab9265fb7fe7cfb6ad673931ee3d071a0df44227851
cbs.pdf  aea9f81cf2d2680619b896498a593ba8a0920c30c91db0c3c50d86b537765589
m3id.pdf fbaf559cd744ae3044f94c1b676f53fa01f6721e946cb21c7b3788f9a18cf687
halc.pdf e4017cdd023af213c233ddf0c5553de56a313c72bdc2176eee7157b35c67bfac
```

## 2. Delta thực sự nhỏ đến đâu?

| Cơ chế họ đã có | Thay đổi cụ thể của candidate | Vấn đề muốn giải quyết — hiện là giả thuyết | Phản ví dụ / phép bác bỏ |
|---|---|---|---|
| NBT: prefix→ROI→lexical output, sentinel, nhiều vùng dương tính | Giữ strong hostC0 frozen; học ROI lexical distribution không đưa nhãn detector vào key/value; giữ host surface distribution trong mỗi class | Tận dụng geometry của predicted proposal mà không bắt semantic class phải trùng nhãn detector; tránh học lại toàn bộ ngôn ngữ từ ít dữ liệu | Detector-label copy với cùng host/interface đã tốt bằng; rho học scene co-occurrence hoặc không sửa được các proposal có nhãn sai |
| NBT Eq13: trung bình probability của các target regions | Candidate dùng sum positive-set mass; match IoU lấy GTclass, không lọc theo detectorclass | Không ép caption noun vào một instance bịa đặt; giữ proposal đúng hình học nhưng sai nhãn detector làm positive | Setloss tự nó không mới: với m targetregions cố định mỗi event, -log[(1/m)sum pi] và -log[sum pi] chỉ khác log m, cùng gradient |
| LSTM-C: generation/copy scores rồi chuẩn hóa chung | Redistribute class identity bên trong một lexical partition; giữ pB ngoài partition | Giảm sự can thiệp trực tiếp lên xác suất từ chức năng/EOS của strong host, khi chỉ muốn sửa object identity | Unrestricted copy tốt ngang/hơn; mass-preservation cản sửa một noun đáng lẽ phải bỏ; EOS vẫn thắng do noun mass bị phân tán |
| CBS: điều khiển nội dung của captioner sẵn có bằng lexical constraints | Learned, soft per-prefix ROI retrieval; không buộc tag phải xuất hiện hoặc mở vocabulary | Cho ảnh quyết định lựa chọn class theo ngữ cảnh thay vì bắt chèn một tag | Retrieved classes chỉ phản ánh image-level tag prior; matched global-key control tốt bằng |
| M3ID/HALC: visual evidence trực tiếp sửa quyết định sinh, có cơ chế giữ chất lượng ngôn ngữ | Một lần đọc ROI từ cache, không chạy lại nhiều crop/image ablations; invariants ở partition thay vì contrast toàn vocabulary | Thích hợp host/caches hiện có với chi phí inference nhỏ hơn nhiều lượt encoder | Cached CLIP ROI không có đủ local semantics; ưu thế chi phí không bù được chất lượng; không được nhận “first direct probability grounding” |

Như vậy **pointer, null/sentinel, ROI lexicalization, positive-set grounding và direct decoder use đều là phần kế thừa**, không chỉ vài primitive xa. C0 frozen và normalization chưa tự động thành contribution. Bỏ detector labels là một lựa chọn có ý nghĩa, nhưng appearance classifier cũng là thành phần chuẩn, không được gọi riêng nó là phát minh.

Ba việc vẫn phải tách: proposal/IoU trả lời có vùng phù hợp; rho trả lời vùng mang ý nghĩa class nào; pi và actual word distribution trả lời vùng ấy có được dùng để sinh từ. Thành công một việc không chứng minh hai việc kia. Chỉ chọn object-level cho candidate này vì sourceGT/synonym/output partition hiện có hỗ trợ nó; không suy rằng quan hệ đã bão hòa hoặc object là toàn bộ ngữ nghĩa. Không nhận relation/attribute/instance grounding.

## 3. Bảo toàn mass: safeguard có giới hạn

Với w thuộc class c, công thức cũ rút gọn thành:

\[
p_M(w)=p_B(w)\frac{q_c}{b_c}.
\]

Do đó nó tương đương một class-specific logit correction log(q_c/b_c) trong N và zero correction ngoài N, với normalizer tổng bằng1 theo construction. Đây là **một dạng probability correction bị ràng buộc**, không là cơ chế căn chỉnh tự thân. Pi/rho mới là nơi học correspondence. Nó khác G/S ở local prefix-conditioned q và inference retrieval; không khác chỉ vì gọi là probability interface thay logit bias.

Ràng buộc thật: tại cùng prefix, p ngoài N và tổngmassN giữ nguyên. Nó ngăn giảm trực tiếp totalmassN hoặc thay trực tiếp xác suất từ ngoài N. **Không ngăn EOS-evasion/shortening**, kể cả ngay prefix hiện tại dưới greedy:

```text
pB: noun1=.4, noun2=.1, noun3=.1, EOS=.3, remainingnonN=.1
mN=.6; basegreedy chọn noun1.
q=(1/3,1/3,1/3) => pM nouns=(.2,.2,.2), EOS=.3.
M chọn EOS dù tất cả invariants đều đúng.
```

Đây là phản ví dụ đại số, không kết quả model. N còn là lexical partition chứ không POS hoàn hảo. Mass giữa các bước cũng thay theo prefix mới. Không bảo đảm grammar, object recall hoặc caption length.

Nhu cầu có thể hợp lý của safeguard: copy evidence cạnh tranh toàn vocabulary làm đổi cả probabilities không liên quan đến object identity. LSTM-C Eq8 dùng normalization chung; NBT Eq9 phân chia region/text probability bằng sentinel. Đó là lựa chọn thiết kế hợp lý cho bài của họ, không lỗi đã chứng minh. Trên C0 mạnh, **chưa có dữ liệu xác định coupling đó là nguyên nhân suy thoái G/S/I/J**, và exact invariants chưa được chứng minh giúp caption. Không lấy thất bại các nhánh trước làm bằng chứng nhân quả cho safeguard.

## 4. Sửa thiết kế đối chứng, không thêm heuristics để cứu claim

Với dot-product/linear key cũ, q^T W_a e_bar là số chung cho mọi ROI, triệt tiêu trong tỷ lệ pi_i/pi_j. Nó có thể còn đổi tổng ROI-vs-null nếu null không được cộng cùng số. Vì thế G-global cũ không có global-appearance-conditioned *relative ROI selection* theo cách mô tả trước; không nói nó triệt tiêu toàn bộ nhánh kể cả null.

Nếu giữ kiểm nghiệm candidate, dùng **cùng additive nonlinear scorer có prior NBT** trong mọi arm:

\[
s_{ti}=v^T\tanh(W_hh_t+W_ae_i+W_gg_i),
\]

g_i là box/conf. Control thay e_i bằng mean e_bar **trước tanh**, vẫn giữ rho_i từ đúng ROI. Thành phần chung không bị triệt tiêu đại số sau nonlinear interaction với g_i; cần báo mức saturation/gradient, không mặc định mọi tham số đều được dùng tốt. Đây là sửa một lỗi control, không contribution mới.

## 5. Minimum experiment nếu user chọn tiếp ứng viên đã sửa

Không nhận đã được duyệt; không code hoặc chạy. So sánh **ba trainarms và frozen B=C0**, cùng data/init/order/loss/schedule/decoding; A là reference. Không đưa gain inheritedC0 vào công của method.

1. **U-local: ordinary unrestricted pointer/copy comparator.** Cùng appearance rho, nonlinear local pointer, null/OTHER, GTmatching và losses. Gom nonnull nonOTHER mass thành g, conditional copied class distribution thành r; r không cần xác định khi g=0. pCopy(w)=r_c uB(w|c) trong N, zero ngoàiN. pU=(1-g)pB+g pCopy. Đây là matched generate/copy baseline, không nhận tái lập y nguyên toàn NBT.
2. **M-local: candidate cũ với output constraint**, q=(1-g)b+g r, pM=mB q_c uB(w|c) trongN, pB ngoàiN. Cùng sources/learnable components nhưU, không detectorclass inputs.
3. **M-global: nonlinear global-key control ở§4**, cùng massconstraint và đúng ROI lexicalvalues; kiểm giá trị thêm của prefix-specific local correspondence.

U/M có cùng g/r mechanism nhưng khác output policy; conditional class mixture của U cũng đổi vì normalization. Vì vậy M−U là total design effect, **không được diễn giải thành pure causal effect của totalmass riêng**. Thêm một inference-only evaluation, không cần trainarm thứtư: từ pU, giữ conditional distribution trongN và ngoàiN, thay totalmassN bằng mB. Do U ngoàiN tỷ lệ với pB, projection phục hồi pB ngoàiN. U-projected vsU giữ learned weights và fixed-prefix conditionals khi kiểm tác động trực tiếp của marginal constraint. Autoregressive rollout vẫn đổi prefix; báo cả fixed-prefix và fullcaption.

Objective giữ các nguồn thật của proposal: ROIclass CE từ GT-matched predicted proposals; -log positive-set pointer mass; actual noun NLL qua distribution đang generate; original CIDEr+.5object SCST. Không có GTROI forward, detector labels, annotation-absence negatives hoặc held labels trong training. Hệ số.1 cũ chỉ provisional để chốt trước run. Không ép count/class coverage giả80. 500updates/arm là một mức pilot đang đề nghị, không ETA/ngân sách đã xác nhận.

Endpoint: CIDEr/fullCHAIR80 counts, recall, length trên reused dev1000, paired CIs; compareM−B vàM−U, rồiM−global. Tách matched/missing proposal coverage, eligible lexical coverage, correct/wrong detectorclass subgroups để hiểu khả năng sửa semantic labels; subgroup không cứu gate chung. Test jointly permuting tất cả ROI records invariant; giữ values/geometry/h/pB rồi derange appearance keys giữa khácclass để kiểm correspondence; chỉ là representation intervention, không physical causality.

**Chỉ tiếp tục xác nhận** nếu M có tín hiệu đúng hai mục tiêu so ownB, có added utility so unrestricted copy và globalcontrol, recall không giảm quá.005abs, và can thiệp key–value đúng chiều. CI rộng chỉ exploratory. Nếu không hơnU, bỏ luận điểm cần constrained adaptation; nếu không hơnglobal/keyintervention vô ích, bỏ claim local correspondence. Không tăngsteps/strength/n để cứu. Caption giốngB, rho sceneprior, null/OTHERcollapse hoặc EOS do phân tánmass đều là failure phải báo. Không dùng attention đẹp thay outcome.

## 6. Claim bài báo và khuyến nghị

Claim **được phép đặt thành giả thuyết**, chưa là kết quả: “Một cách thích nghi grounded lexical generation cho captioner đã huấn luyện, học liên kết prefix–ROI appearance và chỉ cho bằng chứng vùng đổi object identity trong lexical partition; kiểm tra khi nào cách thích nghi này giữ chất lượng và giảm hallucination so với generate/copy thông thường.”

Nếu có bằng chứng dương tính, contribution vừa mức có thể là **thiết kế thích nghi và phạm vi hữu ích của nó trong điều kiện strong frozen captioner/ít dữ liệu**, được phân biệt khỏi gần-prior bằng matchedcopy/globalcontrols. Không nhận pointer/setloss/normalization là mới, không nhận quan hệ, không nhận chứng minh tổng quát cho CLIP hoặc firstness. Tên “method” không thay cho delta và kết quả. Nếu chỉ U cũng cải thiện hoặc safeguard vô ích, chưa có contribution theo claim đó.

**Tôi sửa khuyến nghị trước**: không triển khai ngay haiarm local/global. Tôi chưa bỏ hoàn toàn candidate vì bài toán tích hợp vào strong pretrained host khác với huấn luyện grounded captioner từ đầu, và positiveproposal supervision không buộc nhãn detector đúng là một khác biệt thực. Nhưng độ mới cơ chế thấp; mức contribution chỉ có thể đứng vững nhờ chứng minh cải tiến thiết kế cụ thể vượt gần-prior. Không đầu tư GPU chỉ vì tổ hợp chưa tìm thấy trùng y nguyên. Cần user cân nhắc candidate đã thu hẹp này cùng phản biện độc lập Argon trước khi chọn.

Facts còn thiếu: actual eligible class/token/proposal coverage trên4k; có bao nhiêu noun events có proposal đúngbox nhưng detectorclass sai; candidate correction head có local semantic signal thực; quota hiện tại; final venue requirements; broader later work có cùng constrained factorization hay không. Facts thứba và hiệu quả caption chỉ pilot mới trả lời, không thể khẳng định từ đọc paper. Không hứa deadline/accepted.

Đã làm: đọc năm primary method sections, cache/hash, visualcheck hai equationpages, phân tích đại số/counterexamples/control. Chưa làm: model/Torch/Kaggle, codeexperiment, metricrerun, PDF toànbộ/fullappendix, exhaustive novelty certification. Đề nghị root phân phối notes; không nhận đãpush/ghisheet tại đây.
