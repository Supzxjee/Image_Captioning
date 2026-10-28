# Astra — rà soát tính mới của T theo cơ chế

Ngày 28/10/2026, Asia/Bangkok. Rà soát tài liệu có mục tiêu, không phải systematic review hoặc bảo đảm không còn prior. Đọc các đoạn phương pháp của nguồn primary được ghi dưới đây; không suy cơ chế từ abstract. Không viết code, chạy model/GPU hoặc đọc lại ZIP.

## Kết luận trước

**T hiện phù hợp nhất với mức “cải biên cụ thể trong họ attribution-prior/model-guidance”, chưa có cơ sở vững cho claim phương pháp mới.** Chưa tìm thấy trong các nguồn đã đọc một phương pháp trùng toàn bộ frozen-A + phrase-set log-mass + branch-path attribution + hai ràng buộc cụ thể. Điều đó không chứng minh tổ hợp này mới, và khác tổ hợp chưa đủ thành đóng góp đáng kể.

Phát hiện làm thay đổi đánh giá trước: Energy loss của Rao et al. (ICCV 2023) đã tối ưu trực tiếp tỷ lệ attribution nằm trong box, dùng cả IG và giám sát ở các tầng trung gian. Khi dùng completeness, directional loss của T cũng trở thành một hinge trên tỷ lệ attribution, có trọng số bằng net gain. Vì vậy, phân biệt trước đây “HINT chỉ ranking, T định lượng phần gain trong vùng” bỏ sót một prior gần hơn.

Tôi **hạ đánh giá cũ “novelty potential vừa phải” xuống “incremental adaptation; originality/độ đáng kể chưa xác lập”**. Lý do là prior mới và phép viết lại dưới đây, không phải vì T500 có điểm thấp. Hiệu quả và originality là hai câu hỏi độc lập.

## 1. Đối tượng được rà soát

Theo `GROUNDED_GAIN_PROTOCOL.md` v0.3:

- A và vocabulary head cố định; thêm readout patch trước Q, 262.144 tham số.
- Với cùng prefix, S chứa các lexical phrase-prefix events rời nhau, cùng annotation ID và cùng mask M. Đây chủ yếu là head/name đối tượng được weak-join, **chưa phải modifier, full descriptive phrase hoặc quan hệ**.
- `F(alpha) = log sum_phi P_alpha(phi | I,p)`; alpha chỉ scale nhánh bổ sung; `Delta = F(1)-F(0)`.
- IG theo nhánh cho `c_j`, với `sum_j c_j = Delta`; `CG=sum_j M_j*c_j`; `CU=Delta-CG`.
- C có main SCST + KL, gain hinge và cancellation hinge. H thêm endpoint-importance ranking. T thêm `relu(.8*relu(Delta)-CG)`.
- Cancellation hinge: `relu(abs(CG)+abs(CU)-2*relu(Delta))`. C đã dùng partition, không phải control region-blind. Phần riêng T-vs-C là direction; không thể quy toàn bộ cancellation cho T riêng.

Ba việc phải tách: ghép đúng phrase–instance/mask; biểu diễn đúng nghĩa; và làm decoder sinh câu tốt hơn. Một attribution loss không tự chứng minh cả ba. Kiểm tra completeness chỉ xác nhận accounting theo đường tính toán đã chọn.

## 2. Tám prior đối chiếu chính

### 2.1 Rao, Böhle, Parchami-Araghi & Schiele — Energy loss, ICCV 2023

**Đã đọc:** bản full-text HTML v2, §3.1–3.5, Eq. (2), (5), (6). PDF CVF bị 403; PDF arXiv quá lớn; dùng bản HTML primary.

`EPG = sum_j M_j*A_j^+ / sum_j A_j^+`; `L_energy = -EPG`.

Học từ box, không ép attribution đều trong box; khảo sát IG/IxG/GradCAM/B-cos và tầng trung gian. Đây là prior gần nhất cho ý tưởng phân bổ contribution vào vùng. Khác T ở positive attribution so với net signed gain, baseline/path và semantic target; chưa thấy frozen-captioner incremental phrase gain hoặc budget hai nhóm như T trong phần đã đọc.

[Full text, §3](https://arxiv.org/html/2303.11932v2#S3). [Trang tác giả xác nhận ICCV 2023](https://sukrutrao.github.io/publication/studying-how-to-efficiently-and-effectively-guide-models-with-explanations/).

### 2.2 Erion, Janizek, Sturmfels, Lundberg & Lee — Attribution priors / Expected Gradients, NMI 2021

**Đã đọc:** preprint full text §2.1–2.2, Methods 1–2, pp. 2–3 và 10–11 theo số PDF; publisher xác nhận năm/venue.

`min_theta L_task + lambda*Omega(Phi(theta,X))`, với Omega là penalty khả vi tùy chọn trên attribution. EG lấy kỳ vọng IG qua baseline và alpha; completeness quy tổng attribution về chênh output so với baseline kỳ vọng. Không giới hạn prior ở target cố định từng feature: có relative importance, smoothness, graph và sparsity.

T thuộc framework này; “dùng IG/completeness để học một ràng buộc contribution” không mới. Các ứng dụng đọc được không chứa đúng caption gain/budget T.

[Full text](https://arxiv.org/pdf/1906.10670). [Publisher](https://www.nature.com/articles/s42256-021-00343-w).

### 2.3 Rieger, Singh, Murdoch & Yu — CDEP, ICML 2020

**Đã đọc:** §3.1–3.3, Eq. (1)–(7), pp. 2–3.

CD tách logits `g(x)=beta(x_S)+gamma(x_rest)`. Objective có task loss và explanation penalty, ví dụ `sum_i,S ||beta(x_i,S)-target_i,S||_1`; paper cho phép thay bằng explainer khả vi khác và regularize feature interactions.

T kế thừa ý tưởng giám sát contribution của group/rest. Khác mục tiêu chênh log-mass của adapter và dạng bất đẳng thức. Không được mô tả CDEP như attention matching hoặc không xét nhóm. Framework rộng không tự bác mọi đóng góp cụ thể, nhưng đặt gánh nặng chứng minh giá trị của constraint T.

[Primary full text](https://proceedings.mlr.press/v119/rieger20a/rieger20a.pdf).

### 2.4 Selvaraju et al. — HINT, ICCV 2019

**Đã đọc:** §4.1–4.3, Eq. (2)–(4), pp. 3–4.

`importance_r=sum_i d o_gt / d P_i^r`. Học ranking importance theo human region importance, cộng task loss. Captioning dùng ánh xạ visual words tới COCO categories/segmentations. Importance là signed gradient sum; không phải chỉ magnitude hay attention.

H là adapted HINT-style endpoint control, không phải reproduction đầy đủ. T khác cả attribution path lẫn constraint. Vì vậy T-H không cô lập riêng lợi ích integration. HINT bác claim đầu tiên dạy captioner dựa vào đúng vùng, nhưng Energy gần hơn ở định lượng phần đóng góp.

[Primary full text](https://arxiv.org/pdf/1902.03751).

### 2.5 Sun, Lapuschkin, Samek & Binder — LRP-IFT, Information Fusion 2022

**Đã đọc:** §V.C.2, Eq. (28)–(32), Algorithm 2, pp. 11–12 của preprint v5.

Signed LRP của context/hidden được normalize theo max absolute, tạo `omega=1+norm(R)`; `p_hat=fc(omega_c*c+omega_h*h)`. Fine-tune bằng hỗn hợp CE hoặc SCST từ phân phối điều chỉnh.

Đã có explanation-guided caption fine-tuning và supporting/opposing evidence. T không trực tiếp reweight activation bằng own relevance: T giám sát attribution theo weak phrase–region join và gain so với A. Không được nhận signed relevance, explanation feedback hoặc giảm hallucination bằng attribution là ý tưởng đầu tiên.

[Primary full text](https://arxiv.org/pdf/2001.01037). DOI: 10.1016/j.inffus.2021.07.008.

### 2.6 Ross, Hughes & Doshi-Velez — Right for the Right Reasons, IJCAI 2017

**Đã đọc:** §2.1, pp. 2–3.

`L=CE + lambda_1*sum_n,d A_nd*(d/dx_nd sum_k log p_nk)^2 + lambda_2*||theta||_2^2`.

Mask chỉ vùng không nên dùng; regularize gradient log-probability trong khi giữ task loss. Vì vậy normalized log-probability và học đúng lý do đều là tiền lệ. Khác T ở endpoint gradients so với path contributions, sum class log-probabilities so với lexical event log-mass, và không có đúng frozen-A gain/budget. Đổi logit thành log-probability tự nó không tạo novelty.

[Primary full text](https://arxiv.org/pdf/1703.03717). [IJCAI record](https://ijcai.org/proceedings/2017/371).

### 2.7 Nguyen et al. — GrAInS, ACL 2026; preprint 2025

**Đã đọc:** primary HTML v2 §3.1–3.3, Eq. (1) và IG, signed top-k; ACL record kiểm venue.

`f(x)=log P_theta(y_pos|x)-log P_theta(y_neg|x)`; tính signed IG trên text/visual embeddings; chọn positive/negative tokens để dựng contrastive steering vectors và can thiệp inference, không update weights.

Overlap: normalized sequence log-probability contrast, signed attribution, VLM intervention. Khác: chênh hai responses của một model so với chênh cùng event giữa adapter-on/off của T; steering so với supervised regional adaptation. Không nên đánh đồng hai loại contrast chỉ vì cùng ký hiệu Delta.

[Method](https://arxiv.org/html/2507.18043v2#S3). [ACL record](https://aclanthology.org/2026.acl-long.2159/).

### 2.8 Tao & Thakur — ProGrad, NeurIPS 2025

**Đã đọc:** Definition 1.1/Eq. (1), §5.2, pp. 1, 9.

`min ||theta_hat-theta||` với các linear Jacobian constraints. §5.2 ép normalized token IG gần IG của teacher LLM: `||L_n-L'_n||_infinity <= epsilon`.

Đã có post-training editing để thực thi attribution constraints, kể cả IG trên LLM. Khác T: hard constraints/provable editing, teacher-attribution target; không tìm thấy same incremental grounded-gain constraints trong phần đã đọc. T là soft regularization và hiện chưa đạt constraints đáng kể; không được gọi là đảm bảo hoặc certificate chỉ vì dùng IG completeness.

[Primary full text](https://proceedings.neurips.cc/paper_files/paper/2025/file/9ecca44ee9150cfe350bcb3660d18c98-Paper-Conference.pdf).

## 3. Các nhánh tìm thêm có thể bác claim rộng

- **Model differences / constrained adaptation:** Artelt et al., *Contrasting Explanations for Understanding and Regularizing Model Adaptations*, online 2022, NPL 2023. Đọc §5 và §6.1–6.3, Eq. (15), (23), (25): giữ explanation/counterfactual/pertinent-positive hợp lệ sau adaptation. Không phải phân rã gain của output theo vùng. Nó bác claim rộng rằng kiểm soát explanation khi adaptation là bài toán chưa có. [Full text](https://link.springer.com/article/10.1007/s11063-022-10826-5).
- **Cancellation:** Liu, Najmi & Sundararajan, *The Penalty Imposed by Ablated Data Augmentation*, arXiv 2020. Đọc §2.2, Eq. (1), §3.1 và Remark 2: `-n*sum_j!=k cov(contribution_j,contribution_k)` và thay contributions bằng IG ở deep networks. Là covariance qua dataset, khác budget trị tuyệt đối của hai nhóm trong một example. Đã có nghiên cứu training/regularization giảm cancellation; không claim khái niệm ấy mới. [Full text](https://arxiv.org/pdf/2006.04769).
- **Caption attribution residual:** CAPEV, *Enhancing Caption Fidelity via Explanation-Guided Captioning with Vision-Language Fine-Tuning*, preprint 01/08/2025, chưa peer-reviewed tại trang đã đọc. Đọc §4.6 Eq. (37): `c_tilde=c*(1+lambda*tanh(R(c)))`; cơ chế gần relevance reweighting, không phải same gain allocation. Không dùng các claim thực nghiệm của preprint này làm bằng chứng chất lượng. [Primary text](https://www.preprints.org/manuscript/202508.0076).
- **IG nền tảng:** completeness/path integration là kế thừa từ Sundararajan, Taly & Yan, ICML 2017, không phải theorem mới của T. [Primary paper](https://proceedings.mlr.press/v70/sundararajan17a.html).

## 4. Kiểm tra đại số: novelty còn hẹp hơn cách mô tả ban đầu

Phần này là suy luận riêng trên protocol T, **không phải kết quả của các paper**.

### 4.1 Direction là một attribution-ratio hinge có trọng số

Với `Delta>0`, đặt `r=CG/Delta`. Completeness cho `CU/Delta=1-r`:

`Ldirection = Delta * relu(.8-r)`.

Do đó diễn đạt “target phụ thuộc actual gain” đúng về đơn vị, nhưng không chứng minh một loại objective hoàn toàn khác. Nó là ngưỡng tỷ lệ phần contribution trong support, được cân theo độ lớn gain.

Nếu mọi `c_j>=0`, thì `sum c_j^+=Delta`, và `r=sum M_j*c_j^+ / sum c_j^+` chính là dạng tỷ lệ Energy/EPG. T không trùng objective: hinge target .8 thay vì tối đa tỷ lệ, có trọng số Delta và gain objective riêng. Tuy vậy, trên miền này, khác biệt nằm ở threshold/weight/target-path; không thể bán như nguyên lý localization mới. Khi có signed cancellation, hai tỷ lệ khác nhau; cần chứng minh việc giữ dấu đem lại utility chứ không chỉ đổi denominator.

### 4.2 Cancellation budget là ràng buộc quen thuộc trên signed mass

Đặt `N_group=(abs(CG)+abs(CU)-Delta)/2` khi `Delta>0`. Đây là tổng độ lớn phần âm của **hai aggregate**:

`Lcancel = relu(2*N_group-Delta)`.

Loss bằng 0 cho phép `N_group <= Delta/2`. Kết hợp zero direction loss cho miền:

`.8 <= CG/Delta <= 1.5`.

Ví dụ `CG=1.5*Delta`, `CU=-.5*Delta` vẫn thỏa cả hai constraints. Vì vậy “80% evidence nằm trong box” quá mạnh: đây là tỷ lệ **net signed computational attribution**, không phải 80% positive evidence. Budget không kiểm soát cancellation bên trong support hoặc complement.

Một ràng buộc signed mass cụ thể có thể là cải biên đáng nghiên cứu, nhưng phép viết lại trên không phải theorem khó đủ tạo novelty. Prior về cancellation tồn tại, dù chưa xác lập exact equivalence với budget này.

### 4.3 Baseline A có ý nghĩa ứng dụng, không phải thuật toán attribution mới

Xem toàn bộ patch corrections như input `b` của hàm `f_A(b)`, với A/prefix cố định. Path `alpha*b` đi từ baseline `b=0` tới `b`: đây là IG thông thường trên không gian nhánh. Baseline zero branch tương ứng captioner A là lựa chọn có ý nghĩa cho adaptation; completeness tới Delta kế thừa trực tiếp.

Không đồng nhất nó với `IG_input[f_theta-f_A]`: cách sau thay đổi input và có thể làm A thay đổi theo path; protocol T giữ A fixed theo branch path. Phải so đúng intervention, không chỉ tên “relative attribution”.

### 4.4 Phrase set không tự đem lại compositional semantics

Marginalize nhiều observed lexical alternatives và differentiate logsumexp là phép xác suất/chain rule chuẩn. Common-referent guard xử lý đúng một lỗi nhãn/accounting, có giá trị kỹ thuật nhưng không tự trở thành đóng góp semantic mới. Bank hiện không hỗ trợ claim về quan hệ, binding hai đối tượng hoặc adjective–noun composition.

## 5. Claim tối đa hiện có và evidence còn thiếu

**Có thể mô tả như đề xuất đang kiểm định:**

> Một attribution-prior cho readout bổ sung trên captioner cố định, dùng IG chuẩn để phân rã mức tăng log-probability của tập tên gọi cùng thực thể và áp soft constraints lên tỷ lệ signed contribution của vùng ghép, cùng budget cancellation giữa hai nhóm.

Đây là mô tả cơ chế, chưa phải claim đầu tiên hoặc hiệu quả. Không dùng: new attribution algorithm; first grounding-by-contribution; first signed explanation fine-tuning; causal recognition; certified grounding; compositional relation understanding; đã cải thiện captioning.

Phần có thể còn là cải biên riêng: exact branch-relative lexical-event target và signed two-group objective trong bối cảnh frozen autoregressive adaptation. Chưa có căn cứ coi tổng các lựa chọn này là contribution đủ mạnh cho bài phương pháp. Cụ thể hóa được không đồng nghĩa mới hoặc đáng kể.

**Đối chiếu cần thiết nếu tiếp tục hồ sơ phương pháp:**

1. Energy/EPG-style localization trên cùng attribution path/semantic target/mask là prior-control quan trọng hơn việc chỉ dùng endpoint HINT-style H. Nó giúp kiểm signed gain/budget có hơn positive-attribution concentration không. Đây là yêu cầu để bảo vệ claim tương lai, không tự thêm arm/đổi trial đang khóa.
2. C/H/T hiện chỉ so hai constraint packages; chưa cô lập integration khỏi ranking-vs-fraction và scale. Nếu nói “IG tốt hơn endpoint gradient” thì thiết kế hiện không đủ.
3. Phải phân biệt lợi ích gain objective, localization và cancellation. C chứa cancellation, nên T-C chỉ kiểm direction, không chứng minh toàn bộ package mới.
4. Evidence cần có cả caption benefit, constraint behavior trên fixed adapter-held-out groups, và actual generated-mention evaluation đúng coverage. Train loss/completeness/heatmap đẹp không thay thế các bằng chứng này.

Không đề nghị thay hệ số, threshold, đổi tên hoặc thêm module chỉ để né prior. Nếu các lựa chọn riêng không có ích hơn prior đơn giản thì nhận đây là adaptation/negative result.

## 6. Tác động đến kế hoạch duration

Một trial tới 2.500 updates vẫn có thể trả lời câu hỏi kỹ thuật: 500 updates đã quá ngắn hay chưa. **Nó không còn có thể được trình bày như phép thử đầy đủ cho originality/đóng góp riêng của T**, ngay cả khi T thắng A/C/H. Đừng dùng gain sau train dài làm câu trả lời cho Energy prior.

Khuyến nghị riêng: giữ phân biệt hai quyết định. Nếu mục tiêu là khảo sát candidate đã dựng với chi phí giới hạn, matched duration trial vẫn có giá trị exploratory. Nếu điều kiện chi GPU là phải có một đóng góp mới được bảo vệ ngay, evidence hiện tại chưa đạt điều kiện ấy; không hứa trial sẽ biến T thành paper method. Không launch hay sửa trial theo review này.

Kết quả T500 vẫn là completed quality no-go; A vẫn là mặc định. Không được viết lại kết quả đó hoặc dùng thiếu Energy control để bác hoàn toàn giả thuyết căn chỉnh.

## 7. Phạm vi tìm kiếm và giới hạn

Các cụm tìm theo cơ chế gồm attribution priors/expected gradients; right-for-right-reasons; explanation regularization; model differences/model adaptations; attribution-guided fine-tuning cho caption/VLM; positive/negative relevance; cancellation; sequence log-probability contrast; attribution constraints. Theo citation trail tìm được Energy loss và cancellation paper, thay vì chỉ exact-match tên T.

Full text có sẵn và đọc các đoạn method ghi ở trên; **không tuyên bố đã đọc toàn bộ từng paper**. Không dùng abstract-only hit để kết luận trùng công thức. Search ranking/index coverage hữu hạn; review này không phải patent novelty search, không cho phép nói “chưa ai làm”. Không xác minh toàn bộ cited references hoặc mọi biến thể 2026. Không kiểm tra implementation của các prior hoặc chạy reproduction. Các nhận định hiệu quả C/H/T dựa vào audited results root đã chuyển, không đánh giá lại dữ liệu.
