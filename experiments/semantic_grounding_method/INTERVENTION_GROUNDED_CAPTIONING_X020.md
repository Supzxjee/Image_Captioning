# X020 — Chọn hướng học phụ thuộc vùng–token bằng hoán đổi bằng chứng có nhãn bộ phận

Ngày 01/11/2026, giờ Việt Nam. Base: `074897ef36123943778018e005e2ff17e5258e8d`.

**Khuyến nghị:** chọn phát triển hướng X019, thay vì tiếp tục JS-correspondence của X018 hoặc chuyển sang một biến thể focal contrast/xóa đối tượng chỉ để có tên mới. Đây là **ứng viên cải tiến phương pháp**, chưa là đóng góp mới đã xác lập và chưa được duyệt triển khai. Tài liệu này cung cấp lựa chọn, forward/loss, đối chứng và điều kiện bác bỏ; không báo kết quả mô hình mới.

## 1. Phạm vi và quyết định nghiên cứu

Đã đọc `AGENTS.md`, `RESEARCH_STATE.md`, X015–X019 trong `RESEARCH_EXCHANGE.md`, báo cáo tuần bản ngắn, C0/expanded results và response15. HEAD trùng main tại lúc kiểm tra. Dự án thực được bàn giao là `Supzxjee/Image_Captioning`; không tìm thấy checkout riêng tên `upt_alignment_publish`, nên không suy kiến trúc từ tên đó.

Áp dụng ARS-Codex `academic-research-suite` 3.24.0: scoping → tìm nguồn sơ cấp → đối chiếu cơ chế → phản biện → đề xuất có phép bác bỏ. User đã yêu cầu trực tiếp sinh ứng viên; đã thông báo `[SOCRATIC-NON-GENERATION-EXIT: explicit_user_request]`. Đây là memo thiết kế theo đầu ra user yêu cầu, không phải systematic review hoặc một pipeline viết bài đã được phê duyệt. Có một native subagent hỗ trợ đọc nguồn; nguồn và lựa chọn cuối được người viết kiểm lại. Không có cross-model review hay một hội đồng phản biện độc lập được giả lập.

**Câu hỏi đề xuất:** với cùng captioner, dữ liệu, nhãn vùng–từ và ngân sách cập nhật, giám sát **phản ứng của actual caption readout khi thay riêng evidence value của một vùng** có thêm lợi ích so với giám sát factual từng ảnh không — cả về dùng đúng vùng và caption tự sinh?

Giả thuyết cần thử: factual grounding có thể cho attention đúng nhưng vẫn để generator dựa chủ yếu vào đường global/prompt. Ép một thay đổi có nhãn tại đường local value phải tạo đúng thay đổi token, trong khi giữ các fact đã xác minh không bị tác động, có thể làm correspondence hữu ích cho generation. **Chưa chứng minh đây là nguyên nhân lỗi của C0/D/E.** Đây là causal intervention *trong mô hình*, không khẳng định đã mô hình hóa quan hệ nhân quả của thế giới ảnh thật.

### Những gì đã biết và không được diễn giải quá mức

| Bằng chứng đã lưu | Kết luận được hỗ trợ | Không suy ra |
|---|---|---|
| Expanded4000: A/C0/C CIDEr 1.189771/1.204560/1.207042; false captions 239/235/230 | C/C0 tăng CIDEr so A theo paired intervals; C−C0 và giảm hallucination chưa qua gate | Preservation có đóng góp riêng; đã đạt hai mục tiêu; validation là independent test |
| Response15: direction14/15, Changed-both7/15, Stable-both9/15, joint4/15 | C0 có phản ứng với ảnh dưới original prompt cố định; phản ứng chưa luôn đúng | Lỗi chắc chắn do nhầm region; 4/15 là free-caption accuracy; sửa màu làm CHAIR tốt hơn |
| Các thử nghiệm attention/auxiliary/KV chưa có added caption utility rõ | Cần kiểm đường evidence→actual output, ngoài điểm phụ | Mọi region model đều vô ích; module trước hoàn toàn unused |

Nguồn: [expanded](../concept_mass_grounding/EXPANDED_RESULT.md), [C0](../concept_mass_grounding/C0_RESULT.md), [response15](RESPONSE15_C0_V2_RESULT.md). Không gộp CIDEr cohort1000 với4000; không dùng quota cũ làm ngân sách hiện tại.

## 2. So sánh đúng ba hướng thực chất khác nhau

| Hướng | Cơ chế và ưu điểm | Rủi ro/nearest prior | Quyết định |
|---|---|---|---|
| **A. Học channel vùng→token bằng partial interchange** | Huấn luyện actual generator dưới hoán đổi internal value; known Updated/Protected targets, UNKNOWN bỏ loss. Có phép thử trực tiếp chống nhánh local bị bỏ qua. | IIT đã có, cả vision-language; partial-label masking cũng không mới. Cần chứng minh partial target construction hợp lệ và thêm utility ngoài factual grounding. | **CHỌN.** Phạm vi hẹp và câu hỏi bác bỏ rõ nhất. Kế thừa X019, không giả vờ là một họ mới do X020 phát minh. |
| B. Focal contrast trên nhiều giả thuyết vùng, marginalize trước contrast | Thay lựa chọn cặp FoV cực trị bằng shared-support aggregation để giảm nhạy với việc chọn một detection/view. Tác động trực tiếp logits, ít nhãn paired mới. | HALC đã có detector/FoV/JSD/contrast; ARegionCD đã có region amplification/suppression. Mixture hoặc ratio-of-mixtures là xác suất chuẩn; cùng weights giữa hai mixtures cũng không bảo đảm hai tổng bị chi phối bởi cùng region. | Không chọn làm contribution chính. Có delta kỹ thuật có thể thử, nhưng hiện gần một thay đổi pooling/selection hơn một giả thuyết alignment đủ mạnh. |
| C. Huấn luyện bằng xóa bằng chứng đối tượng trong ảnh | Có vùng cụ thể, supervise actual entity likelihood; không cần nhãn caption phản thực toàn câu. | See or Guess đã dùng phrase→nhiều box, xóa toàn tập và TE/NDE likelihood losses. Xóa một instance nhưng còn instance cùng lớp không cho nhãn noun-absent. Mask artifacts/context shift còn gây nhiễu. | Không chọn việc xóa vùng, multi-box mask hoặc change/stay đơn thuần làm mới. Matched sham-erasure là control đáng có nếu theo họ này, chưa tự là contribution. |

**Tại sao không tiếp tục X018 như cũ?** JS giữa hai router có thể bằng0 khi cả hai cùng sai, và tự bằng0 khi factual grounding đã one-hot đúng. Điều đó không chứng minh JS vô ích; nó chỉ không cung cấp lý do đủ mạnh để lấy JS làm đóng góp chính. Hướng A hỏi một điều khác: *value được lấy từ region có điều khiển đúng từ hay chỉ được attention nhìn tới?*

**Lựa chọn không đồng nghĩa clearance:** nghiên cứu thêm ở lượt này củng cố việc không chọn B/C dưới một tên mới. Không có căn cứ để tuyên bố A đã mới; phần đề xuất là một cách thích nghi IIT với nhãn phản thực chỉ xác định một phần cho native caption generation, cần user/Codex1 phản biện trước code.

## 3. Forward tối thiểu và chính xác vị trí căn chỉnh

### 3.1. Backbone thực và phần thêm

Giữ **C0 frozen**, gồm CLIP, Q-Former32, original prompt/attention/gate, toàn bộ decoder và `fc_out`. A có original prompt; C0 không phải text-only và không phải no-prompt. Không thay tokenizer, vocabulary hoặc policy sinh caption. A là historical reference; C0 mới là reference trực tiếp của adaptation này.

`captioning/models.py:CaptionDecoder` có transformer hidden rồi `fc_out`; thiết kế đặt adapter giữa hai khối đó. Đây là hook cần triển khai sau duyệt, **chưa tồn tại dưới dạng trainer/module mới ở lượt này**. Nguồn native C0 wrapper ở `experiments/concept_mass_grounding/concept_model.py` giúp xác định cùng decoder và cùng output vocabulary; không nạp checkpoint để xác minh runtime ở đây.

K≤6 proposals là mức khởi đầu đề nghị để tương thích bank trước, không phải coverage đã biết cho cohort mới. Detector frozen. Dùng appearance của **crop RGB encode độc lập** bằng cùng pinned CLIP, không gọi pooled contextual patch của toàn ảnh là local evidence đã cô lập. Mỗi crop có thể còn background/overlap; chưa là causal pixel isolation. Phải ghi detector/preprocessing/coordinate policy chính xác trước run.

Với h_t∈R512 là native C0 hidden, z_i∈R768 là crop feature, g_i là geometry/confidence không chứa class-name embedding:

```text
q_t = W_q h_t
k_i = W_k z_i + W_g g_i
s_ti = aᵀ tanh(q_t + k_i)           # d=128 đề nghị
π_t = softmax(s_t0, s_t1, ..., s_tK)
v_i = W_v z_i ; v_0 = 0             # null slot
r_t = Σ_i π_ti v_i
ℓ_t = W_B (h_t + W_o r_t) + b_B
p_θ(· | I, original_prompt, y_<t) = softmax(ℓ_t)
```

W_B,b_B là `fc_out` của C0, frozen. Null scorer có tham số riêng; không tạo crop giả cho null. K=0 thì residual0 và exact C0. Trainable chỉ W_q,W_k,W_g,W_v,W_o và additive scorer/null parameters; detector, CLIP và native C0 không cập nhật. W_o khởi tạo0, phần trước dùng initialization thông thường: factual forward ban đầu exact C0; gradient đầu tiên có thể vào W_o, các projection trước mới nhận gradient sau đó. Không zero toàn branch hoặc khóa một hard gate ở0. Giữ C0 ở eval mode trong training adapter để dropout không gây khác hidden giữa factual/patched pass.

Đây là cross-attention/residual/readout quen thuộc, **không nhận kiến trúc này là phát minh**. Không dùng một vocabulary head chỉ để chấm điểm phụ rồi bỏ khi decode.

**Đơn vị căn chỉnh:**

1. Fact/content-token event ↔ tập proposal dương tính G_j, thông qua π_t.
2. Thay value của proposal thuộc entity đó ↔ token update đã biết tại **cùng fc_out dùng để sinh caption**.
3. Fact khác gắn vùng không đổi ↔ correct-token response được bảo vệ khi dependency hợp lệ.

Đường bằng chứng thật ở inference là `crop pixels → frozen CLIP z_i → W_v → r_t → W_o → native fc_out → token probability → next prefix`. h_t vẫn có global visual/prompt information; kiến trúc không loại bypass và không được mô tả như vậy.

### 3.2. Internal intervention

Với base b, donor s và một correspondence đủ rõ (k ở base, l ở donor):

```text
factual:       r_t^b = Σ_i π_ti^b v_i^b
patched:       r_t^(b←s,k) = r_t^b + π_tk^b (v_l^s − v_k^b)
patched logits = W_B (h_t^b + W_o r_t^(b←s,k)) + b_B
```

Trong paired pass giữ đúng giá trị h_t^b, π_t^b, geometry/keys và mọi value khác. Không chạy lại router sau khi sửa key; đó sẽ là intervention khác. Donor value được tính bởi cùng W_v hiện tại, gradient đi qua cả hai nhánh nếu không frozen. Việc giữ tensor cùng giá trị giữa hai pass **không đồng nghĩa detach gradient của router**. C0 frozen nên không cần gradient cho h_t, nhưng gradient của adapter phải được giữ.

Inference trên ảnh thật chỉ dùng factual pass, không donor, không gold boxes hoặc target-conditioned pruning. Đây là training intervention; không dựng một ảnh mới rồi nhận ảnh base thật chứa donor.

## 4. Nhãn bộ phận: không có oracle cho toàn caption

Mỗi event cần record logic `(base_source, donor_source, proposal_match, prefix, affected_variable, target, status, evidence, split)`. Đây là schema đề nghị, chưa stage dữ liệu.

| Status | Điều kiện đủ để dùng | Supervision |
|---|---|---|
| **Updated** | Donor có label xác minh; base/donor cùng variable type; prefix tương thích cả hai, không chứa identity cũ buộc câu tự mâu thuẫn; binding không ambiguous | CE đúng donor content token/window trên patched pass |
| **Protected** | Fact có label xác minh ở base, vùng hỗ trợ không đổi và độc lập với variable bị thay trong partial semantic model đã khai báo | CE correct token của base; không ép probability giống hệt |
| **UNKNOWN** | Count, relation, donor attribute chưa biết, overlap, câu tham chiếu không rõ, old-identity trong prefix, thiếu correspondence | Không tạo loss đúng/sai cho event đó |

Ví dụ object route: positive base bus và donor truck; prefix vô hại như “a” chỉ được dùng nếu event/binding được định nghĩa rõ và không có các support khác làm target ambiguous. Đây là target cho **virtual local variable**, không xác nhận bus vắng khỏi ảnh base. Với nhiều instance cùng lớp hoặc nhiều proposal khác nhau của một entity, không bắt “đổi một box ⇒ mọi noun cùng class phải đổi”. Initial intervention pilot chỉ giữ unique binding đủ rõ; factual grounding có thể vẫn dùng multi-positive G_j. Không biến việc skip thành bằng chứng phương pháp xử lý multi-instance.

Ví dụ attribute route: bus đỏ→xanh, car trắng không đổi. Chỉ hợp lệ khi có correspondence bus và car ở cả hai ảnh và labels tương ứng. Nếu chỉ biết bus-color mà donor có thể đổi class/shape, các dependent facts phải UNKNOWN. Nhãn color15 cũ vẫn là development; không chấm lại truth labels hoặc dùng làm held test.

**Điểm mới cần bảo vệ nếu theo hướng này:** cơ chế xây một *partial semantic transition* cho native captioner khi không có gold caption toàn cảnh sau internal swap; chỉ supervise những output events suy ra hợp lệ từ positive region labels, cùng channel thực dùng khi decode. Generic IIT, masked CE, positive boxes, typed donor swaps và change/stay không phải phần mới. Nếu nearest prior đã làm cùng partial transition và output coupling, phải rút claim phương pháp; đổi backbone/dataset sẽ không cứu được.

## 5. Loss và vì sao local branch không thể hoàn toàn né nhiệm vụ

Gọi E_real là caption events ảnh thật, E_fact là true fact events đã xác minh, E_bind là positive grounding events, E_cf là Updated/Protected events hợp lệ. Các mean phải lấy trên đúng số valid events; không lấy UNKNOWN làm zero negative hoặc để số skip đổi effective loss không được ghi nhận.

```text
L_common = mean_Ereal [−log p_θ(y_t | I,prompt,y_<t)]
         + λ_f mean_Efact [−log p_θ(a_j | I,prompt,c_j)]
         + λ_g mean_Ebind [−log Σ_(i∈G_j) π_(c_j,i)]

L_partial = mean_Ecf [−log p_θ^(b←s,k)(a*_j | prompt_b,c_j)]
L_candidate = L_common + λ_cf L_partial
```

Không thêm SCST/reward mới trong pilot này; giữ nhãn và question về intervention placement rõ. Các λ/LR/update count là hyperparameters cần khóa trong protocol trước run; không mượn0.1/500 từ C0 làm giá trị đã được duyệt. Caption CE dùng reference thật cho từng ảnh, không dùng một câu tự ghép sau swap làm full-caption gold.

Initial pilot dùng unambiguous **single-token native aliases** để content-token masks và so sánh likelihood không bị độ dài/OOV làm nhiễu. Phải kiểm actual vocab trước quyết định danh sách; không sửa vocabulary hoặc giả mọi COCO class có alias hợp lệ. Multiword extension cần exact teacher forcing và mask từng token, chưa nằm trong pilot tối thiểu.

Với hai từ a,b tại cùng fixed prefix và W_B linear:

```text
Δ[log p(a) − log p(b)]
  = (W_B[a] − W_B[b]) W_o π_tk (v_l^s − v_k^b).
```

Đây là đẳng thức kiểm đường truyền, không theorem mới. Nếu W_o=0, hoặc donor/base values không phân biệt ở readout, patched output không đổi và không thỏa những Updated targets thật sự xung đột với factual label. Grounding π đúng riêng không đảm bảo điều đó.

Toy stdlib, không Torch/model: base donor-minus-original log-odds−2; π=.8; readout contrast của value difference=3 thì log-odds sau swap=.4, donor probability .598688 thay vì .119203 khi branch0. Nó chỉ minh họa đường tác động có thể tồn tại. Nó không chứng minh optimizer học được, label masks đúng hay caption sẽ tốt hơn.

**Vì sao có thể giúp caption:** cùng local value và fc_out được dùng khi generate nên supervision không chỉ tối ưu một probe bỏ đi. Nhưng teacher-forced prefix có thể không xuất hiện trong caption tự sinh; model còn có thể overfit donor/type hoặc gây lệch ngôn ngữ. Vì thế cơ chế là điều kiện cần kiểm, không lời hứa tăng CIDEr/giảm CHAIR.

## 6. Nearest priors và mức nguồn thực đọc

Đối chiếu này nhắm method/collision, không exhaustive novelty search. Full-text availability và code không có cùng sức chứng minh. Không dùng nhãn conference do README tự ghi như đã xác minh mọi chi tiết bibliographic qua publisher.

| Prior | Phạm vi thực đọc lượt này | Phần kế thừa | Delta đề xuất / nguy cơ trùng |
|---|---|---|---|
| **Geiger et al. (2022), Inducing Causal Structure for Interpretable Neural Networks, ICML/PMLR162** | Publisher PDF; §3 Eq1–4, §4 typed intervention Eq6 và phần ReaSCAN §5/Eq9–10. Không chỉ abstract. | Internal activation swap; typed swaps; loss khớp gold high-level counterfactual output; vision-language causal alignment cũng đã có. | Bài đã có symbolic high-level solver. Ta đề nghị partial target construction cho caption ảnh thật thiếu solver/toàn-caption counterfactual, và local channel qua actual native readout. **Task adaptation + masking riêng chưa đủ originality**; cần chứng minh delta này có giá trị so factual replay. Không áp zero-IIT-loss theorem cho sampled partial labels. |
| **Neural Baby Talk, Lu et al. (2018)** | Scope §3.1–3.4/Eq4–13 và primary links được ghi trong review trước; **không đọc lại full PDF ở lượt này**. | Pointer/sentinel, multi-positive region supervision, region→lexical generation. | Regional adapter/readout là thành phần kế thừa. Không loại collision với các grounded-generation extensions chỉ vì backbone khác; source read này không cấp clearance cho cả họ. |
| **See or Guess: Counterfactually Regularized Image Captioning (Cao et al., ACM MM2024, theo author repo)** | README + `cfr_caption_datasets.py`, `CFRLoss.py`, `modelling_opt.py` ở pinned commit. **Không có full paper mới trong lượt này**. | Phrase→list of boxes; blacken/blur toàn set; TE/NDE kiểm actual noun likelihood và có training loss. | Ở code đọc được, can thiệp pixels/image rồi so likelihood; không phải giữ native global h và router cố định để supervise donor-label replacement riêng một value channel. Không nhận counterfactual captioning/multi-box masking/output dependence là mới. Chưa có full-method exclusion. |
| **How Many Counterfactuals Does It Take? Probing VLM Hallucinations Through Circuits and Causal Effects (2026, author repo)** | README; `training/crm_loss.py`, `training/trainer.py` ở pinned commit; **chưa full paper**. | Activation intervention/influence; code CRM reweight factual BCE bằng clipped probability ratio factual/counterfactual với weight detach. | Candidate dùng donor-labeled patched-token CE, không chỉ reweight factual BCE. Chỉ kết luận về các hàm đã đọc, không kết luận toàn paper/repo thiếu partial target training. Đây là blocker novelty quan trọng. |
| **HALC, Chen et al. (2024), ICML/PMLR235** | Full publisher PDF §4.2–4.4/Eq2–4 và author code. | Detector-grounded FoV, top-JSD pairs, two-way focal contrast, BLIP global beam selection. | Đối chứng cơ chế cho hướng B, không phải novelty của A. Không gọi local visual correction/logit contrast mới. Paper nói một retrieved context; quy tắc chọn smallest detection là ở code, không gán nhầm thành phát biểu của paper. |
| **VCD; ARegionCD; RITUAL** | Author README + sampling/highlight/guidance code; **chưa full papers**. | Clean/distorted logit contrast; region-conditioned amplification/suppression; transformed-view combination và plausibility filters. | Các lead này làm yếu claim chỉ đổi cách chọn/pool visual views. Không có nguồn để nhận X020 đã loại hết các biến thể learned/partial intervention. |

### Nguồn sơ cấp, locator và receipts

- IIT: [publisher PDF](https://raw.githubusercontent.com/mlresearch/v162/gh-pages/geiger22a/geiger22a.pdf), [publisher landing](https://proceedings.mlr.press/v162/geiger22a.html). PDF SHA256 `f97e0d0d4b3285af9a2727b26442131f906cc6ff358755b39906d652f4ef665d`;15pages, extracted text đã đọc ở scope trên.
- HALC: [publisher PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/chen24bi/chen24bi.pdf), [publisher metadata](https://github.com/mlresearch/v235/blob/gh-pages/_posts/2024-07-08-chen24bi.md). PDF SHA256 `27f5d7f07a330b5fe3f13f4754e89735d1fc69ded6d047a32899c7f9092dfcff`. Đây là publisher version, không đồng nhất hash với arXiv copy được ghi ở review cũ.
- HALC [crop selection](https://github.com/BillChan226/HALC/blob/c757ced4f1f26d992e6e7200a44d3b899bb000af/decoder_zoo/HALC/context_density/halc.py#L266), [contrast implementation](https://github.com/BillChan226/HALC/blob/c757ced4f1f26d992e6e7200a44d3b899bb000af/decoder_zoo/HALC/context_density/halc.py#L922). Eq3 là standard JSD; các hàm code đọc dùng `kl_div(log p,M)`, tức KL(M||p). Nếu làm comparator phải ghi paper-equation hay code-equivalent, không âm thầm coi chúng bằng nhau.
- NBT: [primary landing](https://openaccess.thecvf.com/content_cvpr_2018/html/Lu_Neural_Baby_Talk_CVPR_2018_paper.html); scope đọc cũ ở [review](REGION_LEXICAL_NOVELTY_REVIEW_2026-10-30.md).
- See or Guess [set masking](https://github.com/Aman-4-Real/See-or-Guess/blob/75b6dd1d316dc9f8ec403c06d8308fd1f8b44daa/src/lavis/datasets/datasets/cfr_caption_datasets.py#L24), [TE/NDE code](https://github.com/Aman-4-Real/See-or-Guess/blob/75b6dd1d316dc9f8ec403c06d8308fd1f8b44daa/src/lavis/transformers_v4p34_local/models/opt/CFRLoss.py). TE trong code lấy logprob entity tại reference position trừ mean logprob entity qua positions của generated caption; không viết lại nó như same-prefix contrast. NDE dùng generated-caption positions dưới original/masked image.
- Mechanistic counterfactual [CRM loss](https://github.com/dsgiitr/mechanistic-counterfactual-grounding/blob/4f2b277b4bab948d8f885fa2057e82a7b7e579ae/training/crm_loss.py), [trainer](https://github.com/dsgiitr/mechanistic-counterfactual-grounding/blob/4f2b277b4bab948d8f885fa2057e82a7b7e579ae/training/trainer.py#L205), [paper lead](https://arxiv.org/abs/2606.08777).
- VCD [sampling](https://github.com/DAMO-NLP-SG/VCD/blob/d6568ff81b8fd306a49e630df44f2db5c2300191/vcd_utils/vcd_sample.py#L139): `(1+α)f_clean−αf_distorted`, clean plausibility filter; distortion là diffusion noise, không gold object erasure.
- ARegionCD [positive/negative paths](https://github.com/ecoxial2007/ARegionCD/blob/ddd220db3950fa6283c2a0ee4e4d0ae52d618138/evaluate_ARegion.py#L20), [guidance](https://github.com/ecoxial2007/ARegionCD/blob/ddd220db3950fa6283c2a0ee4e4d0ae52d618138/src/highlighter_modules/guidance.py#L19): `s log p_cond + (1−s) log p_uncond` khi use_log. Bản này có region masks, không được bỏ khỏi prior chỉ vì medical task.
- RITUAL [sampling](https://github.com/sangminwoo/RITUAL/blob/52065a663927bff2da370837f0290343dc14a643/ritual_utils/ritual_sample.py#L166): positive transformed-view logits được cộng vào clean logits; đây là lead code, không full-text clearance.

Search ngày01/11: GitHub repository search với nhóm `grounded captioning counterfactual`, `visual grounding hallucination`, `captioning grounding`, `counterfactual caption`, `causal grounding`, `multi contrastive hallucination`; follow official author code và publisher metadata; không loại seminal work theo năm. GitHub search có độ phủ hạn chế, không phải scholarly database. arXiv/CVF trực tiếp bị network403; SemanticScholar/OpenAlex/Crossref không được dùng để giả có search coverage. Không dùng0results làm absence proof. Raw PDFs/text/code ở `/tmp/ars_primary`, không commit PDF có bản quyền hoặc private artifacts vào repo. Không tải model/data của các repo prior.

## 7. Dữ liệu cần thật sự có

| Nguồn | Có bằng chứng gì trong repo | Còn thiếu cho pilot này |
|---|---|---|
| C0/A/native vocab, original prompt, captions/visual caches | Source + kết quả/identity/hashes ở protocol/result trước; C0 SHA `5d92cd5cb48077c7332bb3ff9b295cad1df5d76ede818adabef28035f055fa9d` | Payload hiện đang mount hay không phải audit khi chuẩn bị run; metadata lịch sử không là current mount. Hook hidden/pre-head và parity phải kiểm trên Kaggle. |
| COCO positives/boxes, predicted banks | Có matching utilities, unique-GT/IoU policies từng dùng; source `kv_data.py` | Join đúng source image IDs/crop visibility/native aliases/split cho candidate mới. K≤6 không có nghĩa phủ mọi class. Detector miss/reference omission không là visual absence. |
|15MED verified pairs | Truth labels và provenance đã user-confirm; packet lệch màu, đã xem kết quả | Box correspondence/proposal payload phù hợp edited images còn thiếu; không dùng làm held evaluation mới. |
| UPT/V-COCO/VSR | Có các nguồn quan hệ riêng, scopes và coverage đã ghi ở các thử nghiệm | Không thay positive noun/instance labels cho arbitrary crop hoặc gold caption; không tự gom thành một bộ nhãn gold chung. |
| New fit/held interchange events | Chưa chuẩn bị | Base và donor identity, typed target, valid prefix/mask, independent source-group split, protected-fact evidence. |

**Bổ sung nhỏ nhất đề nghị, chưa thực hiện:** audit tối đa6base–donor bindings từ positive object boxes và images hiện có; gồm ít nhất hai hướng class replacement, một self-swap và một case nhiều supports phải loại/UNKNOWN. Chọn theo metadata/order trước model scores. Nếu muốn thêm attribute sanity, chỉ dùng vài development pairs cũ có correspondence rõ; không yêu cầu user chấm lại15truth labels.

Ưu tiên object identity vì nối trực tiếp mục tiêu object hallucination. Chưa cam kết đủ class variety hoặc đủ yield; nếu6bindings cũng không đặt được masks/prefixs hợp lệ thì dừng data route, không dựng thêm một dataset lớn để cứu ý tưởng. Không cần caption phản thực toàn câu, nhưng vẫn cần human audit cho tính hợp lệ của một số bindings; không gọi positive boxes là nhãn đầy đủ của mọi fact.

## 8. Pilot nhỏ với matched controls

### 8.1. Hai train arms + frozen reference

| Thành phần | Factual control F | Candidate I |
|---|---|---|
| Architecture/init/frozen blocks | Regional adapter ở§3 | Giống hệt |
| Real captions + factual labels/grounding | L_common | Cùng L_common, cùng events |
| Extra supervision slot | Factual replay CE trên base/donor với đúng target identities | Partial intervention CE trên các target events tương ứng |
| Nhãn, alias, donor images, event counts | Cùng nguồn target labels, cùng tổng exposures theo từng label/status | Không thêm labels/steps riêng cho I |
| Readout | Native fc_out factual | Cùng fc_out, factual khi inference |

Factual replay không thể giống hệt joint distribution của virtual contexts — đó chính là treatment. Phải match **label identity, valid-token event count, source exposure, optimizer steps, batching/order và loss normalization**, không chỉ gọi hai arms “cùng data” vì nhìn cùng số ảnh. Với target Protected không có donor target tương ứng thì replay trên đúng base fact; không suy một label donor chỉ để cân count. Ghi thêm số crop encodings, decoder/readout passes, wall time và peak memory; equal updates không bằng equal FLOPs. Có thể tính cả factual/patched pass rồi mask loss ở control để match compute khi chi phí cho phép; quyết định trước run.

Frozen C0 là reference không train; A chỉ historical reference. Không mở thêm một loạt arms để tìm cấu hình thắng. Nếu muốn claim vượt generic counterfactual caption training thay vì chỉ claim giá trị so factual supervision, bước confirmation sau pilot cần comparator họ See or Guess/nearest method với annotation và compute được mô tả đúng; pilot hai arms **chưa đủ claim SOTA hay đánh bại mọi prior**.

### 8.2. Quy mô và protocol trước run

Đề nghị sau feasibility: tối đa32base–donor bindings mới, dự kiến20fit/12held **theo connected source groups**, không chia random từng pair. Nếu một ảnh xuất hiện ở nhiều cặp, cả connected component thuộc một split. Nguồn held phải không lọt vào caption-adaptation fit; “held khỏi adaptation” không có nghĩa CLIP/C0 chưa từng thấy ảnh.15diagnostic cũ chỉ development và không tính vào12held.

Đây là ngân sách annotation/engineering pilot, **không power calculation hoặc lời hứa đủ train**. Khóa source IDs, classes/directions, native-token masks, eligibility, grouping, λ, LR, step count, seed, decoding/scorer và endpoint set trước user-run. Chỉ sau biết crop/dataset coverage và ngân sách **hiện tại** mới đề nghị số update/GPU estimate; chưa benchmark nên không hứa ETA. Quy mô factual caption replay có thể dùng tập train đã cho phép nhưng source-disjoint với held; không tự tái mở cohort đã đóng để tuning.

### 8.3. Endpoint cơ chế trên cùng weights

1. Exact C0 parity lúc W_o=0; native checkpoint/vocab/prompt identity; no-grad/frozen guards. Không dùng static PASS thay runtime PASS.
2. Self/identity swap phải không đổi logits trong deterministic eval. Joint permutation của proposal records phải không đổi output.
3. Valid held Updated targets: factual base đúng, donor fact đúng, rồi patched output đổi đúng; báo cả unconditional và conditional rates để không che factual failure.
4. Protected-correct và Updated+Protected joint; UNKNOWN/skip counts và coverage giữ denominator nguồn ban đầu.
5. So swapped **đúng value** với wrong-slot/unrelated-value intervention trên cùng h,π,weights. Wrong slot cần biết không phải một support khác hợp lệ; ambiguity giữ UNKNOWN. Kiểm output log-odds/actual-token changes, không chỉ attention hoặc residual norm.
6. Reader-off cùng weights và free rollout kiểm branch có ảnh hưởng. Nó là diagnostic, không suy “reader-off tốt/xấu ⇒ pixel causality” hoặc dùng nó làm một arm train mới.

Không chọn held probes chỉ vì trước đó C0 sai. Không ép probability của Protected bằng base; chỉ cần đúng semantic target. Không đòi mọi intervention tăng sensitivity: unrelated effects thấp và correct effects đúng chiều mới có ý nghĩa.

### 8.4. Caption tự sinh và tiêu chí tiếp tục/dừng

Trên **cùng ảnh tự nhiên, references, decoding và scorer**: CIDEr, CHAIRs, CHAIRi, raw false captions/mentions, object recall, word length và correct mentions. Tách full cohort khỏi proposal-eligible subgroup; subgroup không thay endpoint toàn cohort. Probe fact không được nhắc trong caption là coverage, không tự tính correct. Color accuracy không là object-CHAIR improvement.

**Tiếp tục sang confirmation chỉ khi** I có thêm giá trị so F ở held Updated/Protected joint và evidence-intervention specificity, đồng thời có tín hiệu trên caption thật so ownC0/F theo hai mục tiêu, không chủ yếu nhờ rút ngắn/bỏ noun. Báo paired deltas/intervals theo source/image groups; pilot12held không đủ xác nhận generalization hoặc significance. Gợi ý safeguard recall không giảm quá0.005absolute phải được user chốt trước run và báo uncertainty, không lấy nó làm threshold đã duyệt ở đây.

Để tránh diễn giải “có tín hiệu” sau khi xem số, **gate định hướng đề nghị khóa trước pilot** là: joint rate của I cao hơn F; Protected-correct rate không thấp hơn F; correct-value log-odds effect đúng chiều và lớn hơn matched wrong-slot effect; trên full natural cohort ΔCIDEr>0 và Δfalse-caption-count<0 so cả F lẫn ownC0, cùng recall safeguard đã chốt. Không đạt một vế thì không tự scale. Đây là gate exploratory để quyết định đầu tư, không phép kiểm chứng kết quả; confirmation vẫn phải có paired uncertainty và test độc lập. Không đặt length threshold tùy ý sau run: luôn đọc recall, correct mentions và length cùng nhau để phân biệt giảm lỗi với bỏ nội dung.

**Dừng hoặc rút luận điểm nếu:**

- Không tạo được labels/prefixs hợp lệ ở feasibility; không đủ coverage/variety mà không vượt effort đã duyệt.
- F đạt ngang I: chưa có bằng chứng cần intervention term, dù cả hai có thể hơn C0 nhờ grounded adapter.
- I chỉ thắng virtual probe nhưng không có lợi ích free captions; hoặc reader/value interventions không có đúng tác động dự đoán.
- Chỉ tốt hơn về màu, chỉ giảm CHAIRi nhờ mẫu số, hoặc giảm false captions nhờ bỏ nội dung/recall.
- Nearest full method cho thấy cùng partial target/readout mechanism đã có: không nhận novelty bằng đổi tên/backbone.

Không mở rộng epochs/λ/seeds trên cùng cohort để cứu claim. Nếu pilot có tín hiệu, muốn viết claim tăng CIDEr **và** giảm hallucination cần protocol confirmation riêng, test độc lập và uncertainty; image bootstrap một seed không giải quyết variability giữa seeds. Thất bại một cấu hình không chứng minh cả họ phương pháp bất khả thi.

## 9. Tự phản biện trước khi user duyệt

| Phản biện | Hệ quả đối với quyết định |
|---|---|
| Partial masks có thể là kỹ thuật chuẩn, chưa đủ delta so IIT | Rủi ro novelty cao; phải đọc full nearest interventional generation trước claim. Không coi “caption task mới” là đóng góp. |
| Donor crop thật không bảo đảm virtual swap có nghĩa cho mọi context | Chỉ dùng partial variables/prefixs đã kiểm; count/relation/identity-dependent Protected có thể phải UNKNOWN. Yield thấp là một kết quả feasibility có thể dẫn tới dừng. |
| Frozen global hidden/prompt có thể phản đối donor, hoặc donor artifacts thắng | Giữ đường global cố định là cách đặt câu hỏi về channel, không bảo đảm learning ổn; đánh giá natural captions và source-held donors bắt buộc. |
| Loss thắng vì memorization hoặc tăng residual quá mạnh | Không nhận theoretical causal abstraction; held class/source combinations và natural endpoint phải kiểm. Không tự thêm một chuỗi regularizers để cứu. |
| Unique-binding pilot không xử lý multi-instance thực tế | Báo giới hạn coverage. Việc các nguồn trước đã có set-valued positives là prior, không nhận mình giải quyết nó bằng skip. |
| Teacher forcing khác free generation | Test actual rollout; correct probe không đồng nghĩa caption correct hoặc dùng đúng noun. |

**Mức tin cậy:** cao cho việc forward có một đường local→native-token tường minh; vừa cho khả năng xây một số bindings hợp lệ; thấp cho novelty clearance và chưa biết utility. Không có kết quả mô hình mới để nâng các mức này.

## 10. Đoạn7câu để trình thầy

“Em đề xuất học căn chỉnh giữa vùng ảnh và từ trong chính đường sinh caption, trên mô hình nền có prompt đã được huấn luyện. Phương pháp kế thừa grounded generation và interchange intervention training: một nhánh bằng chứng vùng tác động trực tiếp vào hidden state trước đầu ra từ vựng của decoder. Cải tiến dự kiến là chỉ gán những mục tiêu phản thực xác định được khi hoán đổi bằng chứng một vùng — từ cần đổi, fact có thể giữ và phần chưa biết — thay vì tạo caption phản thực toàn câu. Ví dụ khi đổi bằng chứng của một đối tượng sang donor đã xác minh, output của đúng vị trí từ phải phản ứng, trong khi fact độc lập của vùng khác vẫn đúng. Em sẽ dùng positive boxes và captions hiện có, kiểm khả thi một số binding nhỏ rồi so với cùng kiến trúc được huấn luyện factual bằng đúng lượng nhãn. Thực nghiệm sẽ kiểm việc thay value thật sự tác động đúng token và đo caption tự sinh bằng CIDEr, CHAIR, recall cùng độ dài. Đây là ứng viên cải tiến đang cần đối chiếu thêm prior và kiểm nghiệm, chưa phải kết quả tăng chất lượng hoặc một phương pháp đã được xác nhận mới.”

## 11. Bàn giao và phần chưa làm

Đề nghị Codex1 phản biện **ba điểm cụ thể** trước triển khai: (1) Updated/Protected rules có tạo target mâu thuẫn khi prefix/global state giữ nguyên không; (2) factual replay có match đúng target/source exposures không; (3) partial transition construction có technical delta đủ bảo vệ so full IIT/interventional captioning priors không. Nếu (3) không đứng vững, không viết introduction nhận phương pháp mới rồi mới tìm nguồn sau.

Hai blocker nguồn chính: full paper mechanistic-counterfactual2026 và full-method comparison với counterfactual grounded generation/partial supervision; See or Guess ở lượt này mới code scope. IIT và HALC full method đã đọc được, không cần user gửi lại hai nguồn đó. Blocker dữ liệu: current payload/crop/proposal coverage và label eligibility chưa audit cho pilot mới. Các blocker này ngăn claim novelty/run readiness, **không ngăn việc đã chọn và bàn giao một phương án cụ thể để quyết định**.

Chỉ sửa tài liệu nghiên cứu/trạng thái/bàn giao. Không viết trainer, stage dữ liệu/crops, chạy Torch/autograd/mô hình/notebook/Kaggle/official scorer, sửa báo cáo cũ hoặc ghi sheet “completed”. Không có budget/run mới được coi là đã duyệt. Kết quả stdlib toy chỉ kiểm đại số, không là validation mô hình. Lượt này cần commit/push memo theo yêu cầu user; tình trạng push sẽ được báo bằng kết quả Git thực tế.
