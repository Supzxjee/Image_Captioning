# Hai ứng viên sau I/J: chọn sửa xác suất khái niệm theo bằng chứng ảnh

> Cập nhật trước train29/10: chính sách negative ở §3.2 được thay bởi [amendment GT/ref weak-negative + detector-veto-only](CG_NEGATIVE_POLICY_AMENDMENT_2026-10-29.md). Detector completeness chưa xác minh; không dùng detector absence làm negative evidence. Các phần model/objective còn lại giữ thiết kế.

29/10/2026. Research proposal, chưa code/launch GPU. I/J và frame200 giữ quyết định dừng. N đã nêu trước đây là control/chẩn đoán, chưa tự thành đóng góp; không yêu cầu chạy N riêng trước phương pháp mới.

## 1. Điều bằng chứng cũ cho phép và không cho phép

- J−I CIDEr +.0983 có CI dương nhưng J−A −.0713 có CI âm; mixed panel chưa có semantic compensation. Chưa biết lỗi chính là coadaptation, objective hay thiếu thông tin.
- R bbox routing chỉ hơn A +.00655, 95% CI[-.015988,+.029683]; R−B cũng chưa rõ. Không dùng trung bình R để gọi routing đúng vùng đã thắng.
- Frozen GT-max-Q T/W diagnostic không tăng noun utility. Nó phản bác routing teacher cụ thể ấy, chưa phản bác mọi local evidence.
- A đã có CIDEr+.5object-count reward. Vì vậy “thêm phạt noun sai” đơn thuần không phải ý tưởng mới; cần một đơn vị và cách cập nhật khác reward toàn câu.
- VSR 256 pairs quá hẹp để mặc nhiên kỳ vọng cải thiện CIDEr/CHAIR toàn cục. Dữ liệu/CLIP có thể không chứa đủ chi tiết để quyết định predicate; thay loss không tạo ra pixels còn thiếu.

Chọn thu hẹp về **object-class grounding**, đúng với mục tiêu CIDEr và CHAIR; không tiếp tục nhận một đóng góp relation-binding chưa đo được.

## 2. Hai ứng viên khác nhau thực chất

| | Ứng viên 1: học sửa semantic-class probability mass | Ứng viên 2: chọn caption bằng factual evidence |
|---|---|---|
| Vấn đề | Reward cả câu không chỉ rõ khối xác suất của quyết định khái niệm nào cần sửa; phạt một cách gọi có thể chuyển sang synonym hoặc bỏ nội dung | Finetune làm thay đổi captioner mạnh; có thể giữ A frozen và chọn câu có bằng chứng tốt hơn |
| Cơ chế | Trên caption lấy mẫu, chọn object-class chưa được hỗ trợ theo ảnh; giảm tổng probability của các alias cùng class tại prefix ấy, giữ class mass của reference spans có GT support | A tạo K=5 candidate captions; verifier ghép noun facts vào các box và relation facts vào cặp; rerank cùng fluency score, giữ các fact được hỗ trợ của caption gốc |
| Supervision | COCO TRAIN boxes/classes+5refs+detector-cache veto, aliases và sampled captions; không dùng held panel | COCO TRAIN và VSR true/false cho fact verifier; inference chỉ predicted boxes/features và caption facts |
| Điểm mạnh | Native A giữ giao diện; gradient đi thẳng vào quyết định caption; không thêm inference model | Tránh làm drift weights A; đơn vị kiểm là fact/caption |
| Giới hạn | Absence label nhiễu; có thể học noun blacklist/giảm recall; cần broad caption TRAIN | Candidate set có thể không có câu vừa đúng vừa CIDEr tốt; verifier/parse domain shift; thêm latency và calibration |
| Test nhỏ | Hai arm native C/G cùng recipe, chỉ G có negative class-mass correction; A fixed | Cùng candidate set, chọn bằng model likelihood vs thêm fact score; A greedy luôn được giữ làm mốc |
| Khả năng đóng góp | Một objective căn chỉnh output mass với supported/unsupported concept sets, kèm preservation, là adaptation có nội dung | Một pipeline evidence-constrained candidate selection; các primitive reranking/verification đã có prior |

**Chọn ứng viên 1.** Nó không cần một head phụ đạt điểm rồi hy vọng decoder sẽ dùng; tránh thêm đường memory chưa khớp A và trực tiếp nhắm lỗi CHAIR. Ứng viên 2 cần verifier đáng tin và candidate headroom mà hiện chưa có bằng chứng, nên không chọn làm bước đầu. Không nhận một trong hai là “đầu tiên”.

## 3. Luận điểm phương pháp được chọn

> Căn chỉnh ngữ nghĩa–thị giác ở mức khái niệm đối tượng: giảm khối xác suất của một nhóm cách gọi không được hỗ trợ trong ảnh tại prefix mô hình thực sự gặp, đồng thời giữ khối xác suất của nhóm ấy ở những ngữ cảnh có bằng chứng đúng.

Đây là thay đổi **đơn vị gán trách nhiệm**: từ một scalar advantage cho cả chuỗi sang một decision class gồm nhiều cách diễn đạt, được chọn theo ảnh. SCST toàn câu là policy-gradient hợp lệ; chưa đo được nó “sai” hoặc có credit dilution trong A. Tính hữu ích của tín hiệu cục bộ là giả thuyết cần test.

Không bắt mỗi từ ứng với một bbox. GT boxes cung cấp weak class-presence support; nhiều box cùng class tạo một support set. Phương pháp không định vị đúng instance hoặc quan hệ bằng công thức này.

Ba chặng được tách rõ:

| Chặng | Vai trò ở ứng viên này |
|---|---|
| Tìm vùng | Box/class annotations và detector hints là nguồn support; không tối ưu box prediction hoặc chứng minh ảnh hưởng từ đúng vùng |
| Biểu diễn ý nghĩa | Object class là tập biểu thức đồng nghĩa/multiword, không phải một token hoặc một instance cố định |
| Decoder sử dụng | Loss tác động vào class probability trong decoder thật tại generation prefix; kết quả vẫn phải kiểm bằng caption tự do |

G/S trước đây sửa logits theo object evidence trên nền A cố định và đã không cải tiến. Ở đây không đặt vocabulary bias tại inference: học các quyết định theo prefix, có positive preservation và policy objective. Khác biệt ấy là động cơ kiểm nghiệm, chưa chứng minh sẽ tránh được thất bại G/S.

### 3.1 Giữ nguyên model, không sửa logits inference

A memory/prompt/Q-Former/visual projection/vocabulary và đường decode giữ nguyên. Không có ROI reader, gate, classifier hay semantic prompt mới. Học hai decoder layer cuối của native A; các phần khác frozen. GT/detector flags chỉ chọn training events, không đưa vào forward hoặc inference.

Lấy ngay caption được sample trong SCST. Stop-gradient với thao tác chọn chuỗi, span và nhãn; rescoring qua decoder thật để lấy gradient. Dùng longest-match aliases/token positions đã kiểm trong dự án, giữ nguyên whitespace tokenizer; trường hợp không xác định được danh từ/class thì unknown.

### 3.2 Ba mức supervision theo ảnh

Với class c:

- **Supported:** có GT class/box hoặc được refs nhắc. Positive preservation anchors dùng giao của reference noun spans và GT class support để giảm nhiễu.
- **Weakly unsupported:** caption sample nhắc c nhưng c không có trong GT và toàn bộ refs, đồng thời detector cache đầy đủ không có prediction c vượt ngưỡng veto khóa trước, đề nghị .5.
- **Unknown:** bất kỳ hint support/ambiguity, thiếu detector record, hoặc không biết cache có bị truncation trước khi xét class hay không. Không dùng negative loss.

Chỉ xét class thực sự được caption sample nhắc; không phạt toàn bộ các class không xuất hiện. **Vắng trong top 12 không đủ để coi negative**; cần xét full cached detector record trước topK. Record rỗng/thiếu không tự chứng minh vắng mọi vật.

Triple absence vẫn không phải nhãn “chắc chắn không tồn tại”: annotation/detector đều có thể bỏ sót vật nhỏ/che khuất. Gọi đúng là supervision nhiễu theo phạm vi COCO 80. Không tạo nhãn phủ định cho predicate hay class ngoài phạm vi ấy.

### 3.3 Xác suất của khái niệm

Với prefix h và alias set \(\mathcal A_c\), định nghĩa sự kiện “next phrase là một alias hợp lệ của c”:

\[
q_\theta(c\mid h,I)
 =P_\theta\!\left(\bigcup_{a\in\mathcal A_c}\{\text{next tokens begin with }a\}\mid h,I\right).
\]

Tính bằng trie/frontier prefix-free của alias sequences, deduplicate prefix-overlapping events. Không cộng chồng xác suất “a” và “a b” nếu cùng semantic event; không dùng geometric-mean/length normalization rồi gọi kết quả là probability. Mỗi phrase event dùng đúng tokenizer và word boundaries. Alias mơ hồ về nghĩa bị bỏ khỏi loss thay vì tự gán noun class.

Với weak-negative mention c ở prefix sample:

\[
L_{\rm neg}
 =\operatorname{mean}_{(h,c)\in E^-}
 -\log[1-q_\theta(c\mid h,I)].
\]

Đích phạt là cả concept mass, không riêng bề mặt “sofa” rồi để mô hình chuyển thành “couch”. Việc tính union và unlikelihood không phải toán mới.

Với supported reference anchors \(E^+\), teacher A frozen:

\[
L_{\rm keep}
 =\operatorname{mean}_{(h,c)\in E^+}
 [\log q_A(c\mid h,I)-\log q_\theta(c\mid h,I)]_+.
\]

Đây là neo bảo toàn khái niệm đúng, không tăng xác suất mọi vật có trong ảnh dưới mọi prefix. Reference chứa noun trong ngữ cảnh hợp lệ; không thay noun sai bằng một class bất kỳ có trong ảnh rồi coi toàn câu mới là đúng. L_keep không bảo đảm object recall khi sinh tự do, vẫn cần đo.

### 3.4 Objective và hai arm

Chung cho hai arm:

\[
L_{\rm common}
 =L_{\rm SCST,\ reward\ gốc\ A}
 +.1KL_{\rm rollout}(p_A\|p_\theta)
 +.1L_{\rm keep}.
\]

KL tính trên prefix caption sample của chính policy, teacher A nhận cùng prefix/ảnh, token-mean; không chỉ reference prefixes. Đây là thành phần chung, chưa quy riêng gain cho nó.

- **C:** \(L_{\rm common}\).
- **G:** \(L_{\rm common}+.1L_{\rm neg}\).

A fixed là mốc gốc. Learning scope/optimizer/LR/clip/data schedule/seed/cap giống nhau, LR decoder đề nghị 5e−6 như I/J; không thêm warmup hoặc adapter. Empty event sets trả 0 và log counts. Các hệ số là starting recipe được chốt trước, không lý thuyết tối ưu.

C/G dùng cùng sampling rule/RNG initialization, không đòi caption sample và negative count sau train giống hệt khi policy đã khác. G cũng có thể thay distribution của events; phải report exposure, usable negative count, class coverage, L_keep và gradient contribution.

**Vai trò N được gộp vào C:** C là native-continuation control cho recipe mới. Không chạy N-VSR cũ rồi mới C/G. Vì thí nghiệm chuyển sang on-policy object grounding và caption data rộng, C không được gọi là bản sao N dùng 256 VSR pairs.

C/G so sánh cô lập negative class-mass correction trên common preservation recipe. Nó chưa tách được riêng lợi ích alias-set, unknown-veto và L_keep; nếu muốn claim từng thành phần tốt hơn token/span UL thì cần ablation tiếp, không tuyên bố từ hai arm này.

## 4. Vì sao có thể giúp cả CIDEr và CHAIR?

- Local negative gradient giảm xác suất noun concept đang tạo CHAIR errors, thay vì ép cả câu tốt xấu cùng một hướng.
- Positive anchors và SCST CIDEr tạo áp lực giữ nội dung có ích; native interface tránh representation mismatch do reader mới.
- Semantic-class mass giảm cơ hội đổi synonym để né một penalty bề mặt.

Đây là cơ chế kỳ vọng, chưa là guarantee. UL vẫn có thể chuyển mass sang EOS/từ chung, làm caption ngắn và CIDEr/recall giảm. Nếu chỉ CHAIR giảm do né noun thì thất bại theo mục tiêu user. Nếu C/G đều kém A thì dừng recipe mới; G ít kém hơn C không được gọi là đạt mục tiêu như cách I/J đã không đạt.

## 5. Phép thử nhỏ đủ quyết định

### Nguồn dữ liệu và cap

Ưu tiên **TRAIN 10k đã dùng cho B/R**, COCO box/span manifest 17,904 spans và original A caches; caption nguồn rộng thay vì lặp 8 epochs trên 484 images. Không trộn VSR templates vào pilot này. Không dùng 175 mixed-panel labels, 64 held source labels hoặc dev GT để huấn luyện.

Full caches của TRAIN 10k từng được dùng không có nghĩa đã staged ở v105. Root kiểm input mounts/full detector coverage một lần trong preflight; không tạo audit/cohort mới. Nếu chỉ có 484 TRAIN images thì không tự thay vào và vẫn gọi là kiểm nghiệm hướng broad-caption này.

Một seed 42, global batch 8, endpoint 500 updates/arm; sampler trên nguồn 10k, khoảng 4000 caption-image exposures trước lặp epoch, báo IDs/order thực. Cùng references schedule và greedy decoding A. Không best-checkpoint hoặc lambda search. Có thể chỉ stage các rows thực dùng và dev theo schedule đã khóa; không cần image extraction/crop mới.

Preflight chỉ kiểm kỹ thuật cần có: baseline parity, class-event probability hợp lệ/không doublecount, longest-match masks, unknown abstention, finite gradients và global reduction trên 2 GPU. Đây không phải một round research audit trước train.

### Endpoint và falsification

A/C/G trên cùng dev 1000 đã dùng, caption metrics/CHAIRs/i/absolute hallucinated captions/object recall/length; paired image CI cho G−A và G−C, report one-seed/reused-dev limits. GT chỉ ở evaluator.

Báo thêm class-conditioned diagnostic trên cùng dev: ở cùng một prefix trung tính cố định, so concept mass của mỗi class giữa ảnh có GT support và ảnh weak-negative. Chọn tất cả class đủ mẫu theo quy tắc khóa trước, macro-average; không chọn class sau khi thấy G thắng. Giảm mass cả supported lẫn unsupported không là bằng chứng tốt về visual conditioning. Diagnostic này cũng chỉ có weak labels, không phải causal grounding.

- **Continue để xác nhận:** G có điểm tốt hơn A về cả CIDEr và CHAIR, cũng có lợi ích so C, recall không giảm quá .005 absolute; effect/counts và CI phải được báo. Nếu CI rộng chỉ gọi tín hiệu exploratory, chưa kết luận cải thiện đã xác nhận.
- **Dừng:** G không thắng A cả hai mục tiêu; hoặc CHAIR giảm nhờ mất nội dung/recall; hoặc G≈C và negative loss không có giá trị riêng; hoặc valid negatives quá ít để objective thực sự hoạt động. Không hạ veto threshold theo dev để kiếm thêm mẫu.
- Muốn viết claim cải thiện: cần bằng chứng xác nhận cho cả hai trục, không chỉ scalar tốt nhất. Một seed/ảnh-dev bootstrap không chứng minh seed robustness; ưu tiên xác nhận hơn thêm mô hình nếu pilot có tín hiệu.

Không cam kết GPU-hours/accepted 31/10; quota hiện tại chưa xác minh. Đây là hai arm trong một phép thử phương pháp, không chuỗi diagnostic N→C→G.

## 6. Phần kế thừa, delta và kiểm prior có giới hạn

Đã kiểm tập trung ba nguồn, không mở thêm vòng 20 papers:

- [Welleck et al., Neural Text Generation with Unlikelihood Training, ICLR2020](https://iclr.cc/virtual_2020/poster_SJeYe0NtvH.html): UL ở token/sequence đã có; kiểm abstract và author repo, không nhận là loss mới.
- [Fei & Huang, Incorporating Unlikely Negative Cues for Distinctive Image Captioning, IJCAI2023](https://www.ijcai.org/proceedings/2023/0083.pdf): đọc §2–3/Eq.2,5,9; negative teacher cho generic captions và soft UL đã có trong image captioning.
- [Enhancing Faithfulness in Abstractive Summarization via Span-Level Fine-Tuning, 2025 preprint](https://arxiv.org/abs/2510.09915): mới kiểm abstract, đủ để loại claim span-hallucination fine-tuning là ý tưởng mới; chưa so toàn bộ công thức.

**Không nhận:** UL, on-policy sampling, aliases, positive preservation, abstention hay SCST là phát minh riêng.

**Ứng viên delta cần bảo vệ:** cách tổ chức weak visual support thành supported/unsupported/unknown để sửa semantic-class event mass tại generation prefix, cùng positive-class preservation trên native captioner, và kiểm dual-goal với matched control. Đây là một cải tiến objective ở mức ứng dụng có luận điểm rõ hơn N chung. Chưa xác minh không có công trình trùng cả thiết kế; không dùng tìm kiếm hữu hạn làm chứng nhận novelty.

Đối chứng C/G chưa đủ chứng minh “class-set tốt hơn mọi span-UL prior”. Nếu kết quả dương, phải mô tả đúng phạm vi hoặc thêm ablation nhắm claim ấy, không đổi tên UL thành thuật toán nền tảng mới.

## 7. Hai rủi ro quyết định

1. **Ground-truth noise/metric shortcut:** missing annotations và detector false negatives có thể phạt vật thật; cùng aliases dùng cho train/eval có thể tối ưu CHAIR mà không cải thiện physical truth. Unknown veto, positive anchors và recall là guards, không chứng nhận không lỗi. Chuyển sang noun ngoài 80 class hoặc từ chung có thể né metric; phương pháp cũng không giải quyết gender/thuộc tính chi tiết bằng class labels.
2. **Không đủ thông tin/headroom:** A chỉ có 52/1000 CHAIR-flagged captions; số weak-negative events có thể ít. CLIP toàn ảnh/low resolution có thể không phân biệt được vật nhỏ, vậy loss có thể dạy im lặng thay vì nhìn đúng. Không hứa thêm objective sẽ đồng thời tăng CIDEr và giảm hallucination.

**Chốt:** chọn concept-mass correction với preservation, gộp native-A control trong cùng C/G trial. Đây là ứng viên phương pháp mới để kiểm, không là kết quả đã thành công và không là cách đổi nhãn I/J.
