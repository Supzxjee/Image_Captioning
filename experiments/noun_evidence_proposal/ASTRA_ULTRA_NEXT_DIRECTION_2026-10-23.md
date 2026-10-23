# Astra Ultra: quyết định hướng nghiên cứu sau các kết quả G/S

Đức Anh vừa chọn Ultra cho bạn và yêu cầu một lượt hỏi thực sự có ích cho hướng nghiên cứu.
Hãy dùng mức suy luận người dùng đã chọn. Bạn định hướng nghiên cứu; root sẽ kiểm tra giả định,
phản biện và triển khai. Lượt này **chưa viết code, chưa khởi chạy thí nghiệm**.

## Câu hỏi trung tâm

**Có thay đổi nào đủ thực chất về bài toán học, nguồn thông tin, biểu diễn hoặc đường sử dụng
bằng chứng để một phương pháp căn chỉnh thị giác–ngôn ngữ có cơ hội cải thiện caption hơn A?
Chọn một hướng có thể kiểm chứng với nguồn lực hiện có, hoặc một phép đo duy nhất đủ quyết
định có nên đầu tư hướng đó.**

Bạn vừa khuyến nghị dừng G/S và các biến thể gần kề. Hãy giữ kết luận thực nghiệm đó, nhưng
đánh giá lại chiến lược ở mức rộng hơn. Nhiệm vụ không chỉ là nhắc lại “giữ A/dừng”: người
dùng vẫn cần quyết định tiếp theo cho đề tài. Bạn không bị buộc phải giữ CSA, UPT, CLIP,
frozen A, word-to-box, object/relation prompts, gate hay residual. Cũng không phải cố nghĩ
ra phương pháp mới cho có. Nếu chưa đủ cơ sở chọn phương pháp, cần chỉ rõ thông tin nào
thực sự sẽ đổi quyết định và cách lấy nó với chi phí hợp lý.

Các quyết định cũ bên dưới là lịch sử, không phải yêu cầu phải bảo vệ. Tách bằng chứng,
giả thuyết và khoảng trống thông tin; nêu phản ví dụ cho chẩn đoán bạn ưu tiên.

## Mục tiêu và tài nguyên

Đề tài là căn chỉnh ngữ nghĩa–thị giác trong image captioning. Ba hướng đóng góp từng bàn:
prompt đối tượng–quan hệ, cơ chế căn chỉnh, sinh/chọn nhiều caption. Không bắt buộc đủ cả ba.
Cần cơ chế gắn nội dung ngôn ngữ với bằng chứng ảnh và kết quả có ích cho caption, không chỉ
matcher score/attention mass/sensitivity tăng. Không tự gán tính mới cho một module ghép lại.

Dự án **đã có gain so Original QFormer** nhờ SCST+CE+thưởng đối tượng. Việc còn thiếu là
cơ chế căn chỉnh tạo gain thêm trên nền mạnh; đừng mô tả dự án là chưa từng tăng baseline.
TRAIN có reference captions COCO; inference trên ảnh mới không có GT caption/box/quan hệ.
Không mặc định yêu cầu zero-shot vì các recipe hiện tại đã dùng TRAIN references.

- Baseline: frozen CLIP ViT-B/16 cache,197×768 (CLS+196patches), resized224vuông bilinear,
  no center crop; cache FP16 đọc FP32. CLIP patches đã contextualized, không phải object crop độc lập.
- Projection768→512; QFormer2layers/32queries. Legacy CLIP-text prompt branch có sẵn,
  prompt→visualattention/gate; decoder đọc grounded legacy tokens và32queries.
  A không có nhánh UPT mới **không đồng nghĩa A hoàn toàn không có text prompt/căn chỉnh**.
- A là endpoint continuation từ checkpoint SCST+CE+object0.5 mạnh, không phải original CE.
  Exact A SHA `630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901`.
  Tensor digest bare base `d2796e30b4e7454798fac5c513be714dca69918b423910233aee95b161e0e81e`.
- Có COCO caption/instances, detection cache, visual/legacy cache, UPT full-COCO prediction
  và all-action prompt cache; UPT hidden-pair cache train10k/dev1000 tối đa8slots×1024.
  UPT là dự đoán, không GT. V-COCO chỉ cover một phần COCO, action=0 không tự là phủ định
  ngôn ngữ; hold/carry hay ride/hold có thể cùng đúng.
- Có VG region/relationship annotations nhưng noise và overlap; từng thử chúng bên dưới.
  Có ít nhãn human về contrast/cặp/vùng; không mặc định nhờ người dùng chấm thêm nhiều ảnh.
- Kaggle: một notebook dùng2T4; có thể hai notebook trên hai tài khoản. Quota GPU còn lại
  chưa xác nhận, đừng hứa giờ chạy. Có thể yêu cầu dữ liệu/encoder/checkpoint khác nếu thực
  sự cần, nhưng giải thích lợi ích, dung lượng/chi phí và điều kiện khả thi. Không chạy model local.

## Lịch sử tối thiểu để tránh đề xuất lại các thử nghiệm

**Không so trực tiếp các điểm khác cohort/IDF/decoding. Những pilot âm không chứng minh cả
họ phương pháp bất khả thi. Có thể kế thừa một thành phần cũ nếu thay đổi giả định thực chất.**

1. Full-train SCST+CE+object0.5: cùng beam5/fixed rerank/test5000, Original CIDEr1.221815→
   1.247342, Δ+.025526, paired95[+.012446,+.038868]; CHAIRs3.48→3.44%,174→172captions.
   Raw rank0 cùng candidate evaluation:1.188245→1.212024,CHAIRs4.32→3.98%.
   Test đã từng xem; một seed; không tách độc lập đóng góp objectreward.
2. Learned global/regional ranker và gain guard đã thử. Ví dụ val512 cùng candidatepool:
   fixed1.263698,regional1.251104; hallucaptions16→13 nhưng ΔCIDEr−.012595,CI chứa0.
   Đề xuất rerank mới cần xác định candidate headroom/supervision/cơ chế khác, không chỉ “dùng CLIP chấm câu”.
3. Cùng dev1000 rawgreedy, A1.205513; UPT visualB1.199067,visual+textC1.192828,
   text-conditioned gateD1.202747. E: text-query/visual-value S/O/R evidence memory,
   restricted bbox/union attention + noisy contrastive loss:1.196816. F dùng verified
   same-image V-COCO matcher supervision:1.198510,off1.197899; chưa vượt A.
4. Diagnostic F20: chỉ đổi predicate text, giữ S/O/GTboxes; attention và evidence tokens đổi
   nhưng greedy captions0/20 khác; likelihood response nhỏ. Không chứng minh decoder hoàn
   toàn bỏ qua evidence; target relation có thể không salient cho toàn ảnh.
5. Human-reviewed phrase CE vs CE+CROSS:14fit/7holdoutgroups,holdout14events. Cả hai
   supplied-phrase preference10/14→13/14; CROSS tăng margin nhưng không tăng quyết định đúng;
   greedy template exact5/14 cảhai (không phải semanticaccuracy). Paircrop so fullframe,
   humanreview16events: đúng/cụthể10→11,nhóm hai chiều đúng2/8→3/8,surfboard0/4cảhai.
   Crop-only chưa đạt gate. Các tập rất nhỏ, không phải full-caption benchmark.
6. H thêm pretrained UPT hidden-pair memory:1.201124,off1.202134 vs A1.205513 dev1000;
   on/off48captions đổi nhưng không caption gain. Nhánh không bị ngắt hoàn toàn.
7. VG regionphrase auxCE: regionG1.208401,allpatchT1.185747,GV chặn aux gradient trực tiếp
   vào decoder1.193220 vs A1.205513 dev1000. CI cácgain không qua gate. Không kết luận
   mọi crop/VG vô ích hoặc đã chứng minh decoder gradient là nguyên nhân duy nhất.
8. CSA-inspired two-phase core (không full reproduction):1250coreCE/basefrozen rồi1250joint
   CE/SCST;core23Mparams. CSAon1.199998,off1.212948 vs A1.205513 dev1000,CI chứa0;
   chưa đạt +.01/positiveCI. Off hậu nghiệm không phải nhánh thắng được chọn trước.
9. COCO noun→patch routing B/R,không thêm inferenceparams: B all-spatial-path1.203607,
   R bbox-path1.212063 vs A1.205513 dev1000. R−A+.006550 CI[−.015988,+.029683];
   R−B+.008455 CI[−.011348,+.028581]. B/R đều50CHAIRflags vs A52. Chưa có caption/spatial gain rõ.

## Kết quả mới nhất: noun-evidence G/S đã audit đầy đủ

Tên G ở đoạn này khác VG-regionG cũ. Cả hai mới dùng **frozen A endpoint**, cùng head108752params:
196patches×768→LayerNorm(noaffine)→Linear128→GELU→Linear80.
g=global logmeanexp patch/class scores; d=clip(g−logit TRAINprior,−2,+2).
Thêm .5d vào413raw unambiguous noun vocabulary IDs, prefix-only compound guard hot+dog.
Không prefix-dependent learned selection, không bbox inference. Cùng correction trong CE,
sampled rollout,greedy rollout,sampled teacher rescoring và final generation.
Zero-init final weights/priorbias làm initial correction0; initial A parity verified.

Presence80classes positive=annotation OR trustedTRAINcaption; masked ambiguity;
remaining weaknegative. Plain observed-pair BCE; prior smoothed image/class counts trên toàntrain10k.
5602unique regions (1257person) từ17904retained noun spans, dedup byimage/annotation/class.
Regionpositive+mean weaknegative softplus; global mean overregions. **G** pool toàn196patches
cho regionloss, **S** pool fractionbbox. Mọi thành phần còn lại giống nhau.

Mỗi arm2500updates/twoepochs train10k,batch8,seed42,AdamW head1e−4/WD.01/clip1.
Loss CE+.1presence+.1region; sau200updates thêm .05SCST(CIDErD+.5object_count_v2).
Endpoint only, rawgreedy, **reused val5000**,no test. Trước đâyA1.205513 làdev1000 IDF;
dưới đâyA chấm cùngval5000, không trộn hai protocol.

| Measure | A | G | S |
|---|---:|---:|---:|
| CIDEr |1.1822477755|1.1721037326|1.1715728517|
| CHAIRs |.0582|.0752|.0712|
| CHAIRi |.0378265155|.0465933037|.0448945747|
| Recall |.466034656|.479658463|.475640382|
| CHAIR flagged captions |291|376|356|
| Hallucinated mentions |307|398|379|
| All object mentions |8116|8542|8442|

20kpaired image-bootstrap,seed20261023:
G−A−.010144043,95CI[−.018156071,−.002270984];
S−A−.010674924,95CI[−.018133493,−.003335472];
S−G−.000530881,95CI[−.003947181,+.002888130].
G/S captions chỉ khác253/5000;348CHAIRflags chung,28G-only,8S-only.
So A: G thêm121flags,sửa36;S thêm99,sửa34.
Last100TRAIN saturation80classscores G40.33%,S20.81%;regionlossfirst/last100 G2.818→.845,
S2.829→.561. Minibatch means không phải heldout recognition accuracy.

AuditPASS: cùng source,hyperparameters,staged tensor bytes,labels/reward,sampler actualorder,
runtime/optimizer,initialhead; cảhai preflight/2500updatesCOMPLETE,Abasefrozenunchanged.
A5000 predictions/perimageCIDEr/CHAIR khớp tuyệt đối giữaG/S;olddev1000 A captions0mismatch.
Count/loss reconstruction và các metric/CI tái lập. Headbytes không có trongZIP; không giả vờ
đã recompute checkpoint tensors độc lập. Không humanvisualreviewmới.

G/S FAIL gate đã khóa. S inferencecontrols TRAINmean/shuffle đúngquytắc bị bỏ qua sau S/Afail.
Giữ A,dừng cấu hình này. Failure không riêng bboxpooling; chưa xác định responsible commoncomponent.
Không được gọi prior/captionreward/CLIP/decoder là nguyên nhân đã chứng minh.

## Đầu ra cần cho một lượt Ultra này

1. **Kiểm toán chiến lược:** chỉ ra tối đa3giả định/ràng buộc đang có khả năng hạn chế nghiên cứu
   (ví dụ cách xác định semantic supervision, budget/staging,representation,objective); mỗi điểm
   ghi bằng chứng, phản ví dụ và điều chưa biết. Không cần bảo vệ những proposal trước của bạn.
2. **So sánh tối đa3hướng khác nhau thực chất**, rồi chọn1. Với mỗi hướng: tín hiệu nào bổ sung/
   giữ lại trước đây chưa có, vì sao nó có thể thành caption gain, priorart nào gần nhất, và
   failure mode dễ xảy ra. Không đổi tên gate/fusion/ranker cũ thành “mới”. Có thể tái dùng nếu
   giải thích thay đổi then chốt và đâu là falsifiable hypothesis.
3. **Hướng chọn phải đủ để root triển khai:** sơ đồ bằng chữ/tensorroles, phần frozen/trainable,
   loss viết cụ thể (positive/negative/mask/normalization), nguồn supervision và split mapping,
   inference trên ảnh mới lấy thông tin từ đâu, cách ngăn copy GT/text shortcut và yếu tố salience.
   Làm rõ đóng góp visual–semantic alignment nằm ở đâu, không chỉ object reward hay proxy.
4. **Smallest decisive experiment:** một pilot/chẩn đoán ít nhánh, đối chứng thực sự cần,
   matchedbudget/initialization, outputmetric và go/stop đặt trước. Có contingency cho cả
   kết quả positive/negative; phân biệt phát hiện tín hiệu với chứng minh full-caption gain.
   Phản biện xem1250–2500updates/nhãn hiện có có đủ để bác bỏ phương pháp không. Có thể đề nghị
   tiêu chí tương lai khác với lý do khoa học, nhưng không sửa gate G/S để gọi kết quả cũ thắng.
5. **Tính khả thi và tự phản biện:** assets dùng lại/thiếu gì, chi phí Kaggle và annotation,
   rủi ro lớn nhất, một điều kiện khiến bạn tự bỏ hướng chọn. Nếu không đủ cơ sở chọn module,
   thay bằng một phép đo mà hai khả năng kết quả sẽ dẫn tới hai quyết định nghiên cứu khác nhau;
   không thêm diagnostic chỉ để giải thích loss trong khi quyết định không đổi.

Hãy trình bày quyết định rõ đầu tiên, sau đó giải thích. Không hứa vượt A, không kết luận
mọi alignment đã hết đường, và cũng không ép phải nặn ra một kiến trúc để lấp chỗ trống.
Nếu nên thu hẹp/đổi phạm vi đóng góp luận văn, nói cụ thể phạm vi nào còn bảo vệ được và
phần nào chỉ là kết quả âm, không biến benchmark có sẵn thành đóng góp mới.

Để tiết kiệm quota: **không mở lại ZIP/repo/Sheet để xác nhận các số root đã audit**, không
tóm tắt toàn bộ lịch sử một lần nữa. Chỉ tra cứu1–3nghiên cứu gốc trực tiếp dẫn tới thiết kế
nếu cần; phân biệt fulltext/abstract, không bịa novelty/citation hoặc quartile. Các đường dẫn
chỉ để truy nguồn khi phát hiện mâu thuẫn cụ thể. Trả lời trọn một lượt, không hỏi lại thông
tin mà bản brief đã có, không viết notebook trong lượt này.
