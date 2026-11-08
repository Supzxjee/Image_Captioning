# Context và yêu cầu phản biện hướng phát triển — 09/11/2026

Bạn là cộng sự nghiên cứu độc lập. Hãy đọc context này rồi đề xuất hướng phát triển thực nghiệm. Đừng chỉ đồng ý với Codex triển khai; đừng bị bó buộc phải chọn XE, SCST, reranking hoặc một cơ chế đã được gợi ý. Chúng tôi cần phương án có lý do, vừa sức và phục vụ đúng đề tài; chưa yêu cầu bạn code.

## 1. Mục tiêu và ràng buộc

Đề tài: chú thích ảnh bằng Transformer, kết hợp đặc trưng thị giác, prompt đối tượng–quan hệ riêng ảnh, căn chỉnh thị giác–ngữ nghĩa và kiểm soát ảo giác đối tượng.

Hướng phát biểu hiện tại: **Prompt đối tượng–quan hệ gắn vùng giúp giảm ảo giác đối tượng trong hệ thống chú thích ảnh có căn chỉnh ngữ nghĩa, đồng thời duy trì chất lượng caption.**

Mục tiêu thực tế: tăng CIDEr và giảm CHAIRs/i. Recall là chỉ số phụ để theo dõi độ bao phủ; người dùng vừa xác nhận không yêu cầu recall tuyệt đối không giảm. Tuy nhiên, không nên cải thiện CHAIR bằng cách làm caption quá ngắn hoặc bỏ nhiều nội dung đúng. Báo cáo mọi đánh đổi.

Bài đầu tiên hướng đến tạp chí trường HCMUE; không bắt buộc phương pháp mới/SOTA. Có thể kế thừa phương pháp đã công bố, nhưng cần thí nghiệm có đối chứng và kết luận đúng phạm vi. Người cùng lab làm hướng truy hồi facts/caption rồi hợp nhất; người dùng muốn giữ trọng tâm prompt và căn chỉnh ngữ nghĩa, tránh biến thành cùng hệ retrieval đó.

Người dùng tự chạy Kaggle, thường một notebook/tài khoản với T4×2; có tài khoản mình và Ngân. Ngân sách gần nhất là13giờ GPU **trước lượt69 vừa hoàn tất**, không coi đó là quota hiện còn. Cần hỏi lại nếu chốt run mới. Không tự chạy mô hình local hoặc notebook Kaggle. Không sửa báo cáo Word nếu không được yêu cầu. Ưu tiên thí nghiệm nhỏ nhưng trả lời được câu hỏi, tránh tạo nhiều nhánh/audit không cần thiết.

## 2. Pipeline thực tế hiện tại

Đây là triển khai tự viết theo ý tưởng CSA-Net với các thích nghi tài nguyên/detector; không khẳng định tái hiện y nguyên bài gốc hoặc đạt điểm bài gốc.

```text
Ảnh → frozen CLIP ViT-B/16 → patch14×14×768
Ảnh → detector YOLO có sẵn → boxes B/classes
patch + B → ROI pooling → linear768→512 → region features V

Caption TRAIN → spaCy POS/lemmatization → noun/verb banks
→ mẫu prompt + frozen CLIP text → fixed bank embeddings
→ learned semantic aggregation (150 prototypes/bank)
V + noun prototypes → hai object-alignment stages → Vo
Vo + verb prototypes → relation-alignment stage → Vr

Ảnh → frozen UPT → cụm đối tượng và quan hệ cùng boxes
→ frozen CLIP text → embeddings riêng ảnh
UPT boxes ↔ YOLO B: cùng class, IoU≥.5
→ object/relation slots (MLP: text + linked region features + geometry)
→ masked cross-attention/residual adapters → Vo', Vr'

Vo',Vr' → Transformer decoder3layers → beam5 caption ứng viên
→ fixedrerank(decoder + CLIP toàn ảnh + object consistency) → câu cuối
```

CONTROL có căn chỉnh bank–vùng, không có adapter prompt riêng ảnh. PROMPT thêm adapter đối tượng/quan hệ riêng ảnh. Vì vậy **CONTROL không phải baseline chỉ CLIP+decoder không có căn chỉnh**. Hai cấu hình không bằng số tham số:49.239.288 vs53.708.024 trainable (+4.468.736).

Không có Q-Former trong pipeline này. Q-Former là lịch sử kiến trúc khác; không ghép điểm/loss của nó vào so sánh nhân quả hiện tại. Không dùng BLIP ITM ở selector hiện tại. Phiên bản YOLO của cache cũ chưa đủ provenance để tự khẳng định.

Bank từ caption chỉ xây trên TRAIN. Prompt riêng ảnh lấy từ dự đoán UPT, không lấy caption tham chiếu của VALID/TEST. CLIP là encoder ngữ nghĩa, không tự trả lời quan hệ bằng caption tham chiếu.

## 3. Dữ liệu và cách học

MS COCO Karpathy:113.287 TRAIN (train+restval),5.000 VALID,5.000 TEST. Vocab10.000. Caption tham chiếu có trong TRAIN để học XE và tính CIDEr reward; không dùng tham chiếu để chọn caption lúc suy luận.

XE là cross-entropy/teacher forcing. Hai nhánh đã train **từ đầu phần caption model**10epoch XE với common base initialization, cùng dữ liệu/schedule và global batch32; visual/text pretrained encoders frozen. Không phải toàn bộ CLIP được train từ đầu.

Sau đó SCST+CE: global batch8, Adam lr5e-6/betas(.9,.99), FP32/dropoutOFF, loss **CE + .05×SCST**. SCST reward là CIDEr(câu sample)−CIDEr(câu greedy cùng mô hình), TRAIN-only reward references/IDF. Gradient cập nhật caption model/adapters, không phải chỉ hậu xử lý. Nhánh prompt hiện không có object reward trong loss.

Suy luận beam5/max20tokens; bảng “normal” là decoder rank0 của beam, **không gọi là greedy**. Greedy chỉ là baseline reward trong SCST.

Rerank cố định: **.3 D + .4 C + .3 O**, D là minmax cumulative raw decoder logprob, C là minmax CLIPScore toàn ảnh, O là detector object consistency đã chuẩn hóa; tie chọn beam sớm. Không dùng refs để chọn câu. NLL teacher-forced không đổi vì rerank.

## 4. Các checkpoint và kết quả cốt lõi

Nguồn:

| Giai đoạn | CONTROL | PROMPT |
|---|---|---|
|10epoch XE|ducanh2403/2-vi-cai-kia-qua-nhieu/178|ducanh3/notebook7bc83c4c76/58|
|Thêm1epoch SCST+CE|notebook người dùng version180|notebook Ngân version60|
|Thêm epoch SCST+CE thứ2|**chưa chạy**|notebook Ngân version69|

### 4a. XE đã được làm, không phải bước còn thiếu hoàn toàn

Chọn maxCIDEr trong10epoch: CONTROL chọn epoch10; PROMPT chọn epoch9. Cùng số epoch tối đa nhưng checkpoint được chọn có số epoch khác. Endpoint PROMPT10 vẫn được giữ.

VALID5000, trước SCST/rerank:

| Chỉ số | CONTROL10 | PROMPT9 selected | PROMPT10 endpoint |
|---|---:|---:|---:|
|CIDEr×100|99,4587|98,8453|97,9000|
|CHAIRs%|6,60|4,40|3,78|
|CHAIRi%|4,8236|3,1451|2,7585|
|Recall%|39,7664|40,5387|40,0239|
|VALID NLL|2,448211|2,516285|2,502108|

PROMPT có tradeoff: giảm ảo giác nhưng chưa tăngCIDEr soCONTROL ở selected/end10. Epoch2–9 PROMPT cóCIDEr cao hơnCONTROL, nhưng lợi thế không giữ tại endpoint10. Không kết luận quáfit chỉ từ train/valid NLL khác điều kiện.

### 4b. Cả CONTROL và PROMPT đều đã học SCST+CE một epoch

VALID5000:

| Cấu hình | CIDEr×100 | CHAIRs% | CHAIRi% | Recall% |
|---|---:|---:|---:|---:|
|CONTROL180 beamrank0|105,1821|5,50|4,0437|40,3001|
|PROMPT60 beamrank0|105,8365|4,06|2,8552|41,4427|
|CONTROL180 fixedrerank|114,2522|4,46|2,9824|43,5962|
|PROMPT60 fixedrerank|113,8004|3,02|1,9274|44,5944|

Do đó đừng đề xuất “thêm SCST” như thứ chưa làm. Muốn cô lập prompt phải kiểm ngân sách học/capacity/khởi đầu, không lấy nhánh được học lâu hơn so nhánh cũ rồi gán gain cho prompt.

### 4c. TEST đã có cho cặp180/60

Cùng KarpathyTEST5000, chính sách đã khóa từ VALID:

| Cấu hình | CIDEr×100 | BLEU4×100 | ROUGE-L×100 | CHAIRs% | CHAIRi% | Recall% |
|---|---:|---:|---:|---:|---:|---:|
|CONTROL normal|105,5606|34,3829|54,0408|5,62|4,1556|40,8294|
|PROMPT normal|107,3603|34,6452|54,4814|3,76|2,6765|42,4108|
|CONTROL fixedrerank|114,3671|36,1105|56,3504|4,54|3,0988|44,0556|
|PROMPT fixedrerank|115,3522|36,1731|56,6351|3,20|2,1181|45,1797|

Đã có BLEU1/METEOR/SPICE bổ sung, không cần chấm lại vô cớ. Ví dụ SPICE sau rerank CONTROL→PROMPT: Objects37,9115→38,2200; Relations5,9781→6,3467; Attributes10,4835→10,4325. Không phải mọi nhóm ngữ nghĩa đều tăng. CHAIR đo đối tượng, không đo đúng quan hệ/thuộc tính đầy đủ.

TEST đã được xem ở các vòng phát triển trước trong dự án. Ghi minh bạch; không nói mô hình tự cập nhật từ việc đánh giá TEST, cũng không trình bày test tái dùng như một blind holdout mới.

## 5. Các kiểm tra/hướng đã thử — tránh đề xuất lặp lại

**Prompt có được dùng không? Đã kiểm inference zero/shuffle.** Cùng trọng số PROMPT60, giữ geometry/ROI/masks:

| TEST5000 normal decoding | CIDEr×100 | CHAIRs% |
|---|---:|---:|
|Prompt đúng ảnh|107,3603|3,76|
|Embedding text zero|89,3710|5,14|
|Text slot từ ảnh khác, donor đã khóa|89,6182|10,78|

Có bằng chứng mô hình dùng thông tin text riêng ảnh ở inference. Nhưng ablation gây OOD, không thay đối chứng training matched adapter/capacity. **Chưa có pairedtraining giữ nguyên adapter/ROI nhưng một nhánh luôn textzero để tách riêng giá trị text.** Đừng nhầm hai loại ablation.

**Word–region attention supervision đã thử ở hệ bank trước đây.** Cùng sourceCSA10, XE vsXE+.1grounding,2000update/64kcaptionevents. TRAIN10k: CIDEr102,9875→103,5297, CHAIRs5,96→5,84%. TRAIN30k (expose đủ30k):103,8046→104,2326, CHAIRs5,66→5,56%. Các CI chênh lệch chứa0, chưa vượt screening gates. Không coi giảm training grounding loss là bằng chứng heldout grounding tốt hơn. Đây là continuation hệ cũ, không phải full from scratch của PROMPT hiện tại; biến thể đủ khác vẫn có thể bàn nhưng phải giải thích.

**Object reward đã thử ở hệ bank15 trước prompt.** Sau full1SCST: CE CIDEr106,3103/CHAIRs5,20%; thêm objectreward105,6063/5,12%. Mất~.70CIDEr để giảm4caption halu, lợi íchhalu chưa rõ. Đừng mô tả objectreward là chưa từng thử hoặc tự tăng penalty mà không phân tích thất bại.

**Reranking hữu ích nhưng còn lỗi.** PROMPT60 VALID: rank0105,84→rerank113,80. Trong5000ảnh,1278ảnh bị giảmCIDEr khi đổi câu. Điểmobject bằng nhau cả5caption ở2682ảnh. Có lỗi chọn câu lặp; cũng có ảnh cả5caption đều sai/thiếuđối tượng. Không thể sửa tất cả bằng selector. Không lọc cứng mọi repeatedtrigram: có ví dụ ảnh thực sự có2chiếc kéo.

**Oracle5caption CPU189:** CONTROL oracle140,2472 vsfixed114,2522; PROMPT oracle139,7454 vsfixed113,8004 trên VALID5000. Oracle dùngrefs, không deployable/không đảm bảo26điểm có thể thu được, CIDEr cao không đồng nghĩa đúng thị giác.

**Hiệu chỉnh trọng số CPU190:** prelock2000TUNE/3000CHECK, cùng66 triplet chohai nhánh, maxmeanCIDEr với CIDEr không giảm/CHAIRs,i không tăng/recall không giảm ở mỗi nhánh. Chỉbộ cũ đápứng. Không có rulemới. Đây là kết quả trong lưới/guardchặt, không chứng minh global optimum. Không đề nghị tiếp tục dò lưới/nới guard như thể chưa thấy kết quả.

**RegionCLIP:** từng bàn như hướng thay đặc trưng vùng; chưa có thực nghiệm chuyển backbone này. Chi phí/chất lượng/compatibility chưa xác nhận, không mặc định tốt hơn.

## 6. Lượt mới nhất69: điều gì đã và chưa chứng minh

Từ PROMPT60 `checkpoint.bin`, restore cảmodel/Adam/RNG; thêm1epoch SCST+CE trên113287TRAIN,14161update; orderseed43 và xoayCEreference. Giữ kiếntrúc/lr/loss/rerank. ChỉPROMPT chạy, CONTROL180 chưa học epoch2.

VALID5000 sau fixedrerank:

| Chỉ số | PROMPT60 gốc | MID thêm7081update | END thêm14161update |
|---|---:|---:|---:|
|CIDEr×100|113,8004|114,2778|115,0628|
|BLEU4×100|36,2831|36,4403|36,9083|
|ROUGE-L×100|56,5679|56,6843|56,8714|
|CHAIRs%|3,02|2,70|2,96|
|CHAIRi%|1,9274|1,7686|1,9197|
|Recall%|44,5944|44,4061|44,3119|
|Captionhalu|151|135|148|

END−gốc: CIDEr+1,2623; conditionalpaired2000bootstrap95%CI[+.2358,+2.2923]. CHAIRs net−3caption (37repaired/34introduced), CI chứa0. Recall−.2825điểm phần trăm:45cặp ảnh–classđúng được nhắc ít hơn trên15928cặpGT; CI[−.5576,0]. Mộtseed/reusedVALID/fixedIDF, chưa điều chỉnh nhiều so sánh; không phải trainingseedCI.

Rawbeamrank0: CIDEr105,8365→106,4984; CHAIRi2,8552→2,9123% tăngnhẹ, nên không nói mọi kiểu giải mã đều giảmhalu. NLL2,423685→2,419434.

Rule đã khóa trướcchạy có recall>=original tuyệtđối nên chỉoriginaleligible; `best.bin` của69 vẫn là source60. `checkpoint.bin` mới làEND69; `mid.bin` làMID. User sauxem kết quả xác nhận recall làphụ, chấp nhận cân nhắc END làmứngviên. Ghi quyết định mới saukết quả, không sửa lịch sử rule như thể từ đầu không cóguard.

Lượt69 là bằng chứng **within-PROMPT continuation**, không chứng minhSCST mới có ích (cả 2 nhánh đã dùngSCST), không cô lập prompt, không có TESTmới. Nếu soPROMPT69 vớiCONTROL180 cần nêu ngân sáchSCST khácnhau hoặc chạy matchedCONTROL tiếp. Nếu nghiêncứu thêm XE, nên phân biệt continuation từcheckpointtrước SCST và XE sau SCST; chúng không phải cùng câu hỏi.

Worker khoảng101,9phút, chưa gồmstaging/CPUscore/rerank/setup, không phải quota tính phí. Không mặc định thêmepoch tiếp tục tăng.

## 7. Vấn đề cần bạn nghĩ độc lập

Người dùng vừa phản biện rằng muốn đánh giá cơ chếcăn chỉnh thì cần trainXE; Codex đáp rằng có thể quaylại checkpointXE trước SCST và trainmatched hai nhánh. **Đây mới là traođổi, CHƯA chốt chạy thêm XE hay epoch cụ thể.** Đừng coi câu trả lời gầnnhất củaCodex là quyết định khoa học đã đúng.

Hãy giúp phân biệt ba câu hỏi:

1. Làm sao tăng chất lượng pipeline hoàn chỉnh hiện tại mà vẫn giảmhalu?
2. Cần thínghiệm gì để chứng minh giá trị của prompt/căn chỉnh, thay vì lợi ích thêm params hoặc thêm updates?
3. Với thời gian/tài nguyên hữu hạn, nên cảithiện phương pháp, học XE thêm, SCSTmatched, hay làm một ablationcònthiếu trước?

Tôi muốn bạn:

- Chỉ ra chính xác bằng chứng hiện tại đủ/chưa đủ cho claim nào. Phảnbiện cả quan điểm “chỉXE mới đánh giáđược” và “họcSCST tăngCIDEr thì đã chứng minhcăn chỉnh”.
- Đềxuất tối đa3phương án thực tế, chọn1ưu tiên và giải thích tại sao; không liệt kê hàngchụcýtưởng.
- Với phương ánưu tiên: cho pipeline/module/loss cụ thể nếu có thay đổi; nói rõ kếthừa phần nào và sửa phần nào. Nếu chỉcần chỉnhthực nghiệm, đừng bịa một module mới.
- Nêu checkpoint xuất phát, nhánhđối chứng, cái giữ cố định, câu hỏi/phép bác bỏ, chỉ sốchính/phụ, rủi roshortcut/OOD và checkpoint selection. Đừng buộc mọichỉ sốcùng tăng hoặc chỉ chọn CIDEr mà bỏ hẳnhalu/độ đầy đủ.
- Đềxuất phép thử nhỏ nhất có thể quyết định triểnkhai/loạihướng; ngân sách phải đo thực tế, không hứa hẹn thời gian hoặcđiểm. Không dùng TEST để dòloss/trọng số/epoch.
- Nói rõ điều gì cần đọc mã/receipt thêm trước khi chốt; nếu chưa truy cập tài liệu thì báo chưa đọc, không đoán. Nếu viện dẫn phương pháp nghiêncứu, dùng nguồn gốc/primary links và phân biệt giả thuyết với hiệu quả đã được xác nhận.

Kết thúc bằng một khuyến nghị rõ: **nên làm gì tiếp theo, vì sao, và kết quả nào sẽ khiến bạn đổi hướng**. Không bảo đảm cải thiện và không mặc định phải tạo đóng góp mới.

## 8. Nguồn để đọc đúng phần, không cần đọc toàn bộ lịch sử/ZIP

Repo: https://github.com/Supzxjee/Image_Captioning

- XE10 và các metrics bổsung: https://github.com/Supzxjee/Image_Captioning/blob/main/experiments/csa_paper/PROMPT_SCRATCH_QUALITY_RESULT.md
- Đườnghọc/checkpointXE: https://github.com/Supzxjee/Image_Captioning/blob/main/experiments/csa_paper/PROMPT_SCRATCH_XE10_RESULT.md (mục CHAIRI chưa đo ởfilecũ được bổ sung bởi filequality, đừng dùng trạng thái cũ như hiện tại).
- SCST180/60: https://github.com/Supzxjee/Image_Captioning/blob/main/experiments/csa_paper/PROMPT_SCST_RESULT.md
- TEST vàzero/shuffle: https://github.com/Supzxjee/Image_Captioning/blob/main/experiments/csa_paper/PROMPT_TEST_RESULT.md
- Lượt69 mới: https://github.com/Supzxjee/Image_Captioning/blob/main/experiments/csa_paper/PROMPT60_CONTINUE_RESULT.md
- Weightcalibration190: https://github.com/Supzxjee/Image_Captioning/blob/main/experiments/csa_paper/valid_rerank_diagnostic/CALIBRATION_RESULT.md
- Grounding30k: https://github.com/Supzxjee/Image_Captioning/blob/main/experiments/csa_paper/GROUNDING_30K_PILOT_RESULT.md
- Các file mô hình nếu thật sự cần: `experiments/csa_paper/model.py`, `prompt_adapter_model.py`, `prompt_scratch_model.py`, `run_prompt_scst.py`, `run_prompt_scst_continue.py`.

Context này tổng hợp số liệu đãkiểm từreview và source pins, không phải tácgiả tựchạy lại model/scorer. Không yêu cầu bạn đọc ZIP lại nếu những số liệu này đã đủ để phản biện.
