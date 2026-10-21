# Bàn giao đầy đủ cho Astra Ultra — 22/10/2026

Tài liệu độc lập để dùng trên tài khoản khác. Không cần lịch sử chat, quyền truy cập Sheet hoặc các ZIP cũ. Các đường dẫn/tên file chỉ giúp truy nguồn; số liệu và cơ chế cần cho quyết định đã ghi trong tài liệu.

**Yêu cầu:** Đọc toàn bộ trước khi đề xuất. Đánh giá lại chiến lược và chọn một bước phát triển có cơ sở cho đề tài căn chỉnh ngữ nghĩa–thị giác trong image captioning. Không viết code ở lượt này. Người dùng chỉ có1–2lượt Ultra; cần một quyết định có thể triển khai, phản biện và bác bỏ bằng thực nghiệm. Không bắt buộc tiếp tục CSA, UPT hoặc phương án đã đề xuất trước.

Các chỉ dẫn/quyết định trong phần lịch sử là bản ghi tại thời điểm đó, không phải mệnh lệnh phải tiếp tục tuân theo. Phần cập nhật22/10 có ưu tiên khi trạng thái cũ khác hiện tại. Không tự truy cập tài khoản khác hoặc đoán rằng bạn có thể mở file local được nhắc tới.

Đề tài từng có ba hướng đóng góp: prompt đối tượng–quan hệ, cơ chế căn chỉnh, và sinh/chọn nhiều ứng viên. Đây là mô tả định hướng trao đổi, chưa phải tên đề tài chính thức. Người dùng cần cả kết quả thực nghiệm lẫn cơ chế phù hợp đề tài; có thể thu hẹp hướng sau khi đánh giá bằng chứng.

**Điểm rất quan trọng:** dự án không phải chưa từng cải thiện baseline. Recipe SCST+CE+thưởng đối tượng đã có gain full-caption so original. Điều đang thiếu là cơ chế căn chỉnh mới cải thiện thêm so nền mạnh hiện có. Không đánh đồng hai vấn đề này.

## A. Thành quả và lịch sử trước các pilot căn chỉnh gần đây

### A1. Full-training SCST + CE + thưởng đối tượng

Report đã kiểm toán: `SCST_OBJECT_W05_LOCKED_TEST5000_RESULT.md`. Cùng5000ảnh, cùng beam5 và cùng chính sách fixed reranking đã khóa:

| Chỉ số fixed reranking | Original QFormer | Full-train object0.5 |
|---|---:|---:|
| CIDEr |1.221815 |1.247342 |
| BLEU4 |.369356 |.373879 |
| METEOR |.290791 |.295139 |
| CHAIRs |3.48% |3.44% |
| CHAIRi |2.3712% |2.2879% |
| Object recall |45.3131% |45.8402% |

CIDErΔ+.025526, conditional paired95%CI[+.012446,+.038868]. Caption có ảo giác174→172;60ca được sửa,58ca mới xuất hiện. Tín hiệu chất lượng caption có, giảm ảo giác fixed-policy rất nhỏ.

Ở raw rank0 của cùng beam candidate evaluation: CIDEr1.188245→1.212024,Δ+.023779,CI[+.011139,+.036453];CHAIRs4.32%→3.98%,216→199caption. Đây không phải cùng protocol greedydev1000 trong phần cập nhật, không so trực tiếp số tuyệt đối.

Test5000 đã có lịch sử tiếp xúc trong dự án, không gọi là test chưa từng xem. Một seed, CI không gồm seedvariance hoặc adaptive experiment selection. Không tách riêng đóng góp objectreward: SCST+CE control mới không được đánh giá cùng ở đây vì selection trước đó chọn fallbackoriginal. Không gọi recipe này là cơ chế căn chỉnh quan hệ mới.

Checkpoint cải thiện này được giữ làm warmstart cho nhóm pilot gần đây. A ở phần sau là continuation từ checkpoint mạnh đó, không phải Original QFormer trong bảng này.

### A2. Sinh/chọn nhiều caption và learned ranker

Đã thử candidate selection, regional/global learned ranker và gain guard. Không mặc định đó là hướng chưa khai thác.

Ví dụ report `LEARNED_CAPTION_RANKER_PILOT_RESULT.md`: fit1024/tune256, validation512, cùng candidatepool. Fixed CIDEr1.263698;global1.259200;regional1.251104. Regional hallucinatedcaptions16→13,recall44.0906%→45.2541%, nhưng CIDErΔ−.012594864,CI[−.041592568,+.016658542]. Không đạt gate đồng thời chất lượng và hallucination. Scores-only chọn epoch0 và không đổi prediction sofixed.

Đây là subset/protocol/checkpoint lịch sử riêng, không lấy số1.263698 để so với Adev1000. Không có bằng chứng reranking mọi kiểu bất khả thi; một ranker cần giải thích cách khắc phục regression/candidateheadroom/supervision, không chỉ đề xuất lại dùng CLIP để chấm câu.

### A3. Grounding/instance và kiểm tra caption supervision

Trước UPT đã thử instance contrastive alignment, CLIP-prior residual, detector boxes/candidate endpoints, soft fusion, triplet verifier, interaction regions, VLM claim verification và direct decoder relation memory. Các tên này mô tả lịch sử, không khẳng định mọi nhánh đã được kiểm toán trong tài liệu này hoặc đều thất bại. Không có bảng gốc đủ để gán số cho từng nhánh ở đây.

Report `CAPTION_RELATION_SUPERVISION_AUDIT_RESULT.md` có590trainclaims: objectalias có trong reference379/590;predicateword306/590;đồng xuất hiệnpredicate+objecttrongmộtreference217/590. Đây là lexicalcounts, không nhãn đúng/sai quan hệ: từ có thể gắn nhầm object/actor, hoặc cách diễn đạt đồng nghĩa bị bỏ sót. Không dùng missingmention làm negative, hoặc wordcooccurrence làm GTrelation. Ép mọi predictedrelation vào caption có thể làm giảm salience/chất lượng dù quan hệ có thật.

## B. Các thử nghiệm UPT và diagnostic trước H/VG/CSA

Phần B giữ dữ liệu/cơ chế từ bản bàn giao20/10. Các trạng thái dừng/nextstep ở đây là lịch sử; những phép thử bổ sung trong phần C/D đã diễn ra sau đó. A luôn giữ legacy prompt cache; nói A không có nhánh UPT mới không có nghĩa hoàn toàn không có text prompt.

### B1. A/B/C/D, E và F


### Protocol chung

- Cùng warm start tốt đã giữ lại, không khởi tạo nhánh sau từ checkpoint pilot A/B/C/D đã fine-tune.
- Cùng deterministic Karpathy train10k/val1000 cohort, seed42.
- Một nhánh/notebook, mỗi notebook dùng cả 2 GPU T4, DDP cho huấn luyện.
- Global batch8 = 4/GPU; 1250 updates; 200 updates đầu CE, sau đó CE + 0.05 SCST.
- Reward CIDEr-D + 0.5 object_count_v2, IDF từ train.
- End checkpoint, raw greedy trên cùng val1000; không chọn checkpoint bằng test.
- **Không so trực tiếp những điểm này với beam/rerank test5000 trong lịch sử.**

### A/B/C/D

| Nhánh | Cơ chế | CIDEr val1000 raw greedy |
|---|---|---:|
| A | Tiếp tục huấn luyện nền, không thêm nhánh UPT | 1.205513 |
| B | Thêm vùng ảnh chọn theo UPT, không thêm text UPT mới | 1.199067 |
| C | Cùng vùng ảnh, fusion trực tiếp CLIP text | 1.192828 |
| D | Gate thị giác có điều kiện text cho S/O/R | 1.202747 |

A là điểm CIDEr cao nhất trong nhóm. Chưa có bằng chứng các nhánh mới cải thiện tổng thể. Tráo text ở D làm gate đổi nhưng không đổi caption decoded. Vì C đã thử direct text fusion và D đã thử gate, một đề xuất “text residual + gate” mới phải chỉ ra khác biệt thực chất; không được gọi là giải pháp mới tự động.

### E — semantic evidence memory

- Frozen CLIP text states → projected query + S/O/R role embedding.
- Cached spatial CLIP patches → keys/values. Keys thêm vị trí và membership trong bbox S/O.
- S chỉ attend trong box S; O trong box O; R trong union S∪O; box quá nhỏ dùng nearest patch.
- Evidence token = LayerNorm của tổng có trọng số visual values. **Không cộng text residual vào values/output.**
- Nối evidence tokens vào memory nền cho decoder. Không có scalar gate như D.
- Patch grid14×14; giả định tọa độ phù hợp direct resize224, không crop offset. Box sai thì attention không tự tìm được object bên ngoài.

Loss E: `L_base + 0.1 L_align`, với margin0.2; matched text–region so với wrong text/region từ donor ảnh khác cùng local batch, loại exact phrase trùng. Đây là pseudo-positives và potentially noisy negatives. S đều “person” nên không có subject contrastive negatives theo luật này.

E CIDEr **1.196816**, CHAIRs **56/1000**. Đối chứng A cùng cohort đạt **1.205513**, CHAIRs **52/1000**. Chưa có lợi ích để scale E.

### F — giữ kiến trúc E, đổi giám sát matcher

- Giữ module image-value evidence như E.
- Thay pseudo-pair loss bằng cặp cùng ảnh từ annotation V-COCO.
- Fit503 cặp/310 ảnh; calibration140/79; probe62/59. Tách IDs giữa các nhóm. Supervision phụ/compute tăng nên A/E/F không cùng budget giám sát hoàn toàn.
- Caption train10k/val1000 và lịch CE/SCST giữ nguyên; loss `L_base + 0.1 verified_hinge`, margin0.2.
- Auxiliary matcher loss so positive/negative score, **không trực tiếp bắt decoder phân biệt hai câu**. CE/SCST vẫn truyền gradient qua memory caption bình thường.

| Chỉ số | A | E | F on | F off (cùng model F, mask nhánh) |
|---|---:|---:|---:|---:|
| CIDEr | 1.205513 | 1.196816 | 1.198510 | 1.197899 |
| CHAIRs (số câu/1000) | 52 | 56 | 52 | 50 |
| Object recall | 0.458039 | — | 0.462957 | 0.462957 |

- F−A CIDEr = −0.007003; paired bootstrap95 theo saved fixed-IDF per-image scores: [−0.024844, +0.011496]. Chưa chứng minh khác biệt.
- Fit margin-positive fraction tăng62.63%→96.75%, loss giảm; 6 nhóm module đều có gradient.
- Calibration predicate agreement giảm11/24→9/24; subject56/116→90/116 nhưng geometry IoU control104.5/116.
- Probe predicate7/12→5/12, object7/12→6/12, subject24/38→26/38.
- Matcher “before” là F mới khởi tạo, **không phải E đã train**.
- F off đổi73/1000 caption so on; textswap đổi3; regionswap đổi55. Không diễn giải caption đổi là caption tốt hơn.

**Đính chính nhãn quan trọng:** action=0 trong V-COCO là âm theo annotation, không bảo đảm phủ định ngôn ngữ tự nhiên. 30/80 cặp predicate fit F là carry/hold, có thể cùng đúng. Đây là nguy cơ label validity, chưa phải xác nhận toàn bộ30 cặp sai. Matcher accuracy nên gọi là annotation agreement cho tới khi có kiểm tra ngữ nghĩa phù hợp.

Quyết định hiện tại: **không full F, không thêm seed F chỉ để cứu kết quả, không tăng alignment weight một cách mù quáng.**

### B2. Frozen F diagnostic20 ảnh (lịch sử20/10)

Người dùng đã chạy xong, gửi `relation_conditioning_diagnostic_20_review.zip`. Đây là suy luận, không huấn luyện.

- Checkpoint F SHA256: `4938bfe875f05104cd0afca4918dad3761b181bece08d773ad2b71b6adb51043`.
- Mounted checkpoint: `/kaggle/input/datasets/ducanh2403/checkpoint-f/checkpoint_F.bin`.
- Chọn20 ảnh duy nhất từ21 ảnh đủ điều kiện matcher F không ưu tiên annotation dương; chọn cố định bằng hash, hậu nghiệm trên failure set.
- Thuộc caption train/restval; từng dùng chẩn đoán. Không phải benchmark độc lập, không test, không dùng chọn checkpoint.
- Hai GPU độc lập,10 ảnh/GPU; weights frozen. Khớp11 sources và manifest, hash full model giữa2rank; replay matcher max error3.13e−7. Kaggle3 data tests+1 synthetic model test PASS; audit local tái tính chỉ số từ JSON, không replay tensor/model.

Bốn điều kiện:

1. `off`: mask nhánh evidence, giữ memory nền của F đã học.
2. `upt`: selected prompt/boxes UPT ban đầu.
3. `annotation_positive`: một triple GT S/O/R.
4. `annotation_negative`: cùng GT boxes/S/O/số slot, **chỉ đổi text R**.

So sánh có kiểm soát là 3 với4. UPT với GT còn đổi box/số slots nên không thể quy tác động riêng cho “quan hệ đúng”.

| Điều kiện | Caption đổi so off | Ưu tiên annotation+ theo tổng logP |
|---|---:|---:|
| off | 0/20 | 11/20 |
| UPT | 3/20 | 11/20 |
| annotation+ | 2/20 | 11/20 |
| annotation− | 2/20 | 11/20 |

Giữa annotation+ và annotation−:

- Attention R L1 distance trung bình **0.435869**.
- Evidence token R relative L2 distance trung bình **0.230696**, min0.029037, max0.941822. Đây là khoảng cách, không phải accuracy hay cải thiện23%.
- Caption greedy khác nhau: **0/20**.
- Phía mô tả được ưu tiên giữ nguyên ở từng ảnh qua cả4điều kiện: **20/20**, không phải các ảnh đổi chiều bù trừ.
- Dịch chuyển chênh logP(+−−) trung bình−0.000885; mean absolute0.003725; max absolute0.014862; 8ảnh tăng,12ảnh giảm.

**Kết luận có thể nói:** đổi quan hệ làm đổi cách lấy bằng chứng và token ảnh; đáp ứng likelihood nhỏ; đổi động từ chưa điều khiển được câu greedy trong20ca này.

**Không được nói:** mô hình hoàn toàn bỏ qua prompt; decoder chắc chắn hỏng; token đổi nhiều là đúng ngữ nghĩa; 11/20 là semantic accuracy; hoặc20ca đủ chứng minh cải thiện benchmark.

Ví dụ:

- 251696: carry phone vs talk on phone, relative token change0.9418 nhưng cả hai vẫn sinh “a man and woman on a subway train talking on their cell phones.” BoxS là phụ nữ bên trái đang nhìn phone, người cạnh áp phone gần tai. Có vấn đề phân biệt instance và mô tả toàn ảnh cần xem kỹ.
- 340406: hold bicycle vs ride bicycle đều sinh “a group of people waiting for a train to come in.” Cảnh chính là ga tàu, target relation có thể phụ.
- 495476: hold backpack vs carry backpack đều sinh “a group of people riding on the back of an elephant.” Không nhắc ba lô phụ không tự động là caption sai.

Đã xem contact sheets20ảnh bằng assistant; chưa có human semantic verdicts. Không coi đó là tập nhãn human đã xác nhận.

### B3. Giả thuyết lúc đó — chưa chốt nguyên nhân

1. Text chỉ đổi attention trên cùng visual values; token có thể chưa giữ rõ khác biệt động từ.
2. Decoder ưu tiên memory nền mạnh hơn nhánh mới; CE/SCST toàn câu chưa tạo đủ động lực dùng R.
3. Matcher có thể học shortcut/cấu trúc nhãn mà không cung cấp thông tin có ích cho caption; loss giảm không đủ.
4. Cặp âm ngữ nghĩa không hợp lệ; holding/carrying, riding/holding có thể chồng lấn.
5. Instance relation được chấm không phải điều đáng kể trong caption toàn ảnh; ép đưa vào có thể hại salience và CIDEr.
6. Union-box patch attention, độ phân giải14×14, thiếu đặc trưng tương tác/pose có thể hạn chế động từ; đây là giả thuyết cần kiểm tra, không phải kết luận từ audit.

Không mặc định chỉ một nguyên nhân đúng. Một “bộ kiểm tra quan hệ” tốt cũng chưa đương nhiên tạo caption tốt hơn nếu salience, representation và decoder learning không được nối đúng.


## C. Các phép thử phrase/pair sau bản bàn giao cũ

### C1. Nhãn human và CE so với CE+CROSS

Đã có humanreview30nhóm:26/30semanticcontrasts+boxes rõ;21/30 còn thỏa cả hai event nhìn rõ ở224.21nhóm gồm42eventviews/38ảnh,4nhóm cùngảnh hai cặp;chủ yếu skateboard/surfboard/phone,bicycle. Một người chấm,không independentbenchmark.

7/42eventviews có ít nhất một box không chứa tâm patch14×14 dù người chấm thấy bằng chứng ở224. Humanvisibility không bảo đảm CLIP phân biệt được, và không đủ để kết luận CLIP không có tín hiệu.

Pilot matched CE versus CE+CROSS:14fitgroups/7holdoutgroups,holdout14events. Cả hai tăng supplied-phrase preference10/14→13/14,nhóm đúng2chiều3/7→6/7. CROSS tăng meanmargin.481→.721 soCE nhưng không tăng số quyết định đúng. Greedy exacttemplate5/14 cho cảhai;stringexact không semanticaccuracy. Fit cảhai perfect28/28. Không scaleCROSS.

Trường hợp cùngảnh208095: một người đứng trên surfboard, người khác mang board; cảhaiarm sinh standing-on cho cảhai. 70642: carrying-skateboardtarget được CE mô tả standing-on;CROSS đổi sang standing-with,giảm lỗi cụ thể nhưng chưa diễn đạt carrying. Teacher-forced ranking đúng không bảo đảm freegeneration đúng.

### C2. Pair-prefix diagnostic: tránh chẩn đoán sai decoder

Chỉ4nhómsame-image:2fitcontrols+2holdoutfailures;8events/arm,không benchmark mới. Cảhaiarm đúng firstrelationtoken trên4fitcontrolevents,nhưng chỉ2/4holdoutevents (standing-on),cảhai carryingtargets vẫn ưu tiên standing.

Sau prefix `a person`,CE ởsurfboard208095 P(standing)47.54% vsP(carrying)1.52%;CROSS41.62% vs.25%. Với skateboard70642,CE16.09% vs13.72%;CROSS12.17% vs9.85% (`is` mới làtop1CROSS).

Wholephrase meanlogP ưu tiên carrying ở70642 dù nexttokenargmax ưu tiên standing; score gồm easycontinuations,câu khácđộdài vàkhônggồmEOS. Đây là observed surrogate/generationmismatch,không chứngminh length/EOS lànguyênnhân duy nhất.

BOSgreedy và continuingprefix giống nhau,nhưng model vốn tự sinh prefixđó nên không thể suy exposurebias từ phép này. Đổi selectedpair vẫn đổi probabilities;vídụCE P(carrying)13.72% chochildcarrying vs4.42% chochildstanding cùngảnh. Nhánh có hoạt động,nhưng chưa đủ cho quyết định sinh đúng. Geometryswap không loại hết geometry vì masks/membership còn chứa hình học;không cô lập nguyênnhân encoder/decoder/globalcontext.

### C3. Paircrop vs fullframe, có blindhumanreview

Giữ selectedS/O supervision,thử re-encode paircrop sofullframe. Review32candidates/16events/8groups mới cho adaptation,đủnhãn,mộtngười chấm AI-assisted.

| Human generation outcome | Fullframe | Paircrop |
|---|---:|---:|
| Đúng và đủ cụ thể |10/16 |11/16 |
| Sai quan hệ/thêmfact |5/16 |4/16 |
| Saiinstance |1/16 |1/16 |
| Nhóm cảhai cặp đúng |2/8 |3/8 |
| Surfboard jointgroups |0/4 |0/4 |

Một skateboardgroup cải thiện,không regression;không đạt gate≥6/8,netgain≥3,vàcó gain cảhaihọ. Một event có bất đồng giữa nhãn target cũ và humanoutputreview mới;giữ nguyên hai nguồn. Ngay cả cho event tranhchấp thànhđúng vẫn≤4/8,không đổi NO_GO.

Không scalecrop-only hoặc thêmSCST chỉ để cứu kết quả. Chưa kếtluận mọi regionencoding vôích;crop giúp mộtcase nhưng chưa cơchế tổngquát. Không xem các tinyphrasepilot này như full-captionCIDEr/CHAIRimprovement.

## D. Cập nhật22/10: H, VG, CSA và quyết định hiện tại

Các số đầy đủ và yêu cầu ra quyết định tiếp theo ở phần D bên dưới. Cùng dev1000 đã sử dụng lặp lại; không trộn bảng test5000/val512/phraseholdout bên trên.

Trong các pilot gần đây, nhãn `original` chỉ checkpoint warmstart đã qua SCST+CE+objectreward trước khi continuation, không phải Original QFormer trước recipe ở bảng A1. A là continuation của warmstart này. Phải giữ tên hệ thống và protocol riêng khi diễn giải các bảng.

# D1. Bối cảnh hiện tại và nhiệm vụ quyết định

Ngày2026-10-22. Người dùng chỉ có thể dùng1–2lượtUltra. Phần này tổng hợp facts mới nhất đã root kiểm toán để bạn quyết định nghiên cứu trong một lượt; **không cần đọc lại ZIP/repo/Sheet để xác minh lại**. Chỉ tra cứu tài liệu nghiên cứu gốc khi cần cho một lập luận cụ thể, và nói rõ có đọc fulltext hay chỉ abstract. Không dùng lượt thứhai chỉ để diễn đạt lại proposal.

## Yêu cầu nghiên cứu

Đề tài: căn chỉnh ngữ nghĩa–thị giác cho imagecaptioning, trước đây định hướng prompt đối tượng–quan hệ. Người dùng cần cải thiện so baseline mạnh và cơ chế/đóng góp căn chỉnh có thể kiểm chứng. Không bó buộc vào CSA/UPT hoặc gán mỗi từ vào một bbox. Bạn có quyền bác bỏ các quyết định trước đây của chính bạn; giải thích bằng chứng khiến đổi quan điểm. Root thực hiện và phản biện, người dùng chạy Kaggle.

Trong **một lượt**, hãy:

1. Chẩn đoán nút thắt khả dĩ nhất; phân biệt điều được chứng minh, giả thuyết còn mở, thông tin chưa có. Đừng gán mọi thất bại cho decoder bỏ qua prompt nếu chưa có phép thử xác nhận.
2. So sánh tối đa3hướng khác nhau thực chất, gồm cơ chế tác động lên caption, supervision/nguồn dữ liệu và rủi ro. Không bắt buộc tạo3hướng nếu bằng chứng chỉ đủ choít hơn.
3. Chọn một hướng ưu tiên hoặc đề xuất phép đo quyết định trước khi chọn kiến trúc. Có quyền dừng CSA hay giữ lại; không lựa chọn chỉ vì vừa triển khai xong.
4. Với hướng chọn, xác định input/output/module/gradient và công thức loss đủ triển khai, positives/negatives/labelcoverage và hành vi trên ảnh mới. Nếu chưa có dữ liệu cho loss, nói rõ, không dùng pseudo-label như chân lý.
5. Đưa phép thử nhỏ nhất có thể bác bỏ giả thuyết, đối chứng cần thiết, gate tiếp tục/dừng, và cách diễn giải kết quả âm. Nếu thiếu một fact then chốt, nêu phép đo lấy fact đó thay vì mở hàng loạt nhánh hoặc giả định có dữ liệu.
6. Tự phản biện phương án chọn. Tách kế thừa/engineering khỏi phần có khả năng mới; không tuyên bố novelty nếu chưa kiểm tra priorart. Không cần viết code.

## Pipeline và nguồn lực hiện có

- CLIPViT-B/16 frozen,last_hidden_state197×768 (CLS+196patches),cacheFP16đọcFP32. Ảnh resize224vuông bilinear,khôngcentercrop.
- Projection768→512→QFormer2layers,32learnedqueries512dim. Baseline không dùng prompt-conditionedQFormer.
- LegacyCLIPtextpromptcache giữ nguyên: promptprojection và prompt→visualattention+gate tạo groundedlegacytokens. Decoder đọc `[grounded_legacy;32queries]` để sinh caption. Đây không đồng nghĩa từng từ có box verified.
- Checkpointoriginal đã qua SCST+CE+objectreward ở giai đoạn trước, không phải model CE yếu. Các pilot phía dưới bắt đầu từ cùng checkpoint đó; A là đối chứng continuation mạnh trên cùng train/dev. Không trộn các số greedydev này với beam5/test5000 lịch sử.
- A:train10k,globalbatch8,T4×2,seed42,1250updates;200CE rồi CE+.05SCST;rewardCIDErD+.5object_count_v2;fullTRAIN IDF. AdamWLR5e-6,clipglobalnorm1,evalmode nhưng gradients bật.
- Object reward dùng TRAINCOCO instance+reference union, salientuniqueobjectrecall và penaltyunsupportedcount; detectorhighconfidence có thể neutralizeunannotatedclasses,không tạo positiveGT. Captionrefs dùng khi train và khi eval, không có ở inference.
- Có UPT fullCOCO predictioncache và promptcache allactions; có hiddenpaircachetrain10k/dev1000, tối đa8slots1024dim,cặp người→vật với detection>.2,khôngverbthreshold. Train6423/10000cópair,dev654/1000cópair. Đây là dự đoán, không GT; VCOCO coverage chỉ một phầnCOCO. Quan hệ đa nhãn, unannotated không tự là negative.
- Có VGregionphraseannotations,trong audit có weak/ambiguous/wrongregionphrases; không GT sạch. Các patchCLIP đã contextualized; regionmask không cô lập hoàn toàn ngữ cảnh. Crop/region224 có thể mất chi tiết phân biệtquan hệ.
- T4×2 một notebook; đôi khi có hai tài khoản/notebook, không mặc định nhiềuGPU hoặc ngân sáchgánnhãn lớn. Quota còn lại hiện chưa biết. Không tự chạy GPU/notebook local.

## Bằng chứng đã kiểm toán trên cùng dev1000

| Arm | CIDEr | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|---:|
| Original |1.189065508 |.058 |.035452323 |.456809099 |
| A continuation |1.205512599 |.052 |.032475490 |.458038733 |
| H: thêm UPT hiddenpair memory |1.201123876 |Không cần cho lập luận hiện tại |— |— |
| G: VG regionphrase auxiliaryCE |1.208400647 |.053 |.033067973 |.456194282 |
| T: cùng phrase nhưng allpatches |1.185746927 |.052 |.032635468 |.455272057 |
| GV: G nhưng aux không gradient trực tiếp vào decoderparams |1.193220147 |.053 |.033027523 |.458653551 |
| CSA hai pha bật |1.199997684 |.051 |.031116534 |.459268368 |
| CSA hai pha tắt sau train |1.212947915 |.048 |.029625151 |.462649862 |

- H−AΔ−.004388723,paired95%CI[−.023219654,+.014609599]. Off H1.202134333; tháo memory làm48/1000caption đổi,swap làm36/1000đổi. Nonzerogradient không chứng minh grounding.
- G−AΔ+.002888048,CI[−.017606965,+.023640698]. G−TΔ+.022653720,CI[+.001459794,+.044562693]. G>T không chứng minh G>A; có thể giảm hại từ phrase không localize.
- GV−AΔ−.012292452,CI[−.032403839,+.007695362];GV−GΔ−.015180500,CI[−.035034832,+.003657362]. Decoder vẫn cập nhật captionCE/SCST, aux functional_call detached decoderparams vẫn truyền gradient về visualmemory. Globalclipactive2500/2500ranksteps nên có tác động gián tiếp lên scale cập nhật decoder. Đã dừng recipeVGregionCE này.
- Các diagnosticpair/phrase trước có nhiều kết quả âm; không dựa vào việc overfit vài mẫu để khẳng định caption tổng thể tốt hơn. Không có số định lượng bổ sung đã xác minh trong brief này để suy rộng.

## CSA cụ thể đã thử — tránh nhầm với tái hiện fullpaper

Nguồn *Cascade Semantic Prompt Alignment Network for Image Captioning*, IEEE TCSVT2024,DOI10.1109/TCSVT.2023.3343520. OfficialCrossmodalGroup/CSA-Net commit0cc570ba7c50ef08c46a21439fe73ef0a5df97ab.

- GlobalRegionCLIPnounbank4764×1024/verb975×1024,đóngbăng. Mỗi bank learnedSFA thành150prototype512dim.
- Giữ nguyên authorcore2objectstages+1relationstage: similarity/pooling,salienceweight,SA/CA/FFN/scalargatedfusion;23,161,644params. Source rawdot/temps.1/.05,khác diễn giải cosine trongpaper; không âm thầm sửa math.
- Input32QFormerlatentqueries512 thay150detectorregions gốc. Vo/Vr32tokensmỗiloại,decoderhiệncó đọc `[M_A;Vo;Vr]`;không dùng originalparallelCSAdecoder. Authorrel_mask sumfeatures==1 được giữ đểparity,không dùng padding;64tokenthêm đềuvalid.
- Smoke xác nhận engineeringparity trên cùng authorforward:outputmaxabs0,N32/150,93paramgradtensor checks;GPUbackwardbankfrozenbypasscorrect. Không haiimplementation độc lập, không chứngminh grounding.
- Pha1:1250CE,baselinefrozen,coreLRwarmup100→1e-4. Pha2:1250joint,base5e-6/core2e-5,CE200rồi objectiveA. Freshoptimizer/RNG/sampler giữa pha;bankfrozen. Nhiều hơn A1250core-onlyupdates.
- Audit12source,phase1baselinebấtbiến,phase1final=phase2initial,phase2joint đúng lịch,2ranksstatehash khớp,fullfinite. Commonruntime/inputgroups/traintargets/originalpredictions/per-imageCIDEr/CHAIR/evalreferencehash khớpA. HistoricalA không lưuactualsamplerorder/RNG; chỉ so đượcpolicyseed/source.
- Stagingtensorlayoutkhác do bỏ UPT không dùng;ZIPkhôngchứa checkpoint/stagetensors để rootrehash độc lập. Runtime đãhashcheck. Không có căn cứ lỗi chạy hiện tại để giải thích sạch mọi kết quả âm.

CSA comparisons:pairedbootstrap20kimagesresampling,seed42:

| So sánh | ΔCIDEr | CI95% |
|---|---:|---|
| On−A |−.005514915 |[−.024690383,+.013728802] |
| Off−A |+.007435316 |[−.012887746,+.028659773] |
| On−Off |−.012950231 |[−.029562424,+.002663415] |

On/off267captionkhác. CHAIRhallucaptionA52,on51,off48. **Chưa có bằng chứng vững on kém A/off hoặc off hơn A**;tất cảCItrênchứa0. OffkhôngphảiA;không được chọn off saukhinghìnsốrồigọi phươngpháp thắng. Chưa làmhumanblindreview mới.

CSAallocatedpeak1.334GiB/GPU,reserved1.377GiB. Staging~63.9phút;pha1train2.04phút,pha2train8.66phút,khônggồmbootstrap/fullTRAINidf/generation. Không suy fulltrainingtime bằng phépnhân4stepssmoke.

## Tiêu chí quyết định và điều chưa biết

Gate đã đặt trước:soAΔCIDEr≥.01,paired95%lower>0,beatOriginal,CHAIRs/i không tăng,recalldrop≤.01;đạt sốmới blindfactualreview. CSAon/offđều chưađạt. Dừngrecipe2500steps này theo gate,không kếtluận mọiCSA/cănchỉnh khôngkhảthi. Ngânsáchkhôngđảmbảohộitụ,devđãdùngnhiều,1seed,bootstrapkhôngđo seedvariance/selectionbias.

Tôi muốn bạn tìm **điều kiện thay đổi thực chất** để kỳ vọng kết quả khác. Tránh chỉ đổi gate/weight/addmodule rồi chạy lại; cũng không mặc định phải lặp fullpublishedsystem,thuê thêmGPU/gánnhãn lớn. Nếu cần phân biệt decoderconditioning với semanticrepresentation/evidencequality, hãy thiết kế phép đo quyết định rõ thay vì tuyên bố một nguyên nhân đã được chứng minh.

Đầu ra cần một quyết định và kế hoạch có thể thực hiện, không danh sách ý tưởng dài. Root sẽ kiểm lại assumptions và triển khai; chưa viết code trong lượt này.

## E. Cách sử dụng toàn bộ lịch sử trong quyết định

- Các kết quả tốt SCST full-caption là nền đã có; nhiệm vụ hiện nay là đóng góp căn chỉnh thêm lên nền mạnh.
- ABCD/E/F cho thấy phải giải quyết cả supervisionvalidity,representation vàdecoderutility,không chỉ thay tên fusion/gate.
- F20/prefix/crop cho thấy evidence có thể đổi nhưng generation chưa phân biệt đúng; đồng thời không đủ bằng chứng kết luận decoder hoàn toàn bỏ qua evidence hoặc localencoding là nguyênnhân duy nhất.
- H/VG/CSA làm các phép thử caption ởquymôlớnhơn,nhưng chưa đạt gatevsA. Kế thừa CSA không tự thànhnovelty; kết quả âm của query-basedadaptation không bác bỏ fullCSA.
- Chọn một hướng hoặc một phép đo quyết định rồi giải thích tại sao thay đổi đó thực chất khác các phép đã làm. Có thể sử dụng lại thành phần cũ nếu lýdo và phép kiểm chứng cụ thể; không cấm kỹthuật chỉ vì từng có một biến thể âm.
- Hãy tự do phản biện các gate/protocol cũ cho nghiên cứu tương lai nếu cólýdo khoa học,nhưng không đổi retrospectivecriteria để gọi một kết quả hiện tại là thắng.
- Chưa có quyết định được kiểm chứng về module tiếp theo. Không hứa tăngđiểm;phải tách bằngchứng,suyđoán và điều kiện tiếp tục.

Yêu cầu cuối: trả lời trọn một lượt theo các mục ở D:chẩnđoán có phảnvídụ,so sánh≤3hướng,chọn1,module/loss/supervision/inference,smallestfalsificationexperiment,go/stop,priorart/novelty,selfcritique. Không viếtcode. Không cần dựng lại mọiartifact hoặc đọc toànrepo. Nếu cần literature mới,ưu tiên nguồn nghiên cứu gốc và gắn citationđúng claim,nêu rõ fulltextavailability.
