# Vòng brainstorm độc lập cho Argon: đóng góp căn chỉnh thị giác–ngữ nghĩa

Ngày 30/10/2026. Đây là brief để bạn nghiên cứu và phản biện, **chưa phải phương pháp đã chốt hoặc yêu cầu viết code**. Hãy trả lời bằng tiếng Việt. Bạn không cần đọc lại ZIP/log hay đoán nội dung các chat trước; các số liệu cần thiết đã được tổng hợp bên dưới.

## Cập nhật sau khi Astra đối chiếu 5 primary methods

Phần bên dưới giữ lại candidate và các câu hỏi ban đầu để bạn phản biện. Astra đã hoàn thành một focused method review, đọc các sections/equations liên quan (không toàn bộ bài/appendix) của NBT, LSTM-C, CBS, M3ID và HALC. **LSTM-C không phải DCC**; đừng nhầm tên hai phương pháp.

- NBT đã có prefix/state→ROI→lexical output, sentinel và nhiều positive proposals. Positive-set sum của candidate so với average trong NBT chỉ khác một hằng số theo event nếu tập positive cố định, cùng gradient; không nhận set loss là mới.
- [LSTM-C](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yao_Incorporating_Copying_Mechanism_CVPR_2017_paper.pdf) đã trộn generation/copy và chuẩn hóa; [CBS](https://arxiv.org/pdf/1612.00576) điều khiển nội dung bằng lexical constraints. [M3ID](https://arxiv.org/pdf/2403.14003) và [HALC](https://arxiv.org/pdf/2403.00425) đã đưa visual evidence vào việc sửa token probabilities. Đừng nhận "đưa grounding trực tiếp vào đầu ra" là firstness.
- Công thức candidate rút gọn thành p_new(w)=p_B(w)*q_c/b_c trong class c: tương đương class logit correction log(q_c/b_c). Correspondence nằm trong cách học q, không phải việc đặt tên probability interface.
- Cả phản ví dụ EOS và lỗi linear common-key cancellation ở mục6 vẫn có hiệu lực. Nếu giữ control global appearance, đề nghị scorer additive nonlinear v^T tanh(W_h h_t + W_a e_i + W_g geometry_i), thay e_i bằng mean trước tanh; đây là sửa control bằng thành phần chuẩn, không novelty.
- Astra chọn **SỬA, chưa chọn triển khai**. Khác biệt còn lại là thích nghi grounded lexical generation vào strong frozen host, lấy GT class từ IoU mà không ép detector class đúng, và ràng buộc phạm vi thay đổi xác suất. Chưa chứng minh vấn đề tích hợp vào host thực sự do coupling mass gây ra hoặc thiết kế này đủ mới.
- Minimum nếu sau này được chọn: U-local pointer/copy không ràng buộc; M-local có mass constraint; M-global nonlinear, cùng ROI lexical values. Có thêm U-projected inference-only giữ learned weights/conditionals tại cùng prefix nhưng đưa mass về host. M−U là tổng hiệu ứng thay thiết kế, không phải pure mass effect. Thêm nhánh không tạo novelty.

**Yêu cầu với Argon:** hãy cân nhắc độ mới cơ chế hiện thấp và chủ động sửa/bỏ candidate nếu cần, không chỉ tìm thêm đối chứng để bảo vệ nó. Nếu đề xuất thay hướng, chỉ rõ closest prior, delta và phép thử quyết định. Không mặc định phải giải quyết bằng probability correction. Các facts chưa có gồm lexical/proposal coverage, số proposal đúngbox nhưng sai detectorclass, local semantic signal thực và quota hiện tại.

## 1. Điều tôi cần bạn giải quyết

Tôi làm đề tài image captioning có căn chỉnh ngữ nghĩa–thị giác. Tôi cần một phương pháp hoặc cải tiến cụ thể, có cơ sở và có khả năng kiểm chứng để viết bài đầu tiên tại một tạp chí đại học trong nước. Mục tiêu thực nghiệm là tăng CIDEr và giảm hallucination so với baseline mạnh đã có, đồng thời chứng minh cơ chế căn chỉnh góp phần vào kết quả.

Tôi không cần một công thức toán hoàn toàn chưa từng tồn tại. Nhưng thêm các thành phần quen thuộc rồi đổi tên, thay backbone, hoặc tăng điểm chỉ nhờ train tiếp thì chưa đủ làm contribution của phương pháp. Tôi cũng chưa chọn đổi đề tài sang một bài chỉ so sánh các kết quả âm.

**Hãy suy nghĩ độc lập. Bạn không phải bảo vệ đề xuất của Astra dưới đây. Có thể giữ, sửa hoặc bỏ nó.** Không mặc định căn chỉnh phải là từng từ vào một bbox: cân nhắc mức từ, cụm mô tả, đối tượng hoặc quan hệ nếu phù hợp, nhưng chọn một hướng có thể triển khai trong điều kiện thực tế.

Hãy phân biệt: (a) phát hiện đúng vùng, (b) biểu diễn đúng ý nghĩa, (c) khiến captioner sử dụng bằng chứng đó. Một heatmap đẹp hoặc một classifier phụ tốt không tự chứng minh (c).

## 2. Hệ thống và tài nguyên hiện có

- Captioner: CLIP visual tokens 197×768 → Q-Former 32 queries; có original semantic prompt cache 20×512 và mask, prompt fusion và caption decoder vocabulary khoảng 10.000. **A có prompt gốc, không phải mô hình no-prompt.**
- Baseline mạnh A là checkpoint đã huấn luyện tiếp, kế thừa SCST + CE + object reward; không phải Q-Former khởi tạo. Có checkpoint A và C0.
- Có cache visual CLIP patches, original prompt embeddings và predicted object boxes/confidences. Có COCO TRAIN instances/classes và 5 reference captions/ảnh. Có UPT full-cache/prompt, một số UPT hidden pair features từ các lượt trước. Không giả định UPT bao phủ mọi quan hệ hay detector cache lưu đủ mọi prediction.
- Có nguồn Visual Genome/V-COCO đã dùng trong các thử nghiệm trước; dùng nguồn mới cần giải thích supervision, mapping/split và chi phí, không giả định đã có một bộ phrase–region gold hoàn chỉnh.
- Có TRAIN inventory 10k; các pilot gần nhất dùng 4.000 ảnh TRAIN, 500 updates, batch toàn cục 8, seed42. Development 1k và validation 5k đã được dùng nhiều lần. **Không coi validation này là tập xác nhận độc lập.** Không dùng test để chọn kiến trúc/hyperparameter.
- Người dùng tự chạy Kaggle. Không chạy notebook hoặc mô hình local thay người dùng. Số giờ GPU hiện còn chưa xác nhận; số 14h là trước lượt evaluation cuối, không phải số hiện tại. Đừng lập kế hoạch dựa trên 14h còn nguyên hay hứa một phương pháp sẽ thành công trước deadline.
- Nhãn người dùng kiểm tra trước đây có panel pha AI/human, không được coi toàn bộ là human gold. CHAIR đo hallucination đối tượng, không đo đầy đủ lỗi quan hệ, thuộc tính hoặc instance.

## 3. Kết quả mới nhất đã kiểm tra

Cùng 4.000 ảnh validation, cùng evaluator/decoding, không train trong lượt đánh giá:

| Mô hình | CIDEr ↑ | Caption có ảo giác ↓ | False object mentions / tổng mentions | Object recall ↑ |
|---|---:|---:|---:|---:|
| A | 1,189771 | 239/4000 | 254/6484 | 0,468087 |
| C0 | 1,204560 | 235/4000 | 248/6502 | 0,471085 |
| C | 1,207042 | 230/4000 | 243/6477 | 0,470296 |

- C0 = train tiếp từ A bằng SCST với reward gốc và rollout KL neo vào A; không có loss bảo toàn khái niệm.
- C = cùng recipe C0 + positive concept-mass preservation. C/G/C0 cập nhật hai lớp decoder cuối; giữ CLIP/Q-Former/prompt cố định. G thêm weak-negative concept suppression và trước đó cũng không có lợi ích riêng rõ.
- C−A CIDEr +0,017271, paired95%CI [+0,008819; +0,025853]. C0−A cũng có CI dương.
- C−C0 CIDEr +0,002481, CI [−0,001357; +0,006263]. Chưa quy gain riêng cho preservation.
- C giảm false captions 239→230 so A, 235→230 so C0. Nhưng CHAIRs CI đều chứa0; exact McNemar p≈0,306 và0,180. Không qua joint gate đã khóa.
- Kết luận đúng: có gain CIDEr và false-count point estimates tốt hơn, **chưa xác nhận lợi ích riêng hoặc giảm hallucination**. Không kết luận mọi hiệu ứng bằng0, mọi cách alignment đều vô dụng, hoặc đã đạt mục tiêu kép.
- Đã đóng vòng C/G/C0; không tuning hoặc tăng n để cứu gate trên cohort này. One seed/reused validation/post-pilot selection còn giới hạn. CIDEr 1k/4k/5k dùng IDF khác nhau: không so trực tiếp raw scores qua cohort hoặc cộng thành score5k.

## 4. Những hướng đã thử: tránh đề xuất lại mà không có khác biệt

1. **Giám sát attention native (R):** noun positions trong reference → decoder attention đến Q → Q attention đến patch; phạt thiếu mass trong bbox. Không thiết lập gain caption riêng so continuation/all-spatial control. Chỉ compose attention rồi thêm box loss đã được làm.
2. **Sửa attention khi giữ captioner cố định (T/W diagnostic):** tăng mass đúng box trên tập TRAIN nhỏ nhưng likelihood từ mục tiêu không cải thiện. Attention mass không đủ đại diện cho bằng chứng hữu ích.
3. **Noun evidence G/S:** head patch→class; S dùng bbox pooling khi train. Cả hai inference dùng global class evidence cộng bounded noun-logit bias. Cùng val5k: A CIDEr1,182248/291 false captions; G1,172104/376; S1,171573/356. Không có lợi ích bbox được xác nhận. Đừng hiểu khác IDF ở đây là A đổi weights.
4. **Pair/region hidden-memory residual I/J:** J tốt hơn control I nhưng dưới A. Dev1k A1,205513; I1,035871; J1,134207. CHAIR false captions52/83/61. Chưa có semantic compensation đủ mạnh. Reader-off không phải bypass toàn mô hình; không suy rộng rằng mọi region alignment thất bại.
5. **UPT relation prompts, verified pair matchers, regional/VG supervision, native high/low-resolution detail và grounded-gain heads:** đã có nhiều pilot chưa đạt chất lượng hơn strong A. Không xác định được một nguyên nhân chung duy nhất. Không mặc định thêm dữ liệu/nhánh/epoch sẽ giải quyết.
6. **Tái hiện lõi CSA:** đã có technical smoke và two-phase pilot, chưa có thành công mới làm nền cho claim. Tái sử dụng thành phần CSA không bị cấm, nhưng tái hiện đơn thuần không phải novelty.

Các cơ chế trên khác nhau; không gộp mọi failure thành một bằng chứng rằng “căn chỉnh không khả thi”. Chỉ mở lại một họ phương pháp khi có thay đổi cơ chế và lý do có thể kiểm chứng.

## 5. Ứng viên hiện tại của Astra — để phản biện

Tên mô tả tạm thời: **region-grounded lexical redistribution trên frozen captioner**. Chưa xác nhận tính mới, chưa triển khai hoặc duyệt GPU.

Giữ C0 frozen. Lấy tối đa12 predicted proposals; fractional-pool CLIP patches theo box, chiếu thành ROI appearance e_i. Không chạy crop encoder mới trong đề xuất hiện tại.

- ROI classifier học rho_i(c), thêm OTHER, chỉ từ appearance; không nhận class ID/name của detector.
- Query từ decoder hidden h_t của ảnh và các từ đã sinh, không nhận target word tương lai.
- Pointer pi_ti chọn ROI từ query và key gồm appearance/box/confidence; thêm NULL fallback.
- GT COCO chỉ match predicted ROI bằng IoU và tạo class targets. Noun reference dùng positive set gồm mọi ROI hợp lệ của class, không bịa nhãn instance duy nhất. Không có matching ROI thì bỏ pointer loss của event, báo coverage; không thay bằng GT box ở inference hoặc gán unmatched ROI thành negative chỉ do thiếu annotation.

Định nghĩa một lexicon N gồm object aliases một token, chia thành các tập N_c không chồng lấn, loại ambiguous surfaces trước run; chưa bảo đảm phủ đủ80classes, chưa xử lý multiword ở pilot.

Với p_B là phân phối frozen C0 tại cùng prefix:

    m_B = sum_{w in N} p_B(w)
    b_c = sum_{w in N_c} p_B(w) / m_B
    u_B(w|c) = p_B(w) / sum_{v in N_c} p_B(v)
    q_c = pi_0*b_c + sum_i pi_i*[rho_i(c) + rho_i(OTHER)*b_c]

Phân phối dùng thực khi generate/sample:

    p_new(w) = p_B(w),                         nếu w ngoài N
    p_new(w) = m_B*q_c*u_B(w|c),               nếu w thuộc N_c

NULL/OTHER quay về phân phối gốc. Giữ mass của N và xác suất ngoài N **tại cùng prefix**, giữ tỉ lệ aliases trong cùng class. Không bảo đảm noun count, length, recall toàn chuỗi vì đổi từ làm prefix tương lai khác. Không tự sửa được quan hệ, thuộc tính, đúng instance hoặc hallucination ngoài lexicon.

Loss đề nghị:

    L = SCST(reward gốc: CIDEr + 0.5*object reward)
        + 0.1*nounNLL(p_new)
        + 0.1*ROI_class_CE
        + 0.1*[-log sum_{i in positive set} pi_ti]

Đây là starting design để bàn, không phải hyperparameter đã validated.

Pilot dự kiến hai nhánh cùng tham số/loss/data/init:

- **Local:** key có appearance của đúng ROI.
- **Global control:** key dùng appearance trung bình của proposals trong ảnh, giữ geometry/confidence; **ROI classifier/lexical values vẫn dùng đúng từng ROI ở cả hai nhánh**. Control không bỏ toàn bộ visual object evidence.
- Can thiệp đánh giá: tráo appearance keys giữa các ROI khác class, giữ query/base distribution/box/confidence/lexical values; kiểm correct-noun likelihood giảm đúng chiều. Hoán vị đồng thời toàn bộ key/value/box phải invariant.

Muốn tiếp tục phải cải thiện caption so frozen C0 và có lợi ích của local correspondence so control; không chỉ thắng A nhờ thừa hưởng C0. Không dùng attention đẹp/low auxiliary loss thay caption outcome.

## 6. Những lỗ hổng cần bạn xem thật kỹ

- Region pointer/copy và visual sentinel đã có. Frozen base + bảo toàn mass có phải một adaptation đáng kể, hay chỉ đóng gói hai safeguard thông thường?
- Hạn chế cụ thể của phương pháp trước là gì? Có bằng chứng/mô hình lý thuyết tối thiểu cho việc bảo toàn mass giúp tránh một failure mode, hay chỉ trực giác “đỡ né nói”?
- Cả Local và Global cùng bảo toàn mass: hai nhánh này không đo lợi ích riêng của constraint. Cần đối chứng gì nếu claim constraint là đóng góp?
- Frozen hidden h_t vẫn có prior của base và ảnh toàn cục; ROI CLIP có scene context. “Không đưa detector class label” chưa chứng minh học local evidence.
- Nếu key là linear additive và score là dot-product, appearance trung bình dùng chung có thể thành một hằng số triệt tiêu giữa ROI trong softmax (NULL có thể tạo ngoại lệ). Kiến trúc control phải được viết rõ; không giả có global conditioning mà thực chất đã bị triệt tiêu.
- NounNLL có thể thúc điền tên object hợp reference nhưng không đúng instance; class presence không phải instance grounding. Pointer-set supervision không tự giải quyết điều này.
- Giữ object mass có thể ngăn bỏ một noun không có bằng chứng khi base vốn muốn nói noun. NULL fallback chỉ quay lại base, không chứng minh tránh hallucination. Có nên giữ constraint này không?
- **Phản ví dụ greedy ngay cùng prefix (bổ sung sau bản đầu):** ba object tokens thuộc ba classes có xác suất gốc (0,4; 0,1; 0,1), EOS=0,3 và phần còn lại=0,1. Giữ object mass=0,6 nhưng phân phối lại thành (0,2; 0,2; 0,2) khiến EOS thắng greedy, dù xác suất EOS không tăng. Vì vậy constraint không bảo đảm tránh kết thúc sớm, ngay cả trước khi xét prefix tương lai khác nhau. Hãy đánh giá lại động cơ thiết kế theo phản ví dụ này.
- Multiword/ambiguous alias coverage có thể làm headroom thấp. Không được hạ đánh giá chỉ xuống nhóm tokens mô hình sửa được để tuyên bố toàn caption tốt hơn.
- Nearest prior nào có probability decomposition/mass-preserving correction tương tự? Một normalization identity không phải theorem mới.

## 7. Tài liệu khởi điểm, không phải danh mục đã chứng minh novelty

Hãy tìm thêm nearest primary papers và đọc đúng phần Method. Không chỉ dựa abstract/snippet; nếu không có full text hãy nói rõ để tôi cung cấp. Không bịa venue/quartile hoặc nói “chưa ai làm” từ một search nhỏ.

- Neural Baby Talk, CVPR2018: https://arxiv.org/pdf/1803.09845 — đã có region pointer, sentinel và lexicalization.
- More Grounded Image Captioning by Distilling Image-Text Matching Model, CVPR2020: https://arxiv.org/pdf/2004.00390 — attention distillation/grounding, không phải ý tưởng mới tự thân.
- FLAIR: https://arxiv.org/html/2412.03561v1 — text-conditioned visual pooling và construction tránh text shortcut; không mặc định caption benefit.
- Align Where the Words Look: https://arxiv.org/html/2509.18369v1 — attention/patch alignment+contrastive+OT preprint; venue/peer-review chưa xác minh. Không coi generic attention+contrastive+OT tự là novelty.
- CSA-Net code gốc: https://github.com/CrossmodalGroup/CSA-Net — cần chọn đúng paper/code version khi đối chiếu; không suy cơ chế từ tên.

Đây chỉ là mốc ban đầu. Tôi cần bạn tìm **phương pháp gần cơ chế nhất**, không cần một danh sách dài paper chỉ liên quan tiêu đề.

## 8. Đầu ra tôi muốn

1. Tóm tắt bạn hiểu nút thắt hiện tại bằng5–8câu; phân biệt bằng chứng và giả thuyết.
2. Chọn3–5nearest works, link primary và phần đã đọc; bảng: họ làm gì / candidate trùng ở đâu / khác ở đâu / khác biệt có ý nghĩa gì. Nếu chưa xác minh được thì ghi chưa xác minh.
3. Kết luận **GIỮ / SỬA / BỎ** ứng viên Astra. Nêu phản biện mạnh nhất, không đồng ý vì nể và không bác chỉ vì primitive từng tồn tại.
4. Nếu cần thay hướng, đưa tối đa2phương án khác nhau thực chất, rồi chọn1. Được kế thừa phương pháp đã có, nhưng phải viết rõ delta; không trả lại một danh sách module để ghép thử.
5. Với hướng chọn: sơ đồ module bằng chữ, tensor/inputs/outputs, nguồn supervision, loss, inference dùng dữ liệu gì, lớp nào học/frozen. Không để một oracle/target tương lai lọt vào inference.
6. Viết một đoạn **đóng góp dự kiến** trung thực và một đoạn nêu chưa được claim gì. Phân biệt novelty với hiệu quả; thắng baseline không tự chứng minh mới, khác kiến trúc không tự chứng minh hữu ích.
7. Đưa phép thử nhỏ nhất cùng baseline/đối chứng quyết định; có cả caption quality/factuality và phép thử correspondence thực sự được dùng. Nêu tiêu chí dừng và giới hạn thống kê; không đề nghị một loạt audit/nhãn người dùng nếu không giải thích quyết định mà chúng thay đổi.
8. Nêu tài nguyên còn thiếu và điều gì phải kiểm trước code. Không viết code, không yêu cầu chạy full hoặc đưa ra cam kết acceptance. Mục tiêu là chốt một giả thuyết đáng làm trước khi tốn GPU, không phải tiếp tục brainstorm không có quyết định.

Hãy kết thúc bằng một khuyến nghị duy nhất: phương pháp nào đáng ưu tiên, khác biệt nào đáng kiểm và bước tiếp theo cụ thể là gì.
