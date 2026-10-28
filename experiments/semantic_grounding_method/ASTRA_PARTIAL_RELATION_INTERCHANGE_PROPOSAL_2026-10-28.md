# Đề xuất: học hoán chuyển bằng chứng quan hệ với giám sát từng sự kiện

> CẬP NHẬT CÙNG NGÀY: khuyến nghị triển khai làm phương pháp chính đã được rút sau kiểm RAVEL/MDAS và các nearest methods. Xem ASTRA_NOVELTY_REAUDIT_DECISION_2026-10-28.md. Nội dung dưới đây giữ như proposal lịch sử; chưa được chốt để chạy.

28/10/2026. Astra, đề xuất nghiên cứu để phản biện; chưa code, chưa launch GPU. Tài liệu này thay khuyến nghị phương pháp trong ASTRA_BGRM_DISCUSSION_2026-10-28.md; không sửa kết quả hoặc quyết định PRETRAIN_GATE_NO_GO của RelPos/RelContrast200.

## 1. Quyết định và phạm vi đóng góp

**Tôi chọn học một giao diện quan hệ mà khi hoán chuyển phần bằng chứng tương tác giữa hai ảnh có cùng thực thể, quyết định ngôn ngữ về quan hệ phải chuyển theo nguồn bằng chứng; thông tin thực thể của ảnh nhận được giữ ổn định.**

Đây là hướng kiến trúc cộng mục tiêu học, không chỉ học xếp ảnh đúng cao hơn ảnh sai. Không dùng GRU hay truth classifier làm cầu nối trước caption decoder. Không cần tạo một ảnh phản thực hoặc bịa caption phản thực đầy đủ.

Đóng góp nhỏ có thể nhắm tới: **căn chỉnh có kiểm soát giữa một thành phần biểu diễn tương tác và một cụm quan hệ trong caption, học từ nhãn sự kiện đúng–sai không đầy đủ, rồi kiểm tra chuyển giao sang mô tả tự do.** Đây là ứng viên đóng góp, chưa phải kết luận mới hoặc đủ để được nhận bài.

Ba việc phải tách:

- Định vị: hai referent được nối với hai instance detector; vùng bao chỉ là nơi lấy bằng chứng.
- Ngữ nghĩa: thứ tự chủ thể–đối thể và tương tác được biểu diễn riêng với thông tin thực thể; union box không phải nhãn predicate.
- Sử dụng: hoán chuyển thành phần tương tác phải làm thay đổi likelihood của cụm quan hệ qua chính decoder; cuối cùng phải sửa được caption trên ảnh thật.

### Tối đa hai hướng

| Ứng viên | Thay đổi thực chất | Lợi ích | Điểm yếu | Quyết định |
|---|---|---|---|---|
| Giữ A, contrast likelihood giữa hai ảnh thật | Đổi mục tiêu và nguồn phản chứng, không đổi giao diện biểu diễn | Ít triển khai, trực tiếp cập nhật decoder, control rõ | Có thể dùng mọi khác biệt ảnh; phương pháp rất gần contrastive captioning đã có | Là đối chiếu ý tưởng, không phải khuyến nghị chính để nhắm đóng góp |
| Hoán chuyển có giám sát phần tương tác, giữ phần thực thể/ngữ cảnh | Tách giao diện đọc quan hệ và ràng buộc đáp ứng khi đổi một thành phần | Đặt ra hành vi có thể bác bỏ về cách decoder dùng quan hệ, vượt mục tiêu ranking ảnh đơn thuần | Hybrid latent có thể ngoài phân phối; CLIP có thể mang nền; tăng chi phí và không bảo đảm novelty | **Chọn một pilot hữu hạn** |

Không tự chạy thêm ứng viên thứ nhất nếu pilot thứ hai thất bại.

## 2. Công trình gần nhất và ranh giới tính mới

Đã đọc phần phương pháp của các primary sources sau:

1. [IIT, Geiger et al., ICML 2022, §3](https://proceedings.mlr.press/v162/geiger22a/geiger22a.pdf): huấn luyện neural model đáp ứng theo một mô hình mức cao khi thay trạng thái bên trong bằng trạng thái từ source input. Hoán chuyển trạng thái để ép model sử dụng một biến **là kế thừa IIT**.
2. [Strict IIT / InterpBench, NeurIPS 2024, §3 và Algorithm 1](https://proceedings.neurips.cc/paper_files/paper/2024/file/a8f7d43ae092d9a5295775eb17f3f4f7-Paper-Datasets_and_Benchmarks_Track.pdf): thêm ràng buộc khi can thiệp các node không thuộc circuit. Vì vậy, giữ ổn định phần không liên quan cũng không phải ý tưởng mới tự thân.
3. [See or Guess, ACM MM 2024, §4.1–4.4](https://arxiv.org/html/2408.16809v1): che vùng thực thể và dùng TE/NDE regularization lên xác suất từ thực thể của captioner. Không thể tuyên bố việc thay bằng chứng thị giác để điều chỉnh likelihood caption là mới.
4. [Evidence-RL, 2026, §3.1–3.2](https://arxiv.org/html/2608.08021v1): so mức giảm support khi neutralize vùng evidence với các vùng tham chiếu cùng quy mô rồi đưa vào correctness-anchored GRPO. Việc thêm vùng đối chứng hoặc mở rộng sang nhiều vùng không đủ làm ranh giới novelty.
5. [MTTSNet, CVPR 2019, §3.1–3.2](https://arxiv.org/pdf/1903.05942): subject/object/union, hình học tương đối và relational captioning đã có tiền lệ. Các thành phần này được kế thừa.

Khác biệt cụ thể cần kiểm chứng của đề xuất: thay vì chỉ xóa evidence rồi đòi support giảm, dùng **hai truth values của cùng assertion** để yêu cầu chuyển quyết định theo nguồn tương tác trong cả hai ngữ cảnh nhận; chỉ giám sát phần sự kiện đã biết, không giả vờ biết toàn bộ caption phản thực. Đầu ra kiểm tra là cùng language decoder dùng để caption ảnh thật.

Đây là một áp dụng có cấu trúc của interchange learning vào relational captioning với nhãn không đầy đủ. Chưa có cơ sở gọi interchange, selectivity, pair representation hoặc loss softplus là phát minh mới. Chưa rà hết literature để chứng nhận không có phương pháp trùng. CofactGVR (ICML 2026) đã thấy abstract nhưng chưa đọc được full text; không dùng nó để khẳng định một khác biệt phương pháp đã được xác minh.

## 3. Pipeline và tensor

Ký hiệu B=batch size, K≤12 detector objects, M≤K(K−1) ordered pairs, D=512. Dùng mọi cặp có thứ tự hợp lệ trong top-K đã khóa, thay vì top-M theo điểm HOI vốn có thể bỏ cặp không có người. Padding được mask. Không có relation label/GT box làm input inference.

### Đường A

Từ cache V có shape B×197×768 và original semantic-prompt cache, A tạo memory C có shape B×(N_prompt+32)×512. Giữ đường này; không nối trực tiếp thêm relation slots vào memory cũ.

### Giao diện thực thể và tương tác

Với mỗi cặp (i,j):

- Fractional ROI mean trên 196 patch, bỏ CLS, cho hai vector thị giác 768 chiều. Shared linear tạo e_i,e_j, mỗi vector 256 chiều.
- Pair attention trong enclosing union rectangle dùng query từ e_i,e_j; key/value lấy patch và hai role masks riêng. Tạo u_ij có 256 chiều. Kế thừa cách đọc vùng có thứ tự của frame encoder trước, không dùng GRU/truth head.
- G_ij có 8 chiều: signed relative geometry và overlap đã có trong pipeline.
- Interaction value z_ij = MLP([u_ij,G_ij]) có 128 chiều. Không ghép trực tiếp hai vector entity appearance vào value; đây là một hạn chế giao diện mong muốn, không chứng minh đã tách hoàn toàn semantics.
- Pair address k_ij = Linear([e_i,e_j,class_i,class_j,box_i,box_j]) có 512 chiều. Address và entity information thuộc ảnh nhận; chỉ interaction value của cặp được hoán chuyển. Geometry trong address vẫn là một đường có thể chứa thông tin quan hệ và phải được thừa nhận.

Một reader song song với cross-attention tại decoder layer cuối:

delta_h = W_out Attention(query=h, keys=k_ij, values=W_value z_ij)

h_new = h + delta_h

Đầu ra vẫn đi qua decoder/output vocabulary của A. Khởi tạo W_out=0, không thêm outer LayerNorm có thể đổi A ở bước0. Kiểm parity thực tế trong preflight, không coi công thức zero init là bảo đảm chất lượng sau train.

Interaction values có shape B×M×128; keys B×M×512; reader output B×T×512; logits B×T×10000.

Frozen: CLIP/cache, visual projection A, Q-Former, prompt fusion, input embedding, vocabulary projection, decoder layer đầu. Trainable: hai decoder layers cuối, pair-pooling projections/interaction MLP/address projection và reader mới. Cả control và proposed dùng đúng cùng tập tham số và initialization từ A; không pretrain thêm một auxiliary captioner, không lấy Contrast200 làm một semantic checkpoint đã được chứng minh.

Giảm embedding xuống128 hoặc zero init không phải phần đóng góp mới.

## 4. Giám sát và ý nghĩa của hoán chuyển

Mỗi training pair có q=(s,r,o), I+ hỗ trợ q và I− mâu thuẫn q. VSR cung cấp truth label; COCO GT chỉ dùng kiểm referent và matching lúc train; **input ROI dùng predicted boxes**. Hai lớp được nhắc không đủ thành nhãn quan hệ.

Giữ C và mọi entity/address từ ảnh nhận a; thay duy nhất z của target pair bằng z từ ảnh nguồn b. Bốn cấu hình là:

| Ảnh nhận | Nguồn tương tác | Ràng buộc mong muốn đối với q |
|---|---|---|
| + | + | hỗ trợ |
| + | − | giảm support |
| − | + | tăng support |
| − | − | không hỗ trợ |

Hai đường chéo là dữ liệu ảnh thật có nhãn. Hai ô chéo là **intervention trong model**, không phải ảnh đã chỉnh sửa có nhãn vật lý được người chấm xác nhận. Việc muốn q đi theo nguồn là giả thuyết phân rã để huấn luyện. Không được lấy nhãn này làm bằng chứng độc lập rằng hybrid tương ứng một cảnh thật.

Không có nhãn cho predicate khác ⇒ unknown. Không có q trong caption tham chiếu ⇒ chưa được mô tả, không phải false. Không bịa quan hệ phủ định bằng role swap hoặc sửa giới từ. Không tạo caption phản thực hoàn chỉnh; không giả định các thuộc tính hoặc quan hệ chia sẻ endpoint đều bất biến.

Inference: C, addresses và z đều từ chính ảnh đang caption; không donor, không q cung cấp trước, không VSR label hoặc GT. Caption được sinh trực tiếp, không reranking.

## 5. Loss cụ thể đi qua caption decoder

Với cùng subject prefix của q, đặt ell_ab là mean log probability của **toàn bộ predicate+object tail** dưới C_a và z_b. Không phạt riêng predicate khi object đích chưa có trong prefix. Tail score được tính bằng đúng decoder dùng inference.

Control ordinary contrast trên ảnh thật:

L_real = −ell_++ + softplus(ell_-- − ell_++)

Proposed intervention objective:

L_transfer = −(ell_++ + ell_-+)/2
             + [softplus(ell_+- − ell_++) + softplus(ell_-- − ell_-+)]/2

Vế positive tạo áp lực duy trì assertion được hỗ trợ; vế ranking yêu cầu cùng một receiver context đổi quyết định theo source interaction. Không đọc truth logits từ một classifier riêng.

Entity preservation: dùng hai standalone entity-name sequences của s,o, không chứa predicate. Tính KL giữa phân phối factual của model (stop-gradient) và phân phối sau interchange ở các prefix của hai sequence này:

L_keep = mean KL(stopgrad p_theta(. | C_a,z_a,entity_prefix)
                 || p_theta(. | C_a,z_b,entity_prefix))

Tên thực thể tồn tại ở cả hai ảnh theo kiểm referent. Loss này chỉ ràng buộc tính ổn định khi đọc danh từ, **không chứng minh giữ được mọi nội dung không liên quan trong free caption**.

Hai arm cùng dùng:

L_common = L_SCST với reward gốc của A + 0.1 KL_to_A trên factual caption prefixes

Đề xuất khóa trước pilot:

- Control C: L_common + 0.2 L_real.
- Full F: L_common + 0.2 L_transfer + 0.1 L_keep.

Tất cả thành phần được mean theo example/token/pair để không tăng trọng số chỉ vì có thêm các ô hybrid. Các hệ số là lựa chọn thử nghiệm hữu hạn, không có bảo đảm lý thuyết hoặc tuyên bố tối ưu. Khóa source/config trước run, không chọn lại trên dev.

Match cả số cập nhật, caption/VSR examples, initialization, trainable set và teacher-forced forward count: control lặp factual configurations theo cùng số lượt thay vì nhận ít compute hơn F. Các entity-prefix forwards cũng được thực hiện cho control; self-reference KL của nó bằng0. Giữ cùng generation và SCST/KL settings.

Ở một receiver context và cùng prefix, address và mọi đường A đều không đổi. Nếu z bị bỏ qua hoàn toàn thì hai likelihood theo hai source trùng nhau, không thể thỏa margin. Đây là ràng buộc sử dụng **tại giao diện model**; nó vẫn có thể được thỏa bằng thông tin sai nguồn nằm trong z. Không suy từ đó ra causal grounding trong ảnh thật.

## 6. Shortcut và phép thử bác bỏ

| Phản ví dụ | Kiểm tra cần có | Kết luận nếu xảy ra |
|---|---|---|
| u trong union mang dấu hiệu nền/global image chứ không phải tương tác | Giữ receiver và geometry receiver, đưa donor background feature qua cùng value encoder; so donor-following với donor interaction thật, chuẩn hóa cùng quy mô/norm. Chỉ làm ở ca có background control hợp lệ | Nếu sham đạt gần real (cách≤5pp), không có bằng chứng đủ cho claim evidence-pair specificity; không cứu bằng heatmap |
| Chỉ cần geometry, u không có ích | Inference intervention thay u, giữ nguyên geometry của donor; báo hai strata cố định | Không claim học interaction appearance nếu kết quả chỉ do geometry; không bỏ contact để đổi overall gate |
| Mô hình nhận ra sự bất thường của hybrid rồi đổi câu tùy tiện | Self-interchange phải parity; kiểm same-truth donor nơi có nhãn phù hợp; kiểm entity stability và caption ảnh thật | Đổi xác suất không đúng chiều truth hoặc làm hỏng danh từ bác bỏ cơ chế mong muốn |
| Decoder học template nhưng sinh tự do tránh quan hệ | Đo đồng thời true-assertion coverage và false-assertion rate trong free captions | Ranking gain đơn độc không qua gate |
| Các đường A/address còn mang quan hệ nên relation slot không cần thiết | Bảng2×2, tắt/shuffle đúng target value; address giữ nguyên | Nếu output không theo target value thì giao diện chưa điều khiển được quyết định |
| Donor value mã hóa image ID hoặc pair ID | Image-disjoint heldout; không input pair ID/truth/relation label vào encoder; thêm cùng-class donor checks | Train fit hoặc donor classification không phải semantic transfer |

CLIP patch đã được contextualize toàn ảnh; ROI mean không tạo ra feature region-pure. Các sham tests ở trên là phép bác bỏ một số shortcut, không phải bằng chứng hoàn chỉnh rằng mọi leakage đã biến mất. Độ nhạy với một latent intervention không tự cho phép tuyên bố một quan hệ nhân quả vật lý.

## 7. Pilot nhỏ nhất và gate đặt trước

Một pilot, hai arm C/F; A gốc là mốc cố định. Không chạy thêm nhánh pretraining, gate hoặc crop.

- Dùng frozen256TRAIN VSR pairs và captionTRAIN2k đã có; giữ mọi image-exclusion của protocol trước. Evaluation64pairs cũ chỉ là reused exploratory diagnostic, không đổi tên thành tập xác nhận độc lập.
- 500 optimizer updates mỗi arm, một seed khóa trước; đề xuất global batch8 caption images và4VSR pairs/update, chia/accumulate cho hai T4. Đây là cấu hình cần model preflight, chưa được chạy hay chứng minh tốc độ.
- Trần đề xuất60phút/arm trên T4×2, kể cả preflight/evaluation neural: tối đa4device-GPU-hours cho hai arm. Chưa xác nhận lại quota hiện có; không hứa500updates sẽ hoàn thành trong trần. Không hoàn tất ⇒ computational-incomplete, không kết luận semantic failure hoặc tự kéo dài.
- Chỉ endpoint500, cùng generation/SCST settings. Caption dev1000 dùng đúng A captions/reference/evaluator đã khóa; so paired differences C/F/A.
- Blind semantic panel: chọn32 trong64heldout pairs bằng thứ tự hash đã khóa,16pairs/stratum, trước khi nhìn caption mới. Chấm A/C/F trên64ảnh, tổng192captions. Shuffle model identity; đánh dấu caption có entail q không, chấp nhận paraphrase/inverse tương đương đã kiểm, không dùng CIDEr chọn winner.

Định nghĩa trên32pairs: RC=true-image fraction có caption entail q; FR=false-image fraction có caption entail q; Y=RC−FR. Không nói q không bị tính sai trên ảnh dương, nhưng không được cộng semantic coverage. Các assertion ngoài q chưa có nhãn không tự gán đúng hoặc sai. Trường hợp chấm ambiguous được báo riêng; gate không được hưởng lợi từ việc loại chúng.

**Go để đầu tư xác nhận tiếp chỉ khi tất cả đạt:**

1. Trên64reused pairs, tỷ lệ cả hai receiver contexts đều rank source-positive cao hơn source-negative đạt≥70%, hơn C≥10pp; paired95%CI cho gain có lower>0.
2. Free-caption panel: Y_F hơn cả A và C≥0.125 (ít nhất4net pair-equivalents trên32); paired stratified95%CI lower>0; RC_F không giảm so với A và FR_F không tăng so với A.
3. Dev1000: CIDEr_F−A≥0.005 và CIDEr_F−C≥0.003, paired95%CI lower>0 cho cả hai; CHAIRs_F≤A; object recall_F≥A−0.005.
4. Không có bằng chứng sham background gần bằng real theo kiểm tra đã nêu; entity preservation không hỏng. Nếu thiếu các control hợp lệ để kiểm specificity, chưa được nâng claim thành local grounding.

Đây là ngưỡng ưu tiên đầu tư do tôi đề xuất, không phải hằng số khoa học. Cần khóa nguyên văn trong run protocol trước train. Diagnostic thuận lợi không thay thế các gate caption. Nếu bất kỳ gate chính nào fail, dừng recipe cho đợt này; không chọn subgroup/epoch hoặc hạ ngưỡng để cứu.

Nếu pass, vẫn cần xác nhận trên dữ liệu/seed chưa dùng chọn phương pháp trước khi viết claim khái quát. Dev1000 và64pairs đều có lịch sử sử dụng, nên CI không xóa được selection bias. Không tuyên bố đã có positive result hoặc sẽ xong paper trước31/10.

## 8. Vì sao tôi đổi khuyến nghị

Đề nghị giữ nguyên A và học direct contrast trước đó giải quyết một điểm kỹ thuật thật: gradient phải đi qua decoder. Nhưng nó chưa xác định được model học khác biệt ngữ nghĩa nào và đóng góp gần với contrastive captioning cũ.

Đề xuất mới có đối tượng học rõ hơn: **một thành phần tương tác phải điều khiển quyết định quan hệ khi nguồn bằng chứng bị đổi, còn phần đọc thực thể được bảo toàn**. Tôi chọn nó vì đưa ra một hypothesis về giao diện biểu diễn–ngôn ngữ có thể bị bác bỏ, không vì novelty đã được bảo đảm.

Giới hạn mạnh nhất còn nguyên: nhãn truth trên hai ảnh thật không xác minh toàn bộ thế giới phản thực của hybrid latent, và features CLIP có thể rò thông tin toàn ảnh. Vì thế phần có thể bảo vệ ban đầu là một phương pháp ràng buộc/đánh giá interchange ở mức sự kiện với kiểm tra caption transfer; các tuyên bố tách nhân quả hoặc grounding cục bộ mạnh hơn phải đợi bằng chứng riêng.
