# Khuyến nghị phương pháp sau đợt đọc mở rộng

28/10/2026. **Chọn một pilot có giới hạn cho cơ chế đọc chung thực thể–quan hệ ngay trong caption decoder.** Đề xuất này thay khuyến nghị chỉ audit 30 case ở bản reaudit trước; không khôi phục partial interchange, không đổi NO_GO của frame200. Chưa có code hay lần chạy mới.

Bằng chứng và phạm vi đọc ở [ledger](ASTRA_SEARCH_AND_READING_LEDGER_2026-10-28.md), metadata 20 bài ở [CSV](ASTRA_LITERATURE_20_2026-10-28.csv). Những con số/hyperparameter dành cho pilot bên dưới là **đề nghị khóa trước thử nghiệm**, không phải kết quả đã đạt hay giá trị tối ưu được chứng minh. Bản BGRM 25/10 đính kèm là đề xuất cần phản biện; lịch chạy và quota trong đó không được xem là bằng chứng readiness hay lệnh triển khai.

## 1. Ba việc phải tách riêng

| Việc | Câu hỏi | Kiểm bằng gì | Chưa chứng minh điều gì |
|---|---|---|---|
| Tìm vùng | Có đọc đúng vật/cặp vật không? | Predicted-box coverage, endpoint matching, probe vùng | Đúng vùng không suy ra đúng quan hệ |
| Biểu diễn ý nghĩa | Biết vai trò, hướng, predicate; phân biệt cùng assertion trên ảnh thật/ảnh sai? | Same-text paired likelihood; kiểm predicate riêng | Teacher forcing tốt chưa suy ra sinh tự do tốt |
| Decoder sử dụng | Thông tin ấy có đổi câu theo bằng chứng? | Caption tự do, đúng/sai quan hệ, coverage và can thiệp reader | Câu khác hay heatmap đẹp chưa là quan hệ đúng |

A có CIDEr dev 1.205512599, COCO-R 1.212063, VG-G 1.208401. Hai nhánh sau chỉ có trung bình dev nhỉnh hơn A; chưa có CI đủ để gọi “vượt trội nhất quán”. Frame200 cho 40/64 so 41/64, delta 1.5625pp với 95% CI[-12.5,+15.625]pp; chưa có gain caption. Truth head và GRU không nhận cùng đường gradient trực tiếp: đây là hạn chế coupling đã thấy, không chứng minh encoder không có thông tin.

Các bằng chứng ấy khuyến khích kiểm **cách đọc và dùng bằng chứng**. Chúng chưa chứng minh A lỗi chủ yếu do trộn referents, cũng chưa chứng minh một kiến trúc OT là cần thiết.

## 2. Hai hướng khác nhau, chọn một

| | Hướng 1: phân phối chung trên cặp trong forward | Hướng 2: teacher cụm đầy đủ, student chỉ thấy prefix |
|---|---|---|
| Thay đổi chính | Cách decoder đọc đối tượng và tương tác; thông tin cặp có thể cập nhật lựa chọn thực thể | Cách học router: teacher xem cụm ở TRAIN, distill sang router prefix lúc sinh |
| Cơ chế | Giải một posterior trên cặp, dùng cả marginal lẫn joint làm context | Grounder theo phrase tạo phân phối cặp/null; student học KL cộng caption/pair loss |
| Điểm mạnh | Delta forward xác định được; cùng posterior ảnh hưởng trực tiếp logits | Ít đổi A; nối được mức phrase với quyết định từng bước |
| Điểm yếu | Solver, collapse; có thể chỉ là structured attention thích nghi cho caption | Teacher tương lai dễ tạo target không dự đoán được từ prefix; rất gần distillation cũ |
| Prior gần | GOT; Sinkformers; KL-relaxed OT; multi-stream caption reader | [Distilling Image-Text Matching2020](https://arxiv.org/pdf/2004.00390) và pooling có điều kiện text |
| Lý do không chọn hướng kia | — | Delta chủ yếu noun→pair và đổi teacher; hiện khó bảo vệ hơn hướng 1 |

**Chọn hướng 1 để thử, không coi đã có một phương pháp mới được xác nhận.** Nó có một phần thay đổi kiểm được bằng đối chứng mạnh; không phải đổi tên loss hay thêm gate. Nếu chỉ hơn A nhưng không hơn đối chứng cùng feature/loss, không gọi đó là đóng góp của cơ chế.

## 3. Delta cần bảo vệ

> So với các nhánh attention đọc bằng chứng riêng và OT dùng để regularize alignment, thay bằng một posterior cặp phụ thuộc prefix, có prior thực thể được sửa bởi tương tác; cả context thực thể lẫn context quan hệ đều lấy từ posterior ấy để sinh từ.

Đối chứng C* cũng có đọc chung và feedback, nên hai điều đó không được dùng để giải thích riêng F thắng C*. Delta hẹp của F là phạt phụ thuộc giữa hai vai nhẹ hơn, trong khi giữ cùng lực neo và độ mềm của từng marginal; phép viết lại objective ở §6 làm rõ điều này.

Nearest priors bắt buộc thảo luận: [GOT](https://proceedings.mlr.press/v119/chen20e.html), [Sinkformers](https://proceedings.mlr.press/v151/sander22a.html), [GRIT](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960165.pdf), [OTF-CBM](https://arxiv.org/abs/2606.26891). Xem mục phương pháp cụ thể trong ledger, không coi “chưa thấy cùng pipeline” là chứng minh mới.

**Kế thừa:** ROI/union features, ordered roles, relative geometry, attention, entropy/KL, transport solver, residual injection, SCST và distillation đều không mới. Subject/object/union cũng đã có trong [MTTSNet](https://arxiv.org/pdf/1903.05942).

**Có khả năng là delta nhỏ:** dùng nghiệm chung trong forward của caption decoder, để bằng chứng tương tác sửa hai phân phối thực thể rồi cùng chi phối logits; kiểm riêng cách phân bổ regularization giữa marginal và phụ thuộc cặp so với joint attention một lần đã cân bằng. Chưa xác nhận không có bài khác làm đúng cơ chế này; đây là ứng viên về kiến trúc/readout, không phải OT hoặc loss mới.

## 4. Module và tensor

### 4.1 Đầu vào và biểu diễn

- A giữ memory gốc \(C_A\in R^{B\times(N_p+32)\times512}\): prompt gốc và 32 Q-Former queries; giữ visual cache \(V\in R^{B\times197\times768}\).
- Tối đa \(K=12\) box **dự đoán**, lớp 80-way và confidence có sẵn. TRAIN có GT để match nhãn; inference chỉ dùng predictions. Không giả định UPT cần có mặt.
- Fractional ROI pooling từ 196 patch cho \(e_i\in R^{512}\), kết hợp class embedding và box. Không chạy CLIP/crop lại.
- Mỗi cặp có thứ tự \(i,j\): union-patch pooling với hai role masks phân biệt vùng i/j; ghép \(e_i,e_j\), pooled interaction và geometry 8→MLP→\(r_{ij}\in R^{512}\). Cùng encoder này dùng cho mọi đối chứng.
- Thêm một null key có value 0. Self/null pairs có interaction value 0; chúng phục vụ từ chức năng, noun hoặc lúc không có cặp phù hợp, không phải nhãn “ảnh không có quan hệ”.

Mức căn chỉnh chọn: **đối tượng vào vùng riêng; cụm mô tả có thể dùng tổng hợp vài vùng; quan hệ vào cặp có thứ tự và phần tương tác**. Union không tự chứa đầy đủ ý nghĩa: cần hai vai và appearance/hình học. Không ép “under” có bbox riêng; cũng không ép mọi bước sinh phải dùng hai vật.

CLIP patch đã có receptive field toàn ảnh. ROI/union pooling chỉ định vị phép đọc, không tạo bằng chứng cục bộ hoàn toàn độc lập với bối cảnh.

### 4.2 Posterior và phản hồi sang thực thể

Tại bước t, \(h_t\in R^{512}\) chỉ phụ thuộc ảnh và prefix đã sinh. Hai role queries cho priors \(\alpha_t,\beta_t\) trên \(K+1\) keys; query tương tác cho scores \(S_{tij}\), chẳng hạn cosine giữa query và projected pair key. Không đưa embedding của target phrase vào forward.

Với \(\Delta=\{T\ge0:\sum_{ij}T_{ij}=1\}\), đặt:

\[
T_t=\arg\max_{T\in\Delta}
 \langle T,S_t\rangle+\epsilon H(T)
 -\lambda_s KL(T\mathbf1\|\alpha_t)
 -\lambda_o KL(T^\top\mathbf1\|\beta_t).
\]

Softmax priors có floor nhỏ rồi chuẩn hóa; không mask support theo GT hay target class. Đây là bài toán entropy với KL biên đã có tiền lệ, không đề xuất solver mới. Tổng mass được cố định bằng 1: không gọi nó là UOT co mass để “abstain”; null key mới mang vai trò không dùng cặp.

Dùng differentiable optimization/unrolled scaling phù hợp đúng objective; không dùng Sinkhorn hàng/cột cứng rồi gọi là giải bài toán mềm. Đề nghị bắt đầu cosine scores, \(\epsilon=.1,\lambda_s=\lambda_o=.2\); kiểm độ chính xác solver trên TRAIN trước so sánh. Số vòng lặp/tolerance phải xác nhận ở preflight, không suy từ con số3–5 vòng của Sinkformers sang objective này.

\[
\hat\alpha=T_t\mathbf1,\quad \hat\beta=T_t^\top\mathbf1,
\]
\[
c_s=\sum_i\hat\alpha_i e_i,\quad
c_o=\sum_j\hat\beta_j e_j,\quad
c_r=\sum_{ij}T_{tij}r_{ij}.
\]

Ghép ba context→projection 512→cộng vào hidden ở layer cuối→vocabulary head thật của A, logits \(B\times L\times10000\). Không đặt một truth classifier thay caption decoder. A memory vẫn được đọc như trước.

### 4.3 Vì sao sửa ý tưởng ép bằng nhau

Bản nháp trước ép \(T1=\alpha,T^\top1=\beta\). Tôi rút lựa chọn đó: nếu \(\alpha\) one-hot thì \(T\) đã cố định thành \(\alpha\beta^\top\); score tương tác không còn quyền đổi lựa chọn. Khi hai prior sắc, bài toán gần thoái hóa.

Bản mềm cho phép tương tác sửa cả marginal, đồng thời trả posterior ấy về context thực thể. Đây là lý do thay thiết kế, không phải một kết quả đã chạy. Cả prior và score đều có thể sai; sửa được không đồng nghĩa sửa đúng.

Ngay cả posterior chung cũng không khóa literal binding: phân phối .5(dog,table)+.5(cat,chair), sau pooling/phi tuyến, vẫn có thể sinh “dog under chair”. Vì vậy không tuyên bố loại bỏ mọi caption ghép chéo hay bảo đảm truth.

## 5. Loss, nguồn giám sát, tham số học

TRAIN hiện có VSR 256 cặp, 484 ảnh; 64 cặp held với 128 ảnh đã được xem chỉ dùng exploratory. Mỗi cặp có cùng assertion \(q=(s,r,o)\) trên ảnh \(I^+\) đúng và \(I^-\) sai; hai thực thể có mặt ở cả hai. COCO caption và GT boxes chỉ dùng phía TRAIN. Những quan hệ khác chưa có nhãn là **unknown**, không gán âm.

Giữ SCST với reward đúng như A, thêm KL neo tới A. Cho \(\ell^\pm\) là mean log probability **toàn predicate+object-tail** của cùng assertion qua chính decoder A:

\[
L_{\rm pos}=-\ell^+,\quad
L_{\rm pair}=\operatorname{softplus}(\ell^- -\ell^+).
\]

Tính riêng predicate log-prob để phát hiện gain chỉ từ noun. Không CE toàn assertion sai như một target dương. Endpoint supervision trên các bước tail đã đánh dấu:

\[
L_{\rm end}=-\log\alpha_t(i_s^*)-\log\beta_t(i_o^*).
\]

Dùng endpoint loss cho cả hai ảnh vì nhãn hai thực thể hợp lệ trong cả hai; index GT chỉ ở loss. Nếu không match được predicted box thì mask loss ấy và báo coverage; không lặng lẽ thay bằng GT region.

Đề nghị chung cho các arm:
\[
L=L_{\rm SCST(A)}+.1KL(p_A\|p_\theta)
  +.2(L_{\rm pos}+L_{\rm pair})+.1L_{\rm end}.
\]
Lấy KL trên prefix chung; stop-gradient teacher A. KL direction/hệ số phải như nhau giữa các arm. Đây là bộ hệ số cố định đề nghị, không có tuyên bố là loss mới.

Học new projections/queries/pair encoder, role embeddings, output projection và hai decoder layer cuối; giữ frozen CLIP, visual projection, Q-Former, prompt fusion, input embedding, vocabulary head và decoder layer đầu. Cùng trainable scope ở các đối chứng. Output projection khởi tạo 0 giúp parity lúc0, không bảo đảm giữ nguyên A sau train; gradient nhánh trong có thể chỉ bắt đầu sau output projection, cần kiểm ở preflight.

Nguồn loss và đường dùng decoder được nối trực tiếp. “Có gradient” vẫn chưa đủ: mô hình có thể học né assertion khi sinh tự do.

## 6. Đối chứng và phép thử nhỏ nhất

### Hai arm đầu tiên, không mở sweep

Giữ A làm mốc có sẵn, train cùng init/data/steps/loss cho:

- **C*:** joint attention một lần đã cân bằng \(T^{(0)}_{ij}\propto(\alpha_i\beta_j)^{\lambda/(\lambda+\epsilon)}\exp[S_{ij}/(\lambda+\epsilon)]\), với \(\lambda_s=\lambda_o=\lambda\); cũng trả marginal về entity contexts.
- **F:** posterior KL biên mềm ở trên, cùng context/readout.

C* là đối chứng mạnh và rẻ hơn, đã có joint binding và feedback. F chỉ được nhận công cho **cách regularize phụ thuộc giữa hai vai**, không nhận công của pair features, nhiều tham số, dữ liệu thêm, feedback hoặc làm mềm prior vốn C* cũng có. Nếu chỉ F>A nhưng F≈C* thì không giữ claim cho phép tối ưu posterior.

Phản biện của Astra phát hiện đối chứng ban đầu \(T\propto\alpha\beta\exp(S/\epsilon)\) chưa công bằng: ngay khi \(S=0\), F cho marginal tỉ lệ \(\alpha^{\lambda/(\lambda+\epsilon)}\), \(\beta^{\lambda/(\lambda+\epsilon)}\). F có thể thắng chỉ do làm mềm prior. Vì vậy thay arm C ban đầu bằng C* bên trên, không thêm một lần train.

Có thể thấy delta rõ bằng cách viết \(m=T1,n=T^\top1\), \(I(T)=KL(T\|mn^\top)\):

\[
J_F=\langle T,S\rangle+(\epsilon+\lambda)[H(m)+H(n)]
-\epsilon I(T)+\lambda[\mathbb E_m\log\alpha+\mathbb E_n\log\beta],
\]
\[
J_{C^*}=\langle T,S\rangle+(\epsilon+\lambda)[H(m)+H(n)]
-(\epsilon+\lambda)I(T)+\lambda[\mathbb E_m\log\alpha+\mathbb E_n\log\beta].
\]

Hai arm có cùng lực neo và entropy marginal; F phạt phụ thuộc giữa hai vai nhẹ hơn. Với score cộng tách \(S_{ij}=a_i+b_j\), nghiệm hai arm trùng nhau; đây là sanity case phải kiểm ở preflight. Score cần chứa tương tác không phân rã thì delta mới có chỗ tác động. Đây là phân tích objective, không phải bằng chứng novelty hay kết quả train.

Nếu hai arm này qua gate, ablation tiếp theo mới là **C_ind**: priors đọc entity riêng, softmax trên S đọc pair riêng. Arm đó xét lợi ích của feedback so với đọc độc lập, không bắt buộc mở ngay ba lần train.

### Pilot đóng

1. Metadata preflight: xác nhận đúng SHA A \`630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901\`, original prompt/cache/predicted boxes; kiểm split không giao ảnh TRAIN/HELD, coverage và tensors cho eval. Không thay prompt gốc bằng class prompt.
2. TRAIN dùng đúng 256 cặp/484 ảnh đã đủ cache v105 và COCO references của các ảnh ấy; chưa giả định có thêm 2.000 ảnh đầy đủ tensors. Mỗi update có caption batch và VSR batch, cùng lịch cố định. Một seed khóa trước, 500 updates/arm, không early-pick theo dev.
3. Đánh giá free caption trên dev 1.000 đã có **nếu preflight xác nhận đủ các tensor/box của reader**. Nếu thiếu, không dùng GT boxes thay và không coi pilot đã hoàn tất. Không đọc lại ZIP hay chạy feature extraction ở bước đề xuất này.
4. Một panel 32 cặp giữ từ held 64 trước khi xem caption mới, 16 mỗi strata bằng hash khóa sẵn; cả hai ảnh/cặp, captions A/C*/F được giấu tên khi chấm. Ghi assertion đúng/sai/không đề cập/uncertain; không coi mọi câu ngoài assertion là sai.
5. Cửa sổ dự kiến tối đa 60 phút trên T4×2 mỗi arm kể cả kiểm cuối. Đây là cap, không phải ước lượng tốc độ đã đo; quota hai tài khoản chưa xác minh. Hết cap mà chưa hoàn tất 500 updates thì “incomplete”, không kéo dài để cứu số.

Với 32 cặp, đặt RC là tỉ lệ ảnh đúng mà caption đề cập q, FR là tỉ lệ ảnh sai mà caption vẫn khẳng định q, Y=RC−FR. Tăng Y nhưng giảm RC mạnh vẫn có thể là né nói; phải báo cả hai. Mẫu có uncertain giữ nguyên và báo worst/best bounds.

### Gate đề nghị để tiếp tục

Chỉ tiếp tục nếu đồng thời:

- F−A CIDEr ≥.005 và F−C* ≥.003 trên thang hiện tại 1.x; lower bound 95% paired bootstrap của cả hai >0. Đây là ngưỡng thực dụng khóa trước, không phải tiêu chuẩn phổ quát.
- CHAIRs không tăng so A; object recall không giảm quá.005 absolute. Báo chiều dài/câu trùng A, không tối ưu theo chúng.
- Y_F hơn cả A và C* ít nhất 4/32=.125; RC_F không thấp hơn, FR_F không cao hơn. Với uncertain, chưa vượt được bằng bounds thì không gọi pass.
- F thật sự đổi posterior/readout: báo mass null, entropy, KL giữa prior và posterior, endpoint accuracy, độ khác với \(\hat\alpha\hat\beta^\top\), thay đổi predicate khi zero interaction values. Không lấy metric attention làm thay thế caption correctness.
- Không có leakage hay lỗi triển khai làm đối chứng bất công.

Không qua thì dừng **ứng viên trong phạm vi pilot này**, không kết luận mọi joint grounding đều vô dụng. Qua thì chỉ GO sang xác nhận với seed/data độc lập và C_ind; chưa đủ kết luận bài báo. Dev và held đã dùng lại nên CI/panel chỉ hỗ trợ quyết định exploratory, không phải test xác nhận độc lập.

## 7. Shortcut và phản ví dụ phải giữ trong kết luận

| Nguy cơ | Kiểm cụ thể | Giới hạn suy luận |
|---|---|---|
| Class/geometry đủ phân biệt; visual interaction thừa | Inference bỏ union appearance hoặc geometry theo từng lượt; so với C* | Can thiệp off-manifold chỉ là sensitivity, không tự là chứng cứ nhân quả |
| Bối cảnh ảnh phân biệt I+/I− | Same-text pairing; báo breakdown hai strata và predicate-only likelihood | Cùng thực thể không làm hai ảnh thành counterfactual chỉ đổi quan hệ |
| Model bỏ reader hoặc toàn null | Zero reader values; caption/predicate changes, gradient và null mass | Có thay đổi vẫn có thể là nhiễu |
| Endpoints rõ làm bài toán hết tự do | Báo entropy priors, posterior correction và rank-one distance | VSR chọn thực thể tương đối duy nhất; không phải benchmark mạnh của ambiguous binding |
| Trộn các cặp sau expectation pooling | Chấm caption tự do, giữ cả lỗi gán vai/referent | Coherent attention không chứng minh coherent sentence |
| Tăng template score, caption né predicate | RC/FR/Y; caption dev, recall và length | Teacher-forced gain không được tính là free-caption gain |
| Học trên nhãn âm thiếu | Chỉ dùng assertion âm đã xác thực, còn lại unknown | Không suy từ caption không nhắc đến thành quan hệ sai |

Khóa cách phân loại **cơ hội sửa endpoint** ngay trong pilot, không mở audit mới hoặc chọn case F thắng. Trên toàn bộ tập đã khóa, báo: (a) hai endpoint có predicted-box coverage hay không; (b) số ứng viên cạnh tranh; (c) prior chọn sai hoặc mass tại GT-matched endpoint dưới .8; (d) có coverage và prior đã đúng, mass≥.8 ở cả hai vai. Ngưỡng .8 chỉ là phân loại exploratory, không phải định nghĩa truth. Dùng metadata GT ở bước đánh giá, không đưa vào forward. Báo cả sửa đúng và sửa sai của F/C* so với prior riêng, giữ mẫu số đầy đủ; panel 32 cặp vẫn chọn hash trước, không lấy theo nhóm F có lợi. Nếu nhóm có cơ hội quá nhỏ thì báo “uninformative” cho giả thuyết sửa endpoint.

**Điểm yếu lớn nhất của lựa chọn này:** VSR hiện hữu có ít mơ hồ referent. Pilot F–C* kiểm được ích lợi caption, nhưng không mặc nhiên kiểm được giả thuyết sửa endpoint. Nếu không có cơ hội sửa, kết luận là **không đủ thông tin về cơ chế ấy**; vẫn có thể dừng đầu tư vì dữ liệu/ngân sách hiện tại. Nếu có đủ cơ hội mà F không sửa tốt hơn C*, đó mới là bằng chứng chống lợi ích cơ chế trong phạm vi đã thử. Posterior gần rank-one và C* đã đủ thì không thêm solver để tạo claim.

## 8. Kết luận nghiên cứu được phép nói

Hiện tại: “Đề xuất một phép đọc chung, với hypothesis rằng tương tác nên được quyền sửa lựa chọn thực thể trước khi decoder phát từ.” Chưa được nói “đã giải quyết hallucination”, “quan hệ nhân quả” hoặc “phương pháp OT mới”.

Nếu F thắng C* lẫn A trên caption tự do và qua kiểm shortcut, có thể nhận claim lợi ích readout/regularization. Chỉ khi còn sửa endpoint đúng hơn đối chứng mới hỗ trợ câu chuyện “tương tác sửa lựa chọn thực thể”. Nếu chỉ loss pair giảm, heatmap đẹp hoặc CIDEr hơn A nhưng không hơn C*, chưa có bằng chứng cho delta dự kiến.

Đến 31/10 có thể đạt một quyết định rõ hoặc một kết quả âm có ích; chưa có căn cứ hứa đủ thí nghiệm chính thức hay công bố. Không triển khai và không launch từ tài liệu này.
