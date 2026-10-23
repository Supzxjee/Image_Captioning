# Astra Ultra — đề xuất sau G/S

Bản ghi nguyên văn phản hồi nghiên cứu, không phải chỉ dẫn tự động triển khai.
Thread: `01a15544-6a7a-7903-a6e1-23e7b9f6caa6`; turn: `01a16976-ffbc-71b1-8c57-5eaf80c87b96`.
Trạng thái: proposal only; chưa chạy cache/model/pilot mới.

**Tôi chọn kiểm tra một thay đổi về nguồn thông tin: giữ nguyên đường toàn ảnh của A, bổ sung các vùng ảnh được mã hóa từ ảnh gốc trước khi chi tiết bị mất ở bước thu nhỏ 224×224.**

Thí nghiệm quyết định sẽ dùng **một adapter chung**, được huấn luyện với cả hai loại đầu vào: vùng lấy từ ảnh gốc và vùng lấy từ ảnh đã thu nhỏ. Sau đó so caption dưới **cùng trọng số**. Phiên bản dùng chi tiết gốc phải thắng cả A và phiên bản giảm chi tiết mới đáng đầu tư tiếp.

Tôi giữ kết luận dừng G/S. Nhưng khuyến nghị trước đó của tôi về dừng rộng hơn là quá mạnh: các kết quả âm chưa phân biệt được **khó khai thác thông tin đang có** với **thiếu thông tin sau bước mã hóa đầu vào**. Đây là khoảng trống đáng kiểm tra, không phải chẩn đoán rằng CLIP hay độ phân giải đã được chứng minh là thủ phạm.

**1. Ba giả định cần xem lại**

| Giả định đang hạn chế lựa chọn | Bằng chứng khiến phải nghi ngờ | Phản ví dụ và điều chưa biết |
|---|---|---|
| **Cache 224×224 đã đủ bằng chứng; chỉ cần căn chỉnh lại.** | Nhiều nhánh tiếp tục biến đổi hoặc định tuyến trên cùng đặc trưng contextualized. Chúng không thể khôi phục chi tiết đã bị loại bỏ lúc resize. | UPT đã bổ sung thông tin khác mà chưa có gain; paircrop cũng chưa thành công. Chưa biết lỗi quan trọng của A có thực sự cần chi tiết cao hơn hay chủ yếu là chọn nội dung/ngôn ngữ. |
| **Biết đối tượng hiện diện sẽ giúp biết nên nhắc gì.** | G/S tăng recall và số đối tượng được nhắc nhưng giảm CIDEr, tăng hallucination. | Thưởng đối tượng trong quá trình huấn luyện trước đã giúp dự án tăng điểm. Vì vậy không thể kết luận supervision đối tượng vô ích; vấn đề có thể nằm ở độ chính xác, độ nổi bật hoặc cách can thiệp. |
| **Một ngân sách ngắn tương tự nhau đủ đánh giá mọi module.** | Các thử nghiệm từng thêm từ head nhỏ đến core 23M tham số nhưng nhiều lần chỉ có 1.250–2.500 update. | G/S nhỏ và đã tạo tác động xấu rõ ràng; không thể cứ viện “chưa hội tụ” để kéo dài. Cần phân biệt dừng một cấu hình theo ngân sách với bác bỏ cả họ phương pháp. |

Đồng thời, A **đã có prompt và cơ chế kết hợp thị giác–ngôn ngữ**, cũng đã hưởng lợi từ huấn luyện SCST+CE+thưởng đối tượng. Bài toán hiện tại là tạo lợi ích bổ sung trên nền đó.

**2. Hai hướng khác nhau thực chất**

| Hướng | Tín hiệu bổ sung | Vì sao có thể thành caption gain | Rủi ro chính |
|---|---|---|---|
| **Bổ sung chi tiết thị giác, giữ ngữ cảnh toàn ảnh — tôi chọn** | Các chi tiết ảnh gốc mà đường resize toàn ảnh có thể làm mất; không thêm nhãn ngữ nghĩa dự đoán | Giúp các memory slot decoder đang dùng phân biệt thuộc tính, vật nhỏ hoặc tương tác nhìn chưa rõ; tác động được lên toàn vocabulary | Chi tiết mới có thể không liên quan đến caption; crop cắt vật; adapter hoặc encoder không biết sử dụng nó |
| **Đối chiếu lựa chọn từ trong cùng prefix tự nhiên** | Hai ảnh có cùng prefix TRAIN nhưng từ tiếp theo khác nhau, có annotation hỗ trợ và loại trừ từ đối thủ | Học quyết định “ở vị trí này nên nói gì”, thay cho hiện diện đối tượng toàn ảnh | Có thể giải bằng legacy prompt, nhãn âm thiếu chắc chắn, exact-prefix pairs ít; lại cải thiện teacher-forced discrimination mà không cải thiện sinh tự do |

Hướng thứ hai có tiền lệ về học caption bằng đối chiếu; tôi chỉ kiểm tra **abstract** của *Contrastive Learning for Image Captioning*, không coi đó là bằng chứng rằng thiết kế paired-prefix ở đây sẽ hiệu quả. [Dai & Lin, 2017](https://arxiv.org/abs/1710.02534)

Tôi không chọn nó lúc này vì để tách ảnh khỏi shortcut qua prompt cần thêm các điều kiện ảnh–prompt không khớp. Khi đó bài toán có thể chuyển thành học chống prompt sai, chưa chắc là nhu cầu của inference thực tế.

Với hướng được chọn, *HAAV* cung cấp tiền lệ về kết hợp nhiều nguồn biểu diễn cho caption. Tôi đã đọc phần phương pháp trong **full text**: nghiên cứu đó dùng các view dị thể và decoder tổng hợp phân cấp. Thiết kế bên dưới không phải bản tái hiện HAAV và không được nhận tính mới chỉ vì kết hợp nhiều view. [Kuo & Kira, CVPR 2023](https://arxiv.org/pdf/2305.16295)

**3. Thiết kế được chọn: cùng adapter, khác lượng chi tiết đầu vào**

Với mỗi ảnh, giữ nguyên cache toàn ảnh và legacy prompt của A. Tạo bốn vùng cố định là bốn phần tư ảnh, theo cùng tọa độ chuẩn hóa:

- **H — chi tiết gốc:** lấy mỗi vùng từ ảnh gốc, resize vùng về 224×224, chạy đúng CLIP hiện tại.
- **L — đối chứng giảm chi tiết:** resize toàn ảnh về 224×224 trước, rồi lấy bốn vùng tương ứng, phóng mỗi vùng về 224×224 và chạy cùng CLIP.

H/L có cùng vị trí vùng, số lượt encoder, preprocessing ngoài thứ tự lấy vùng/thu nhỏ, số token và tham số. Không chọn vùng bằng bbox, reference caption hoặc lỗi của A.

Đây là khác biệt then chốt với các lần trước:

- Mask patch trên cache không tạo thêm thông tin ảnh.
- Paircrop từng thay ngữ cảnh bằng một vùng được chọn; ở đây **đường toàn ảnh của A vẫn nguyên vẹn**.
- UPT đưa biểu diễn chuyên biệt từ mô hình khác vào; ở đây dùng cùng CLIP và projection của A để giảm khác biệt giữa hai loại đặc trưng.
- Không thêm điểm hiện diện danh từ giống nhau ở mọi vị trí như G/S.

Để hạn chế cache và attention:

```text
Mỗi vùng:
CLIP patch map 14×14×768
→ average pool không gian 2×2
→ 7×7×768
→ projection 768→512 và ReLU có sẵn của A, frozen

Bốn vùng:
4×49×512 = 196×512
```

**Pool trước projection/ReLU**, không đảo thứ tự. Đây là mã hóa chi tiết cao hơn **trước khi nén**, không phải tăng độ phân giải của lưới token cuối.

Gắn hai tọa độ tâm vùng con trong hệ tọa độ toàn ảnh vào mỗi token. Không chỉ dựa vào thứ tự nối token: attention không tự biết token thuộc góc nào.

Đường sử dụng bằng chứng:

\[
Q_A\in\mathbb R^{32\times512},\qquad T\in\mathbb R^{196\times512}.
\]

\[
q=\mathrm{LN}(Q_A)W_q,\quad
k=[\mathrm{LN}(T);x;y]W_k,\quad
v=[\mathrm{LN}(T);x;y]W_v,
\]

\[
Q'=Q_A+\mathrm{MHA}_{4\ heads}(q,k,v)W_o.
\]

Kích thước attention là 128:

- \(W_q:512\to128\)
- \(W_k,W_v:514\to128\)
- \(W_o:128\to512\)

Không bias, LayerNorm không affine: **262.656 tham số trainable**. Khởi tạo \(W_o=0\) để mô hình ban đầu khôi phục A.

Sau đó:

\[
M'=[L(Q',P_A);Q']\rightarrow \text{decoder A}.
\]

Giữ nội dung prompt \(P_A\) ban đầu, nhưng **tính lại \(L\)** từ \(Q'\). Toàn bộ tham số A frozen; gradient vẫn phải đi qua legacy branch và decoder đến adapter. Không đặt toàn bộ đoạn này trong `no_grad`.

Nhánh mới không biết prefix trực tiếp, nhưng nó thay đổi memory mà decoder truy cập tùy prefix. Vì vậy tác động không bị giới hạn vào một danh sách danh từ hay một độ lệch logit cố định.

**4. Supervision, loss và mức căn chỉnh**

Pilot này chỉ huấn luyện adapter bằng **full-caption CE**:

\[
L_{\mathrm{CE}}=
-\frac{
\sum_{i,k,t}m_{ikt}
\log p_\theta(y_{ikt}\mid y_{ik,<t},Q'_i,P_{A,i})
}{
\sum_{i,k,t}m_{ikt}
}.
\]

- Dùng đủ năm reference TRAIN.
- \(m=1\) cho token đích hợp lệ, gồm EOS; không tính PAD hoặc BOS như token đích.
- Chuẩn hóa theo tổng token hợp lệ của global batch, không trung bình các local mean có mẫu số khác nhau.
- Không tạo caption âm, không gán quan hệ phủ định, không có loss bbox/presence/attention.
- Reference chỉ là target và prefix teacher forcing chuẩn; không được đưa vào visual adapter hay legacy prompt.

Tôi chọn CE-only cho **phép thử tiếp cận thông tin này** để tiết kiệm rollout và không thêm một mục tiêu khuyến khích nhắc đối tượng. A vẫn giữ toàn bộ kiến thức đã học bằng SCST; H/L dùng cùng objective nên so sánh chúng không bị lẫn với khác biệt reward. Đây không phải kết luận SCST không còn cần thiết.

Mức căn chỉnh là **ngữ nghĩa của caption với bằng chứng thị giác toàn cục và cục bộ**, được học qua đường sinh thực tế. Không nhận 32 query là 32 đối tượng; không coi một tile là một thực thể. Quan hệ có thể cần nhiều tile cùng ngữ cảnh toàn ảnh. Kiến trúc cho phép sử dụng chúng, chưa bảo đảm đã học được quan hệ.

Độ nổi bật của nội dung được giám sát bằng các caption hoàn chỉnh, thay vì yêu cầu nói mọi thứ xuất hiện trong ảnh. Caption COCO vẫn có thể bỏ qua chi tiết hữu ích; đó là giới hạn thật của nguồn supervision này.

**5. Thí nghiệm nhỏ nhất tôi cho là đủ giá trị quyết định**

Tôi chọn **một lần train chung**, không hai mô hình được tối ưu độc lập:

- Exact A endpoint đã xác minh.
- Train10k hiện có; **4 epoch = 5.000 update**, global batch 8.
- Mỗi ảnh được dùng H hai lần và L hai lần, theo lịch khóa trước, độc lập với caption. Cân bằng điều kiện trong từng epoch.
- Cùng adapter xử lý cả H và L; không cung cấp cờ “đây là H/L”.
- AdamW \(10^{-4}\), weight decay 0,01, clip norm 1; seed 42.
- Chỉ adapter được cập nhật. Không chọn endpoint bằng val.
- Đánh giá raw greedy trên **reused val5000**, cùng tokenizer, IDF, giới hạn độ dài và evaluator hiện có.

Ba đầu ra cần so:

1. A.
2. Adapter chung với L.
3. Cùng adapter đó với H.

Đối chứng này khớp cả tham số lẫn lịch sử tối ưu. Hai điều kiện H/L đều đã xuất hiện trong train, nên không phải chỉ đưa nhiễu lạ vào mô hình rồi đo độ nhạy.

**Gate đầu tư tiếp, khóa trước:**

- H−A ≥ **0,01 CIDEr**, cận dưới paired bootstrap CI > 0.
- H−L ≥ **0,005 CIDEr**, cận dưới CI > 0.
- CHAIRs/i không xấu hơn A; object recall không giảm quá **0,005**.
- Báo thêm độ dài và số đối tượng được nhắc; không chọn riêng nhóm ảnh có lợi để quyết định.

Ngưỡng +0,005 cho H−L là ngưỡng thực dụng đề xuất để đòi hỏi lợi ích đủ rõ từ nguồn chi tiết mới, không phải một hằng số có cơ sở thống kê phổ quát. Các CI trên val đã tái sử dụng chỉ phục vụ sàng lọc; chúng không khôi phục tính độc lập của benchmark.

| Kết quả | Quyết định |
|---|---|
| H thắng cả A và L, qua safeguards | Có cơ sở đầu tư hướng khai thác chi tiết ảnh; bước tiếp theo là xác nhận độ ổn định, chưa tuyên bố phương pháp mới |
| H thắng L nhưng không thắng A | Có tín hiệu chi tiết mới được sử dụng, nhưng chưa có captioner tốt hơn; không tự động scale |
| H và L đều thắng A nhưng không khác đủ | Hỗ trợ lợi ích của đường xử lý nhiều vùng; chưa hỗ trợ giả thuyết chi tiết gốc là phần tạo gain |
| Không có gain hoặc tăng hallucination | Dừng cấu hình này trong dự án; không mở quét tile/gate/hệ số trên val vừa xem |

**Vì sao không chỉ 1.250 update?** Adapter bắt đầu từ nhánh đầu ra bằng 0 và phải học một giao diện mới. Bốn epoch với khoảng 263 nghìn tham số là ngân sách sàng lọc hợp lý hơn một lượt train10k, đồng thời CE-only rẻ hơn SCST. Tuy vậy, 5.000 update cũng không chứng minh đã tìm được nghiệm tốt nhất. Kết quả âm chỉ bác bỏ **giá trị đầu tư của cấu hình và ngân sách này**, không bác bỏ mọi phương pháp độ phân giải cao.

**6. Tính khả thi và điều khiến tôi bỏ hướng**

Không cần annotation mới hoặc checkpoint encoder mới. Cần ảnh gốc, đúng CLIP hiện tại, A và các cache đang có.

Với train10k + val5000:

- Hai điều kiện × bốn vùng × 15.000 ảnh = **120.000 lượt encode vùng**.
- Cache FP16 sau pooling và frozen projection khoảng **3,01 GB mỗi điều kiện**, khoảng **6,02 GB tổng**, chưa tính metadata/cache cũ.
- Inference H cần một lượt toàn ảnh và bốn lượt vùng nếu chưa cache; decoder vẫn dùng số memory slot như trước.
- T4×2 là cấu hình mục tiêu. Chưa có benchmark thời gian nên tôi không cam kết số giờ.

Rủi ro lớn nhất là **chi tiết cần thiết không nằm trong những gì CLIP này phân biệt tốt**, hoặc đã bị mất tiếp khi pooling. Ngoài ra, huấn luyện trộn H/L có thể khiến adapter chỉ dùng phần thông tin chung của chúng. Những khả năng đó giới hạn kết luận âm.

Tôi sẽ bỏ hướng trong phạm vi dự án nếu không đủ quota cho **cả hai cache và một lần train đủ ngân sách đã khóa**. Không thay đối chứng bằng vùng ngẫu nhiên, không giảm xuống một thử nghiệm quá ngắn rồi gọi đó là bác bỏ giả thuyết.

**Phần đóng góp có thể bảo vệ được**

Adapter cross-attention, zero initialization và kết hợp nhiều view đều là thành phần kế thừa. Thử nghiệm H/L cũng không tự trở thành đóng góp mới chỉ vì được thiết kế cẩn thận.

Nếu có gain, điều có thể bảo vệ trước hết là: **một cách đưa bằng chứng cục bộ vào các query mà captioner mạnh đang sử dụng, kèm bằng chứng thực nghiệm phân biệt lợi ích của chi tiết mới với lợi ích của thêm nhánh xử lý**. Tính mới vẫn cần đối chiếu tiếp.

Nếu không có gain, dự án vẫn giữ kết quả cải thiện Original nhờ recipe đã có; phần căn chỉnh bổ sung được trình bày như nghiên cứu thực nghiệm có kết quả âm và giới hạn rõ. Không cần ép luận văn phải đồng thời có prompt đối tượng–quan hệ, module căn chỉnh mới và reranker mới.

**Khuyến nghị của tôi là dành lượt thử tiếp theo cho phép kiểm tra nguồn thông tin này, thay vì sửa tiếp G/S hoặc thêm một matcher trên cùng cache.**

