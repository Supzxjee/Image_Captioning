# Kiểm lại tính mới: chưa có phương pháp đủ rõ để triển khai

28/10/2026. Research decision sau yêu cầu tìm một đóng góp phương pháp bảo vệ được. Không đọc lại ZIP, không code, không neural local hoặc launch GPU.

**Quyết định: rút khuyến nghị lấy partial relation interchange làm phương pháp chính. Sau khi xét hai cơ chế dưới đây, tôi chưa có delta đủ thuyết phục để khuyến nghị triển khai một phương pháp mới. Không mở thêm train để tìm điểm dương rồi gọi đó là novelty.**

Đây là thay đổi khuyến nghị nghiên cứu vì có nearest prior cụ thể hơn, không phải kết quả thực nghiệm bác bỏ hai cơ chế. NO_GO của RelPos/RelContrast200 giữ nguyên.

## 1. Bằng chứng thực sự có

- Frame200 hoàn tất; RelPos40/64, Contrast41/64; difference1.5625pp,95%CI[-12.5,+15.625]pp. Chưa chứng minh linguistic transfer tốt hơn qua proxy đã chọn, không chứng minh representation vô dụng hoặc xác định nguyên nhân.
- Code cho thấy truth/pair losses không trực tiếp cập nhật GRU/initial/output; GRU học positive CE và chỉ nhận frame qua projection512→256. Đây là một hạn chế coupling có thật, nhưng direct decoder contrast đã có tiền lệ nên sửa nó chưa đủ thành phương pháp mới.
- A error atlas hiện có20focus cases và10controls. Theo README,20focus không phải20lỗi đã xác nhận:9clear errors,4unsupported claims,6under-specified potentially correct,1uncertain fine-detail. Các nhận xét là của assistant, không phải human gold hoặc mẫu đại diện. Atlas có caption_A, references và ảnh; không có original prompt/per-case CLIP-Q tensors.
- Chưa có phép đo cho thấy A thất bại chủ yếu vì đổi cặp referent giữa các từ. Tôi không dùng một giả thuyết về binding drift làm kết luận từ NO_GO.

## 2. Hai cơ chế đã xét, rồi loại khỏi khuyến nghị triển khai hiện tại

### A. Hoán chuyển phần tương tác và bảo toàn thực thể

Câu delta đề xuất: **So với IIT/MDAS, ta thay biến thuộc tính và đầu ra attribute bằng interaction value của cặp thị giác và likelihood cụm quan hệ trong caption decoder, nhằm ép sử dụng bằng chứng đúng đồng thời giữ phần thực thể.**

Novelty collision mạnh nhất: [RAVEL/MDAS, ACL2024, §3.6](https://aclanthology.org/2024.acl-long.470.pdf). L_Cause đẩy đầu ra của thuộc tính đích theo source sau interchange; L_Iso giữ các thuộc tính khác theo receiver; tối ưu kết hợp cả hai. Đây rất gần phần được mô tả là “đổi quyết định quan hệ nhưng giữ thực thể”. IIT và Strict IIT đã được đọc ở lượt trước và cũng bao phủ các nguyên lý hoán chuyển/selectivity.

Khác biệt còn lại của bản đề xuất là typed visual-pair interface, masking nhãn unknown và áp vào caption. Chưa có lập luận cho thấy chúng đòi một phép học hoặc ràng buộc mới, thay vì adaptation của Cause+Iso. Việc fine-tune decoder thay vì chỉ học alignment featurizer là khác biệt triển khai có thật, nhưng chưa tự thành novelty.

**Kết luận: không chọn làm đóng góp phương pháp chính.** Không nói rằng một pipeline giống hệt từng dòng chắc chắn đã được xuất bản; nói rằng core delta vừa viện dẫn chưa đủ.

### B. Một cặp bằng chứng cho cả cụm quan hệ

Câu delta đề xuất: **So với attention chọn bằng chứng từng từ, ta dùng một biến cặp có thứ tự dùng chung trong toàn đoạn quan hệ, nhằm ngăn ghép một câu từ nhiều cặp mà không cặp nào hỗ trợ cả sự kiện.**

Hai cách tính minh họa:

token-wise: product_t sum_k pi_tk p(y_t | prefix,k)

shared binding: sum_k pi_k product_t p(y_t | prefix,k)

Đây là thay đổi factorization có nội dung; không đồng nhất với việc làm attention heatmap ổn định. Tuy nhiên, core này cũng không mới:

- [Shen et al., ACL2020, §4, Eq.(2)–(4)](https://aclanthology.org/2020.acl-main.641.pdf) học jointly segmentation và correspondence: chọn một record cho mỗi segment, sinh toàn segment từ record đó, marginalize segmentation/alignment bằng dynamic programming. §5 dùng cả WebNLG với RDF(subject,relation,object) triples.
- [Human-like Controllable Image Captioning, CVPR2021, §3.1.3](https://arxiv.org/pdf/2103.12204) đã có role-shift caption generation theo grounded semantic-role sequence. [MTTSNet](https://arxiv.org/pdf/1903.05942) đã xử lý subject/object/union cho relational captions.

Đổi record thành visual pair hoặc thêm truth-negative loss chưa xác định được một delta riêng vượt các cơ chế đó. Hơn nữa, chưa đo được lỗi binding cụ thể của A để biện minh chi phí thêm latent-state decoder/controller.

**Kết luận: chưa chọn để train.** Không lấy việc search không thấy đúng tên phương pháp làm bằng chứng mới.

## 3. Chỉ một việc còn thiếu có decision value

**Một audit đóng trên toàn bộ30case của A error atlas đã có, để xác định đối tượng của lỗi, không chạy model và không thêm mẫu theo kết quả thuận lợi.**

Đầu vào đã có: cases20.json, controls10.json, caption_A, references và ảnh trong local review viewer. Không coi mọi focus case là lỗi. Không dùng caption của nhánh khác làm bằng chứng trạng thái bên trong của A.

Với mỗi caption_A, ghi sự kiện cụ thể và phân một loại, cùng một câu bằng chứng:

1. Hai thực thể/referents xác định được, predicate hoặc thuộc tính quan hệ sai.
2. Nội dung câu gán vai/thuộc tính cho sai referent; chỉ ghi khi ảnh và câu đủ xác định, không suy internal attention từ quan sát bề ngoài.
3. Không đủ thông tin để phân biệt hai loại trên, hoặc thông tin thị giác mơ hồ.
4. Caption chấp nhận được/thiếu chi tiết nhưng không có lỗi đã xác nhận.

Đây là một audit, không bốn thí nghiệm. Output là30rows và counts, giữ đầy đủ uncertain. Cần người chấm xác nhận các cases dùng để phát triển luận điểm; assistant notes không tự được nâng thành gold. Không cần feature extraction, CLIP forward, bộ dữ liệu mới hoặc huấn luyện GPU.

Decision:

- Nếu không xác nhận được lỗi referent-binding, bỏ premise “A đổi cặp giữa chừng”; không dựng persistent-pair decoder để giải một lỗi chưa thấy.
- Nếu lỗi nằm ở predicate dù referents đúng, không tuyên bố chunk locking hoặc entity preservation giải quyết đúng nút thắt. Việc có lỗi loại này vẫn chưa xác định encoder hay decoder gây ra nó.
- Nếu có lỗi binding rõ, lấy chính counterexamples ấy làm yêu cầu mà một delta sau này phải xử lý vượt nearest segment/role methods; **không tự suy đã có phương pháp mới hoặc tự mở train**.
- Nếu phần lớn case vẫn underdetermined, chưa đủ cơ sở chọn một cơ chế mới từ atlas này.

Audit được chọn để loại giả thuyết thiết kế sai, không để cứu gate200, ước lượng tần suất lỗi toàn tập hoặc thay thế đánh giá độc lập. Chưa đặt thêm một lịch GPU; quota hiện tại chưa được xác nhận.

## 4. Điều tôi đã kiểm và điều chưa biết

Đã đọc method §3.6 của RAVEL/MDAS; §4 và dataset §5 của Shen2020; role-shift §3.1.3 của Chen2021; các primary IIT/StrictIIT/MTTSNet ở lượt trước. Đã đọc README/schema của atlas để kiểm khả năng làm audit từ artifact sẵn có.

Show, Control and Tell đã tìm thấy nhưng PDF access gặp lỗi; không tự ghi là đã đọc full method. Không dựa vào review/survey để kết luận novelty.

Chưa có chứng minh toàn bộ pipeline interchange hay shared-pair captioner giống hệt một paper; cũng chưa có delta phương pháp đủ thuyết phục so với các cơ chế gần nhất vừa nêu. Hai điều này có thể đồng thời đúng.

**Khuyến nghị riêng: hiện không triển khai hai hướng trên với nhãn “phương pháp mới”. Hoàn tất đúng audit có sẵn nêu ở mục3 rồi mới chọn vấn đề cơ chế tiếp theo. Tôi không có cơ sở hứa một phương pháp mới hoặc positive paper trước31/10.**
