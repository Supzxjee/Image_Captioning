# Quyết định sau panel ngữ nghĩa I/J

29/10/2026. Research lead đọc [SEMANTIC_PANEL_RESULT](SEMANTIC_PANEL_RESULT.md)/json và quy tắc đã đặt tại [quyết định trước panel](ASTRA_RESULT_DECISION_2026-10-28.md). Không đọc lại ZIP/175 nhãn; không code mô hình, neural hoặc GPU.

**Chốt: đồng ý với root—giữ A và dừng cấu hình/recipe I/J hiện tại. Không chạy J_bypass, seed 2 hoặc full train từ kết quả này.** Quyết định dừng có cơ sở; claim “đã chứng minh J kém ngữ nghĩa nói chung” thì vượt bằng chứng.

## 1. Vì sao dừng dù panel chưa là human gold?

Hai nguồn bằng chứng có vai trò khác nhau:

- Dev 1.000: J−A CIDEr −.071306, 95% CI[-.108043,−.036360] trong run đã kiểm; CHAIRs 61/1000 so 52 của A, recall .45435 so .45804. Đây là bằng chứng chất lượng theo các metric này chưa cải tiến, với giới hạn một seed/reuseddev.
- Mixed panel: RC J 13/32 so A 15/32; FR 1/32 ở cả hai; definite coverage 28/64 ở cả hai; correct/wrong J 25/3 so A 26/2. Y_J−Y_A=−.0625, pairedCI[-.21875,+.0625]. Không thấy tín hiệu semantic benefit để đánh đổi phần CIDEr giảm.

Không cần chứng minh mọi giá trị ngữ nghĩa của J đều vô ích mới được dừng một cấu hình tốn tài nguyên. Với baseline mạnh hơn và không có benefit nhiệm vụ đã quan sát được, tiếp tục đòi một giả thuyết mới cụ thể. Bất định annotation không tự là lý do huấn luyện thêm.

Trong recipe chung, J−I CIDEr+.098336 có CI dương vẫn là kết quả tương đối có giá trị. Panel J−I Y=0, CI[-.125,+.125] chưa thêm bằng chứng về relation generation. Không xóa kết quả J−I và cũng không lấy nó làm bằng chứng thắng A.

## 2. Phản biện cách diễn giải panel

1. **17/175 unique human-confirmed là mẫu ưu tiên, không random.** Không dùng nó để suy ra accuracy/agreement của 158 nhãn AI-only hoặc “chất lượng human validation” của cả panel. Blinding trước khi mở arm key giúp giảm bias theo tên mô hình; chưa xác định độ tin cậy của nhãn.
2. **Annotation bounds không bao phủ toàn bộ annotation error.** J−A bounds[-.125,−.03125] chỉ xét các unknown đã gắn cờ và coi những nhãn definite còn lại đúng. Một definite label do AI chấm vẫn có thể sai. Không viết “J kém A bất kể mọi sai số nhãn”.
3. **Bootstrap có điều kiện trên mixed labels.** Nó không mô hình hóa bias/sai sót judge, seed uncertainty hay việc panel đã được dùng nhiều vòng. CI của Y đi qua0: chưa có kết luận population-level J<A về ngữ nghĩa. FR difference CI[0,0] cũng không chứng minh hai mô hình có cùng false-relation rate ngoài mẫu này.
4. **Bounds và bootstrap trả lời hai câu hỏi khác nhau.** Fixed-panel assignment cho unknown có thể luôn cho J−A âm trong khi resampling pairs vẫn chứa 0. Không có mâu thuẫn; cả hai đều chưa xử lý definite-label errors.
5. **Y/RC/FR không phải full-caption accuracy.** Coverage bằng nhau không làm mọi nội dung caption tương đương; Y thấp hơn có thể gồm paraphrase/target-identity ambiguity. Các cột correct/wrong/unknown riêng là cần thiết. Duplicate captions không tạo thêm independent test units.
6. Giữ #6X/#49X uncertain đúng theo xác nhận người dùng, bảo tồn raw records. Không yêu cầu người dùng giải quyết lại hai ca ấy.

Vì thế dùng cụm **“không quan sát được lợi ích ngữ nghĩa của J trong sàng lọc hỗn hợp này”**. Chưa yêu cầu chấm lại 158 ca: điều đó không cần để quyết định dừng hiện tại. Nếu sau này dùng panel làm kết quả human-evaluated trong bài thì cần nguồn human evaluation thực sự; còn hiện tại phải công bố đúng provenance mixed screening.

## 3. J_bypass và nguyên nhân thất bại

Điều kiện trước đây để xét J_bypass là có semantic benefit đáng theo đuổi, rồi mới phân tách cái giá của reader/decoder state. Điều kiện ấy chưa xuất hiện. Không chạy bypass chỉ để tìm một con số thuận lợi hoặc một lời giải thích cho kết quả âm.

Vẫn giữ giới hạn nhân quả đã nêu: off chỉ zero pair score/value, còn entity contexts và adapted decoder; chưa cô lập common objective, coadaptation hay residual mismatch. Dừng không có nghĩa đã xác định “decoder forgetting”, “giao diện mới gây hại” hoặc “CLIP không có quan hệ”.

Không chọn subgroup projective/containment để cứu overall: J có 8 positive q-mentions so A 7 trong nhóm ấy nhưng contact/proximity J 5 so A 8. Đây là mô tả hữu ích, không là validation của phương pháp chuyên biệt đã được chọn trước.

## 4. Nếu mở một thí nghiệm mới: chỉ một giả thuyết cụ thể

**Quyết định hiện hành vẫn DỪNG I/J; không tự launch phần dưới.** Nếu root/user tiếp tục tìm cải tiến, tôi chọn một phép thử ít thay đổi hơn, không chọn thêm reader/solver:

> Giám sát quan hệ qua decoder A gốc có thể giữ/chuyển giao caption tốt hơn khi giữ nguyên toàn bộ giao diện memory/readout, thay vì đồng thời học một đường đọc mới.

Đây là engineering hypothesis, chưa là nguyên nhân được xác nhận. Động cơ: nhiều nhánh thêm/điều biến thông tin đã không thắng A; I/J đổi cả reader lẫn cập nhật decoder/objective nên chưa biết native-A có hưởng lợi từ cùng supervision không. Thử trực tiếp việc còn thiếu đó tốt hơn đoán cách sửa gate, temperature hoặc đổi checkpoint.

**Phép thử nhỏ nhất: một arm N (native A), A fixed làm mốc.**

- Từ đúng checkpoint A gốc; không mang learned reader, không ROI/union mới, không thay prompt/memory, không nạp I/J weights.
- Học đúng hai decoder layer cuối, còn lại frozen như I/J. Dùng cùng 484 caption TRAIN images, 256 VSR TRAIN pairs, order/seed 42 và endpoint 500 updates.
- Giữ đúng recipe đã chạy \(L_{\rm SCST(A)}+.1KL(p_A\|p_N)+.2[-\ell_N^++softplus(\ell_N^--\ell_N^+)]\); relation loss qua native decoder thật. Không thêm objective/CE warmup và không đổi LR để cùng lúc “cứu” recipe.
- Export A/N free captions và paired metric deltas trên cùng dev 1.000; held likelihood chỉ diagnostic. Chỉ khi quality có tín hiệu đáng theo đuổi mới đánh giá N trên panel 32 đã khóa; không thêm cohort.
- Đây là một thí nghiệm mới sau kết quả âm, phải ghi như vậy; không gộp thành arm I/J đã preregistered.

**Giá trị quyết định của N:**

| Kết quả | Kết luận/việc tiếp |
|---|---|
| N cũng giảm CIDEr rõ so A, không có task benefit khác | Recipe thích nghi này có thể làm hại A ngay cả khi không có reader mới. Dừng recipe; không tiếp tục sửa cấu trúc để chữa một nguyên nhân chưa tách |
| N giữ/cải thiện quality và có relation benefit với coverage so A | Có lý do theo hướng native decoder objective; cần ablation positive-only vs thêm pair loss trước khi quy công riêng cho contrast |
| N giữ quality nhưng chỉ tăng likelihood/ranking | Chưa thành cải tiến caption; không dùng ranking để gọi thành công |
| Delta nhỏ bất định, semantics chưa rõ | Báo chưa đủ bằng chứng; không mở sweep để tìm seed/epoch dương |

Nếu N>I/J thì không tự suy ra pair features là nguyên nhân: model capacity/optimization trajectory cũng khác. Một arm này kiểm lợi ích thực dụng của native interface và tính đủ để gây suy giảm của recipe không-reader; không chứng minh từng loss có lỗi.

N chỉ là **ứng viên thử tiếp khi bắt đầu một thí nghiệm mới**, không phải khuyến nghị chạy ngay hoặc lời hứa cứu bài trước 31/10. Quota/venue chưa xác nhận. Nó không có thuật toán contrastive mới; giá trị phương pháp, nếu có, nằm ở cải tiến hệ caption được kiểm đúng và chuyển giao sang sinh tự do.

## 5. Điều hiện có thể viết

Một kết quả empirical có giới hạn: shared-weight readout giảm suy giảm so control có cùng pair attention/features dưới recipe đã thử; cả hai chưa cải tiến A, panel mixed không cho thấy semantic compensation. Kết quả này dùng được cho báo cáo thực nghiệm/limitations; chưa đủ cho claim “phương pháp cải tiến caption so baseline”.

Không cần mở vòng literature/audit mới để chốt điều đó. Lưu đầy đủ artifacts và giữ A làm mô hình hiện tại. Bước tiếp, nếu được chọn, phải là thử nghiệm N độc lập ở trên hoặc kết thúc nghiên cứu hướng này—không kéo dài I/J.
