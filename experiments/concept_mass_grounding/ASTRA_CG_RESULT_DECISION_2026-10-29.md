# Quyết định sau C/G v2: giữ gain C, dừng G, một ablation nếu tiếp tục

29/10/2026. Research lead đọc [RESULT](RESULT.md) và46logged rank-update rows của C từ cache text hiện có; không đọc ZIP, không neural/GPU. Source/result provenance theo root: core33dbef3, same initial/frozen/input identities,500updates và4000unique TRAIN exposures.

**Chốt: bảo tồn C như ứng viên cải thiện CIDEr; dừng mở rộng G hiện tại. Nếu còn tài nguyên cho đúng một run, chọn C0=no-positive-preservation, không chọn tăng UL hoặc seed2 của G.** Chưa có phương pháp giảm hallucination được xác nhận.

## 1. Kết quả được phép nhận

| Model | CIDEr | CHAIRs /1000 | Hallucinated mentions | Recall |
|---|---:|---:|---:|---:|
| A | 1.205513 | 52 | 53/1632 | 1490/3253 |
| C | 1.225678 | 51 | 52/1632 | 1492/3253 |
| G | 1.229535 | 52 | 53/1633 | 1489/3253 |

- C−A CIDEr+.020166,95%pairedCI[+.002931,+.038610]; G−A+.024023,[+.006684,+.042807]. Đây là gain trên dev/run đã thử, không xóa nó vì G chưa thành công.
- G−C+.003857,CI[−.003236,+.011555]: chưa thiết lập lợi ích riêng về CIDEr của UL.
- G−A CHAIRs=0,CI[−.008,+.008]. C−A=−.001,CI[−.009,+.007]: chênh52→51 không chứng minh hallucination giảm, cũng chưa là equivalence/noninferiority.
- G CHAIRi hơi giảm so A vì denominator1632→1633; numerator vẫn53. Không gọi là giảm hallucinations.
- Length C9.987/G9.968 so A10.213, recall gần A. Chưa thấy collapse lớn theo các point estimates, nhưng không coi đó là chứng minh giữ nguyên mọi nội dung.

**Mục tiêu user “tăng CIDEr và giảm hallucination so A” chưa đạt cả hai trục.** C có kết quả chất lượng đáng giữ; không nên trình bày toàn pilot là thất bại hoàn toàn hoặc chọn G chỉ vì CIDEr mean cao nhất.

## 2. Cơ chế: những gì đã và chưa tách được

C gồm SCST continuation, rollout-prefix KL và positive concept-mass preservation. C−A chưa quy riêng gain cho L_keep; broadened caption exposure/tiếp tục huấn luyện và KL cũng có thể giải thích.

Neutral-prefix70-class diagnostic:
- Supported mass A.10158658→C.10670764/G.10642853.
- Weak-negative mass A.0001295722→C.0001310964/G.0001313372.
- Separation tăng chủ yếu từ supported mass; G không có lợi thế điểm so C.

Đây phù hợp với hướng positive preservation hơn negative suppression nhưng **chưa chứng minh nguyên nhân**. Neutral prefix khác các on-policy prefixes của UL; không dùng nó để phủ định mọi tác dụng cục bộ của UL. Caption hallucination endpoint mới là đích cuối, và nó chưa giảm.

G có211weak-negative events nên không phải zero-event run. So211negative với4447positive anchors không đủ suy ra liều UL quá yếu: reduction, probabilities và gradients khác nhau. Chưa có lý do tăng coefficient hoặc steps.

Kiểm cache text C:46logged rank-update rows (hai ranks, step1–500),18rows có keep>0; mean keep .00241315, max .02211022. Đây là logged snapshots, không phải18/500active updates và không là đo norm gradient. Nó cho thấy loss không hiển nhiên bằng0 ở mọi snapshot, nên ablation không dư thừa; giá trị loss nhỏ cũng chưa dự đoán effect size.

## 3. Một run đáng làm nhất: C0

**Câu hỏi:** positive preservation có đóng góp ngoài continuation+rolloutKL không? Đây là phần còn thiếu để nâng C từ recipe có gain thành ứng viên đóng góp căn chỉnh có bằng chứng.

Từ A gốc, cùng source/training path C:

\[
L_{C0}=L_{\rm SCST,\ reward\ gốc\ A}+.1KL_{\rm rollout}(p_A\|p_{C0}).
\]

C giữ thêm .1L_keep. UL=0 trong cả C và C0. Chỉ thay coefficient keep từ.1→0; giữ positive-event computations để monitor nếu thuận tiện. Không thêm architecture hoặc đổi sampler.

- Giữ seed42, chính xác4000TRAIN IDs/order/ref schedule đã dùng, globalbatch8,500updates, native A và frozen scope/optimizer/LR/clip/decoding như C.
- Không khởi tạo từ C endpoint; cả C/C0 khởi tạo A. Không replay captions sample của C như targets vì đây là on-policy comparison.
- C0 và C ban đầu cùng rule/RNG; rollout captions sau train không bắt buộc giống do policy khác. Đây là hệ quả objective, không là data-order mismatch.
- Pin configuration/source delta và runtime receipts. Run này được chọn sau xem C/G, phải ghi là post-pilot ablation, không nhận đã preregistered từ đầu.
- Chỉ endpoint500; export cùng dev1000, full CHAIR80/CIDEr/recall/length, conditioning diagnostic và per-image values. Paired C−C0 và C0−A với cùng evaluator/CI. Không cần panel human mới trước run này.

C0 đã là câu trả lời trực tiếp về native common recipe; không cần chạy N-VSR hay thêm Tune arm riêng. Source hiện có có thể reuse, không cần một vòng literature/audit.

## 4. Quyết định sau C0

1. **C hơn C0 có bằng chứng ở CIDEr, hoặc bảo vệ CHAIR/recall rõ trong khi giữ gain CIDEr:** có căn cứ theo positive-preservation objective. Nói rõ trục nào được hỗ trợ và uncertainty; chưa nhận dual-goal success so A chỉ từ việc thắng C0.
2. **C≈C0 hoặc difference bất định:** giữ gain recipe nhưng contribution riêng của L_keep chưa được thiết lập. Không tăng keep/UL coefficients để cứu claim trước deadline.
3. **C0 tốt hơn C:** positive preservation hiện tại không có lợi ích đã chứng minh; C/G chưa xác lập contribution mới. C0 có thể là continuation baseline tốt hơn, không tự là phương pháp grounding mới.
4. **Nếu không còn ngân sách/time:** kết thúc ở C/G, giữ C checkpoint và báo gain CIDEr exploratory; không tự nâng claim C thành grounding contribution.

Không yêu cầu C thắng C0 ở mọi metric. Nhưng một lợi ích không được mua bằng suy giảm lớn ở trục còn lại. Đối với mục tiêu gốc, hallucination reduction so A vẫn phải có bằng chứng riêng;51vs52không đủ.

Tôi ưu tiên C0 hơn chỉ repeat C seed2 vì bottleneck hiện tại là **attribution của đóng góp**, không chỉ tái hiện một scalar gain. Sau khi có effect riêng đáng theo đuổi, seed/holdout confirmation mới có ý nghĩa đầu tư cao hơn. Một run C0 không thay thế xác nhận robustness và không được extrapolate image-bootstrap thành seed confidence.

## 5. Cách định vị bài nếu positive preservation được hỗ trợ

“Bảo toàn khối xác suất các khái niệm có support khi tiếp tục tối ưu caption” là một objective adaptation kế thừa, không là UL hoặc thuật toán nền tảng mới. C được chọn thành ứng viên sau khi nó là control có gain; phải báo cả G và lý do chuyển trọng tâm.

Hiện chỉ được viết: **continuation recipe C tăng CIDEr trong một seed/reuseddev mà các point estimates hallucination/recall gần A; negative correction G chưa cho ích lợi riêng hoặc giảm hallucination**. Không được viết positive grounding đã gây ra gain, hay phương pháp đạt đồng thời hai mục tiêu.

Nếu positive preservation có effect, một bài phạm vi vừa phải có thể trình bày quality–factuality tradeoff và ablation minh bạch. User cần biết nếu scope chuyển từ “giảm hallucination” sang “tăng CIDEr với factuality được giữ theo đánh giá hiện có”; không tự coi hai mục tiêu ấy như nhau. Venue/quota/accepted31/10 chưa xác minh.

**Quyết định cuối: giữ C; dừng G; tối đa một C0 ablation nếu tiếp tục đầu tư. Không mở sweep hay đổi một họ phương pháp nữa trước câu trả lời này.**
