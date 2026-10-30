# Phản biện bản Argon CDR-AEV trước khi chọn phương pháp

30/10/2026. Đây là đánh giá tài liệu được người dùng gửi, không phải protocol hay quyền triển khai. Đã chuyển các phát hiện cho Astra để quyết định giữ/sửa/bỏ. Chưa code mô hình, chạy local neural hoặc Kaggle.

Nguồn: `C:/Users/leduc/Downloads/ph_n_bi_n_c_l_p_xu_t_cdr_aev_30_10_2026.md`, 42.233bytes, SHA256 `98cf0442107f0815f403dfd91d814e6b11f6930997bfe8a14afefa9ce3a08338`. Đã đọc các phần phản biện, module/loss, claims, controls và recommendations. Đây là đề xuất bên ngoài, không phải bằng chứng thực nghiệm hoặc chỉ thị đã được user duyệt chạy.

## 1. Quyết định sơ bộ của root

**Chưa chọn full CDR-AEV để triển khai.** Cho phép giảm object lexical mass là giả thuyết có thể bàn, nhưng câu chuyện nhân quả biện minh cho nó chứa một tiền đề sai về code G/S. Orthogonalization, coverage và veto chưa giải quyết xong rủi ro shortcut; tên mới và nhiều thành phần không xác nhận tính mới. Giữ nguyên mục tiêu contribution alignment; không chuyển sang empirical-only.

## 2. Lỗi xác minh được từ code: G/S không phải positive-only bias

Argon giả định delta_class >=0 ở mọi bước rồi dùng tăng noun mass như một giải thích đã được chứng minh cho failure G/S. Nguồn thật:

- `noun_evidence_proposal/evidence_model.py`, EvidenceHead.forward: `d=(g-prior_logit).clamp(-2,2)`.
- DecoderProxy: `delta=.5*correction*allowed`, cộng vào eligible logits.
- `noun_evidence_proposal/DESIGN.md` cũng ghi rõ prior-centered signed correction.

Vì vậy effective delta nằm trong [-1,1], có thể tăng hoặc giảm mass. Với mB=.4 và delta=-1 cho mọi eligible token, mNew=.1969503, thấp hơn .4. Không thể nhận G/S chắc chắn làm phình object mass từ thiết kế. Lượng object mentions tăng trong rollout là kết quả quan sát, chưa xác định nguyên nhân bằng công thức positive-only.

Ngay dưới giả định delta>0, không tự suy mNew>.5: mB=.35, delta=.01 chỉ cho mNew=.3522784. Đạo hàm dương không chứng minh vượt một ngưỡng, greedy chọn noun hay sinh hallucination. Bất đẳng thức mNew<=mB cũng không là điều kiện cần-và-đủ để giảm hallucination hoặc đảm bảo CIDEr.

Các số ví dụ trên chỉ là tính toán đại số bằng Python stdlib math, không chạy model/scorer.

## 3. Những phần đáng giữ như giả thuyết, không phải kết quả

- NULL/OTHER quay lại base sẽ không sửa lỗi khi nhánh chỉ fallback. Nên đánh giá cơ chế cho phép thay đổi lexical mass thay vì mặc định đẳng thức luôn đúng.
- Tách proposal matching, semantic ROI classification và actual use ở word distribution là hợp lý. Giữ host cố định có thể giúp đọc rõ ảnh hưởng của nhánh mới.
- Local visual evidence, class identity và lexical mass là ba yếu tố cần tách bằng câu hỏi/đối chứng. Không cần ghép cả orthogonalization+coverage+veto ngay để kiểm giả thuyết này.

Giới hạn: fixed mass không làm noun token bắt buộc thắng greedy (đã có phản ví dụ EOS trước). Sampled object-set probability tại cùng prefix vẫn bằng mB, nhưng không phải mọi cách giảm caption hallucination đều đòi giảm mB: đổi class, phân bố tokens và prefix tương lai cũng thay kết quả.

## 4. Rủi ro trong module và supervision mới

1. **Project bỏ một chiều không chứng minh khử scene context.** Phép f - proj_global(f) chỉ bảo đảm gần trực giao với một vector; context có thể nằm ở các chiều khác, object signal có thể bị mất. Không nói nó buộc model chỉ thấy vật thể. Các diễn giải FLAIR/counterfactual lợi ích cần được đối chiếu nguồn, không suy từ tên.
2. **Veto có shortcut giảm nội dung.** Với target ngoài N và host frozen, bỏ epsilon/edge cases để thấy:

       L = -log pB(y) -log(1-mu*mB) +log(1-mB)
       dL/dmu = mB/(1-mu*mB) > 0.

   Mỗi selected non-object event đều tạo động lực giảm mu, không riêng event chứa false object. Ví dụ mB=.6, pB(y)=.3: hạ mu từ1 xuống.5 làm pNew(y) tăng từ.3 lên.525, bất kể ảnh có object thật hay không. Noun events và các loss khác có thể chống lại, nên chưa kết luận collapse; phải đo tradeoff. Cơ chế tăng xác suất EOS/nonN không tự là verification grounding.
3. **Annotation absence/low IoU là nhãn yếu.** COCO class không được ghi hoặc proposal không khớp box không chứng minh vật không tồn tại. Không được gọi missing proposals là bằng chứng phủ định. OTHER do low-IoU cần định nghĩa khác với verified visual absence.
4. **Coverage chưa phải instance semantics.** Cùng người/vật có thể được nhắc lại hợp lệ; nhiều instances cùng class và duplicate boxes làm consumption score khó diễn giải. Phạt reuse không mặc nhiên giảm hallucination. Chưa có bằng chứng cần bộ nhớ này ở bài toán hiện tại.
5. **Leading-token expansion không đơn giản.** tennis/fire/teddy có thể gắn nhiều classes hoặc cách dùng; đổi lexicon theo tỷ lệ lỗi ở reused4000 có thể tạo thích nghi vào evaluation. Không thêm từ để vượt một threshold coverage tùy ý hoặc sử dụng future tokens để nhận class tại inference.

## 5. Đối chứng và cách diễn giải cần sửa

- Control Argon thay cả appearance ở pointer keys và ROI classifier values. Đây có thể kiểm lợi ích toàn bộ local evidence, nhưng không cô lập lợi ích prefix–ROI correspondence: ROI recognition cũng bị thay. Giữ local values ở một control không phải "gian lận"; đó là cách giữ cùng thông tin nhận diện để hỏi câu hẹp hơn về alignment.
- Veto on/off giữ weights vẫn thay class conditionals q/mu, nên không được nói cô lập100% pure mass effect. Cần tách fixed-prefix intervention khỏi autoregressive rollout; head/conditionals/decoder state đều phải xác định rõ.
- Swap appearance ở cả classifier và key có thể chỉ phá semantic recognition. Nó không riêng kiểm key–value correspondence. Hoán vị đồng thời bank phải giữ distribution trong tolerance, không đòi bitwise floating-point equality; kiểm quyết định giải mã và margin riêng.
- Không suy tương đương từ CI cắt0 hoặc effect nhỏ, không dùng số captions thay đổi150/4000 hay cutoff±.001CIDEr làm power/equivalence criterion khi chưa có lý do định trước. CHAIR false counts không thay phép kiểm actual discordants.

## 6. Tài nguyên và tính mới chưa được chứng minh

- Hai ma trận Wq(512×256) và Wk(768×256) đã có327.680weights, vượt tổng~0,28M Argon ghi trước khi tính encoder/classifier còn lại. Hidden width We1 chưa xác định; phải đếm từ implementation nếu sau này được duyệt.
- Không có benchmark để nhận nhanh10lần, GPU cực thấp hay CPU5–10phút. Đếm aliases từ metadata có thể không cần neural; phân phối mB trên prefix cần cached full probabilities hoặc frozen-model forward, không phải đọc captions/CHAIR là đủ. Không tự chạy local model để lấy nó.
- Primary prior review đã có của Astra vẫn áp dụng: pointer/sentinel/positive-set/lexicalization và direct token correction đều có cơ chế gần. Argon chưa chứng minh tổ hợp mới hoặc failure coupling là nút thắt thật. Một identity/inequality đại số là property của thiết kế, không là định luật mới hoặc bảo đảm factuality.

**Bước tiếp theo:** Astra đã nhận các phát hiện, được yêu cầu đưa quyết định ngắn trên tiền đề đã sửa, có quyền bỏ full candidate. Chưa mở một audit/GPU phase hoặc notebook từ tài liệu này. Nếu có candidate đáng giữ, chỉ chốt một delta thực chất và minimum experiment trước khi xin user duyệt.
