# Brainstorm: correspondence vùng–từ với bảo toàn lexical mass

30/10/2026. User giữ yêu cầu contribution căn chỉnh, không chọn đổi scope sang empirical-only. **Khuyến nghị một hướng: learned region→lexical pointer ở phân phối từ, giữ native captioner frozen và bảo toàn mass của tập từ đối tượng.** Chưa duyệt code/budget/GPU; đây là phương án để root phản biện/prior-check và user chốt.

## 1. Giả thuyết và delta

Một nguyên nhân có thể của các pilot trước: routing đúng vị trí chưa tạo đúng semantic value, còn đưa vector mới vào hidden memory có thể không khớp cách decoder đã học. Đây chưa phải nguyên nhân được xác định.

Ta học correspondence **prefix sắp sinh object word ↔ tập ROI phù hợp ↔ phân phối lexical class**. Cùng pointer ấy trực tiếp tạo probability sinh từ. Không có bước “head phụ tốt rồi hy vọng decoder dùng”; không đưa thêm vector512 vào memory.

Delta task-specific đề nghị: bằng chứng ROI chỉ đổi **chọn đối tượng nào trong object lexicon**, không trực tiếp lấy mass từ EOS/từ chức năng hoặc giảm toàn bộ object-lexicon mass để né hallucination. Pointer/copy/classification đều có prior; constraint/dataflow cụ thể và utility của alignment mới là ứng viên đóng góp. Không nhận novelty từ một đẳng thức chuẩn hóa.

Đây là object-level region–word grounding, chưa giải quyết predicate, attribute hoặc đúng instance trong nhiều vật cùng class. Không đổi tên keep/C0 thành alignment đã thành công.

## 2. Biểu diễn và supervision thực có

Host B là **C0 frozen**, SHA5d92cd5… đã có mount/weights; A vẫn là reference. C0 là baseline mạnh hơn về CIDEr, không là phần novel. Mọi parameter encoder/Q-Former/prompt/decoder/vocab của B giữ frozen; dùng nguyên original caches/memory.

TRAIN: cùng4000images đã chọn từ TRAIN10k và nguồn71950COCO annotations,5refs/noun spans đã có. Predicted boxes/confidences từ cache; không dùng GT boxes để extract inference features. GT chỉ match IoU và tạo losses.

Region source khả dụng là pinned detector cache498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c: shared_pair_readout/readout_data.py đã đọc bbox/conf và lấy top12; cùng nguồn đã dùng cho I/J. Không cần biết extractor completeness vì không dùng absence làm nhãn âm. Root vẫn kiểm coverage của4000IDs cụ thể khi stage; empty/missing bank giữ ảnh và null-only fallback, không dùng GT thay. Matching lấy class từ GT theo IoU, không đòi predicted class phải đúng và không truyền label detector vào model.

Cho K≤12valid proposals:
- ROI fractional mean của196projected CLIP patches→trainable e_i∈R128. Không chạy crop/CLIP mới.
- ρ_i(c)=softmax regional classifier từ **appearance e_i**, trên các lexical classes biểu diễn được cộng OTHER. Không đưa predicted class ID/name vào classifier/key/value. GT-matched ROI (IoU≥.5) dùng class CE; class ngoài lexicon→OTHER. Unmatched/ambiguous ROI không tự gán background.
- h_t là pre-output hidden của B dưới ảnh+prefix thật, không có target word làm input.
- Query W_h h_t; local key W_k[e_i;box_i;conf_i]. Softmax trên ROI và null tạo π_ti,π_t0. Không có topK/box filtering theo target class lúc forward.

Nếu reference noun class c có một/nhiều predicted boxes khớp GT class c, positive target là **cả set** M_c, không bịa instance alignment. Nếu không có matching ROI, bỏ pointer loss của event ấy, báo coverage; không thay bằng GT ROI.

Trích chính xác các position dự đoán từ tokenizer A. Alias/class labels chỉ ở supervision hoặc lexical map, không đưa reference text tương lai vào h_t.

## 3. Output interface có ràng buộc

Khóa lexicon N gồm các **single-token object aliases** có trong vocab B, chia thành các tập N_c không chồng lấn. Bỏ ambiguous surfaces theo rule cố định trước run; multiword chưa mở ở pilot đầu. Root đếm số classes/tokens thực từ existing aliases, không giả định đủ80/72.

N là một tập lexical tokens, không phải bộ nhận biết POS hoàn hảo. Không gọi m_B là probability “sắp nói danh từ” chính xác về ngôn ngữ học.

Với p_B tại cùng prefix:
\[
m_B=\sum_{w\in N}p_B(w),\quad
b_c=\sum_{w\in N_c}p_B(w)/m_B,\quad
u_B(w|c)=p_B(w)/\sum_{v\in N_c}p_B(v).
\]

ρ_i(OTHER) quay về b thay vì ép một class đại diện:
\[
q_c=\pi_0b_c+\sum_i\pi_i[\rho_i(c)+\rho_i(OTHER)b_c].
\]

Distribution thật để sample/greedy và tính loss:
\[
p_{\rm new}(w)=
\begin{cases}
p_B(w), & w\notin N,\\
m_B q_c u_B(w|c), & w\in N_c.
\end{cases}
\]

Tính trong log-space để không chia mass underflow. OTHER/null đều fallback, và sum_c q_c=1. Vì vậy probability ngoài N và tổng mass trong N được giữ **đúng tại cùng prefix**. Relative synonyms/number/gender trong cùng class vẫn do B quyết định, không học từ coarse class label.

Đây là bảo toàn output probabilities có giới hạn, không bảo đảm caption length/recall toàn chuỗi: đổi một noun làm prefix tương lai khác. Cũng không bảo đảm sửa hallucination ngoài N.

Init null ưu thế (thí dụ bias8) giúp khởi đầu gần B; finite softmax không bằng B tuyệt đối. Test chế độ null-only phải phục hồi đúng B, còn init trainable phải ghi initial prediction differences. Không dùng một hàm gate0 có gradient chết để giả parity.

## 4. Loss chung, không có language head tách rời

\[
L=L_{\rm SCST,\ reward\ gốc}
+.1L_{\rm nounNLL}(p_{\rm new})
+.1L_{\rm ROIclass}+.1L_{\rm pointer}.
\]

- NounNLL chỉ eligible reference nouns, qua chính distribution dùng lúc generate; không CE một classifier proxy thay caption likelihood.
- ROIclass: CE trên các GT-matched predictions, positive class/OTHER; không dùng annotation absence để nhãn background.
- Pointer: \(L_{\rm pointer}=-\log\sum_{i\in M_c}\pi_{ti}\) tại eligible noun spans. Mean theo semantic events; DDP/global counts như các pilot trước.
- SCST giữ CIDEr+.5object reward như pipeline; classifier và pointer supervision dùng TRAIN, không dùng dev/panel175mixed labels/VSRheld.
- Không cập nhật B, không keep/UL cũ và không riêng một frame/GRU. Weights trên là starting design để chốt trước thử nghiệm, chưa optimized.

ROI classifier có nhiệm vụ semantic thật và được dùng trong q_c; π vừa được supervision vừa quyết định word probability. Tuy vậy, cả hai vẫn có thể sai; “có đường gradient” không là caption success.

## 5. Hai arm: control mạnh giữ cùng thông tin

Giữ A và B=C0 fixed. Chỉ train hai versions có cùng parameters/data/loss/init/schedule:

- **L-local (đề xuất):** key dùng e_i của đúng ROI.
- **G-global control:** key thay e_i bằng mean appearance e_bar của các proposals trong cùng ảnh; box/confidence và query h_t giữ. **ρ_i vẫn lấy đúng ROI e_i ở cả hai arms.**

G-global vẫn có visual object evidence trong lexical values và có toàn ảnh/ngôn ngữ/hình học để chọn. Nó mạnh hơn language-only hoặc copy topdetector. L−G kiểm giá trị của correspondence prefix↔ROI appearance cụ thể, thay vì chỉ thêm danh từ được detector gợi ý.

Đây là thay đúng một pooling input ở key; không vô hiệu hóa parameters hoặc bỏ vision classifier của control. Learned weights sau train có thể khác vì gradient khác, là effect cần đo.

Không được nói mass-preservation tốt hơn free-mixture chỉ từ hai arm này: cả hai cùng constraint. Constraint có property đại số đã nêu; lợi ích performance riêng so free-mixture cần ablation sau nếu claim ấy được đặt.

## 6. Phép thử nhỏ và test image use

Đề nghị500updates/arm,batch8,same seed42/same4000TRAIN order; chỉ head/pointer nhỏ được học. Đây là đề xuất nguồn/budget để user chốt, không authorization chạy. Không có một chuỗi classifier-pretrain/diagnostic trước caption: losses học cùng actual caption policy.

Endpoint free captions trên dev1000 đã dùng, với cùng scorer/fullCHAIR80 và decoding gốc. Phải so **L với B** để không nhận gain continuation thừa hưởng từ C0 thành benefit alignment; đồng thời so L với G-global. A là reference mục tiêu ban đầu.

Test correspondence trên cùng held noun spans có GT support (chỉ evaluator):
1. Tất cả ROI/key/value/box cùng permute phải invariant về distribution (technical equivariance).
2. Giữ h_t,p_B,boxes/confidences và lexical values ρ_i cố định; derange **appearance keys** giữa các ROI khác class trong ảnh. Đo mass/likelihood của correct noun giảm so đúng correspondence. Đây là representation intervention, không physical counterfactual proof.
3. G-global đã giữ language/global/geometry và đúng visual lexical values. Nếu L≈G hoặc key mismatch không ảnh hưởng đúng chiều, chưa có bằng chứng localized correspondence có ích.
4. Report set-pointer accuracy/mass, proposal coverage, null/OTHER fallback, actual used lexical mass và caption changes. Pointer đẹp mà caption không tốt hơn vẫn fail.

Không lấy heatmap/teacher-forced noun score thay cho caption outcome. Geometry-only/global shortcuts có thể vẫn đi qua CLIP features đã contextualized; control và interventions chỉ giới hạn cách giải thích, không loại mọi shortcut.

## 7. Go/no-go cần chốt trước run

Tiếp tục đầu tư xác nhận chỉ khi có cả:
- L có tín hiệu cải thiện CIDEr và giảm CHAIR false counts so **B**, không chỉ A; recall không giảm quá.005absolute. Report paired effect/CIs; CI rộng chỉ là exploratory, chưa dual-goal confirmation.
- L có added utility so G-global và correspondence test đúng chiều; không chỉ classifier/π losses thấp.
- Gain không do null/OTHER collapse, lexical coverage thấp, skip khó hoặc giảm output nội dung.

Không hơn B hai mục tiêu, không hơn control hoặc không sử dụng correspondence: dừng cấu hình tại endpoint đã khóa; không tăng strength/steps để cứu. Không có new-feature/headroom thì ghi giới hạn representation, không chứng minh alignment nói chung vô dụng.

Nếu có tín hiệu mới bàn confirmation/ablation gần prior; không tự dùng expanded4000 làm vòng tuning tiếp hoặc đổi các kết quả C/G trước.

## 8. Mạnh nhất vì sao có thể thất bại, và vì sao chưa bị refute

**Mạnh nhất:** ROI từ CLIP toàn ảnh có thể không giữ discriminative appearance; mọi vùng đều chứa cùng scene context. Khi đó ρ_i dựa vào co-occurrence, local pointer chỉ dùng geometry/prior. Hoặc detector bỏ sót vật khiến lexical branch đẩy sang vật khác; fallback không bảo đảm tránh lỗi. A/C0 đã mạnh nên headroom sửa noun đúng có thể nhỏ hơn số lỗi mới gây ra.

R/B và T/W chỉ tăng/đổi attention/memory values, chưa thử lexical probability correspondence này. I/J đưa hidden residual vào decoder; ở đây probability interface trực tiếp có constraint, host không coadapt. Keep/UL không học ROI↔word correspondence; ở đây positive GT region matching và same pointer trực tiếp quyết định p_new. Những khác biệt đó làm hypothesis mới đáng kiểm, không chứng minh nó sẽ thắng hoặc vô hiệu kết quả cũ.

## 9. Prior và claim boundary

Root cần đối chiếu Neural Baby Talk/grounded pointer-copy captioners, vocabulary-controlled decoding và nearest class-probability methods trước novelty claim. Attention/pointer/ROIclass/setloss/copy đều có prior. Không nhận đẳng thức bảo toàn mass là theorem mới.

Đối chiếu cụ thể: [Neural Baby Talk,CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Lu_Neural_Baby_Talk_CVPR_2018_paper.html) đã sinh template/slots gắn vùng rồi lexicalize bằng visual concepts từ detector (kiểm primary abstract, chưa nhận đã so mọi equation). Candidate này giữ captioner mạnh frozen, không học lại template và không copy detector class labels; sửa class identity ở output với invariants lexical mass/non-lexicon probabilities. Đây là delta adaptation cần chứng minh, không claim NBT hoàn toàn thiếu mọi constraint tương tự trước khi đọc đầy đủ.

R đã supervise native decoder→Q→patch box mass; candidate không chỉ thêm box-attention loss: π và ROI semantic values được dùng trực tiếp trong word probabilities. G/S có bboxpool khi TRAIN nhưng inference global class pooling+bias; candidate dùng ROI retrieval theo prefix **ngay tại inference** và không cộng global noun bias. Hard probability constraint ở cùng prefix không bảo đảm noun count/length toàn caption; tất cả guards thực nghiệm vẫn cần.

Ứng viên đóng góp vừa mức là **thiết kế region-grounded lexical redistribution trên một strong frozen captioner, tách bảo toàn lexical mass khỏi chọn object identity**, với control giữ global/geometry/lexical evidence và kiểm đúng correspondence được dùng. Nếu không có added utility thì chưa thành positive alignment contribution.

Phương pháp chỉ là recommendation brainstorm; root phản biện/prior-check rồi user chốt trước triển khai. Không đổi mục tiêu sang empirical-only và không hứa deadline/accepted.
