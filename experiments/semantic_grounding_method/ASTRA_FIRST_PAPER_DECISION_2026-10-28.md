# Chốt hướng cho bài đầu tiên: chia sẻ attention giữa đọc thực thể và đọc cặp

> Kế hoạch I/J v2 đã chạy xong. Cả hai kém A; quyết định tiếp theo ở [đánh giá sau kết quả](../shared_pair_readout/ASTRA_RESULT_DECISION_2026-10-28.md). Nội dung dưới đây giữ nguyên như lịch sử thiết kế trước run.

28/10/2026. Research decision sau cập nhật mục tiêu công bố và phản biện với root/Astra. **Chọn joint reader có công thức đóng; hai arm I/J và A cố định làm mốc.** Chưa viết code hoặc launch GPU.

## 1. Vì sao đổi lựa chọn

Mục tiêu hiện tại là bài đầu tiên gửi tạp chí của một trường đại học trong nước, chưa xác định venue. Ta cần một cải tiến có động cơ, đối chứng đúng và kết quả minh bạch; không cần đặt điều kiện phải có thuật toán nền tảng mới. Điều này không nói gì về độ dễ của tạp chí hoặc khả năng accepted trước 31/10.

Bỏ solver KL-marginal F khỏi kế hoạch đầu: nó kiểm một delta regularization khá hẹp, thêm lỗi số và khó giải thích trên VSR ít mơ hồ thực thể. Giữ cấu trúc đọc cặp nhưng kiểm một sửa đổi đơn giản hơn. Không rút phương án chỉ vì attention/ordered pairs đã có prior; cũng không coi prior ấy đã biến mất.

Khuyến nghị này supersede kế hoạch F–C* chưa chạy trong [đề xuất trước](ASTRA_METHOD_RECOMMENDATION_AFTER_LITERATURE_2026-10-28.md). **NO_GO của frame200 và gate đã thất bại giữ nguyên**: 40/64 vs 41/64, delta 1.5625pp, 95% CI[-12.5,+15.625]pp. Thí nghiệm mới đi từ A và caption decoder thật; không chạy tiếp để cứu frame200 hay dùng trọng số frame ấy.

## 2. Đơn vị đóng góp vừa mức

> Một cách tích hợp bộ đọc cặp đối tượng có thứ tự vào caption decoder, trong đó context thực thể và context quan hệ chia sẻ phân phối attention; kiểm so với cách đọc thực thể riêng khi giữ cùng pair features, pair attention, tham số, loss và ngân sách.

**Kế thừa:** subject/object/union, hình học tương đối, attention, residual projection, likelihood/ranking và SCST/KL đều là cơ chế đã có. Tham chiếu [MTTSNet](https://arxiv.org/pdf/1903.05942), [GRIT](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960165.pdf) và [GOT](https://proceedings.mlr.press/v119/chen20e.html), cùng [ledger 20 bài](ASTRA_SEARCH_AND_READING_LEDGER_2026-10-28.md). Không tìm thêm vòng tài liệu.

**Phần chỉnh sửa của nghiên cứu này:** trả marginal của attention cặp về hai entity contexts trong decoder A, thay cho trọng số đọc thực thể riêng; tối ưu qua đầu ra caption thật. Đây là một adaptation/readout inductive bias có thể đánh giá thực nghiệm, không nhận là nguyên lý joint attention mới hay “đầu tiên”.

Nó không chỉ thêm feature so với đối chứng: cả I và J có đúng cùng feature. Tuy nhiên J đại số tương đương attention trên pair values được ghép endpoint values; I cũng có thể tiếp cận endpoints qua r_ij. Vì vậy không có đặc quyền suy luận quan hệ hoặc thông tin mới chỉ J mới thấy.

Với phép ghép tuyến tính, đúng là:
\[
W_s\sum_i(T1)_ie_i+W_o\sum_j(T^\top1)_je_j+W_r\sum_{ij}T_{ij}r_{ij}
=\sum_{ij}T_{ij}[W_se_i+W_oe_j+W_rr_{ij}].
\]
Vì vậy gọi là shared-weight readout/cải tiến ứng dụng trên A, không gọi thuật toán feedback hay tối ưu mới.

Claim chính nếu kết quả hỗ trợ: **chia sẻ attention giữa entity và pair contexts giúp chất lượng/độ đúng quan hệ của caption trong điều kiện thử nghiệm**. Không đặt claim chính là sửa binding, chọn đúng referent hay chứng minh posterior hiểu predicate.

## 3. Pipeline khóa để triển khai

Giữ memory A gốc \(C_A\in R^{B\times(N_p+32)\times512}\), prompt gốc, Q-Former và visual projection. Dùng raw CLIP cache 197×768; 196 patch sau projection frozen của A có chiều 512.

- K≤12 predicted boxes/confidences, cùng quy tắc chọn ứng viên của v105. GT chỉ dùng nhãn/cohort/đánh giá; không chọn ROI bằng GT lúc forward.
- Object value \(e_i=P_e[\operatorname{ROImean}(V,b_i);b_i;conf_i]\in R^{512}\). ROImean dùng fractional patch overlap. Dùng appearance, box và confidence; không thêm embedding lớp ngẫu nhiên mới.
- Pair value \(r_{ij}=P_r[e_i;e_j;\operatorname{UnionMean}(V,b_i,b_j);g_{ij}]\in R^{512}\), với geometry 8 và thứ tự i,j đã dùng trong thiết kế trước. Pooling/geometry được tái sử dụng; không nạp learned frame200 weights.
- Thêm null key, value 0; giữ self pairs. Self/null interaction values bằng 0, dùng learned neutral key chung để tính score. Một bước sinh có thể đọc một vật hoặc không đọc cặp; không ép từ chức năng có quan hệ/bbox.
- Từ hidden \(h_t\) của decoder layer cuối, trước reader mới, tạo scores role1 \(a_{ti}\), role2 \(b_{tj}\), pair \(s_{tij}\) bằng projected dot products có chia căn chiều. Mọi query chỉ thấy ảnh và prefix; target assertion không là input riêng.

Cùng temperature cố định \(\tau=1\), đặt:

\[
\alpha_t=\operatorname{softmax}(a_t),\quad
\beta_t=\operatorname{softmax}(b_t),\quad
T_{tij}=\operatorname{softmax}_{ij}
 [\log\alpha_{ti}+\log\beta_{tj}+s_{tij}/\tau].
\]

Dùng log-softmax/logsumexp trực tiếp, không exp rồi chia thủ công. Không có solver, vòng Sinkhorn hay hệ số marginal KL. Cả I và J dùng **cùng công thức T**, cùng \(c_r=\sum_{ij}T_{tij}r_{ij}\).

| Arm | Context thực thể1 | Context thực thể2 | Pair context |
|---|---|---|---|
| I: đọc thực thể riêng | \(\sum_i\alpha_i e_i\) | \(\sum_j\beta_j e_j\) | \(\sum_{ij}T_{ij}r_{ij}\) |
| J: đọc chung | \(\sum_i(T1)_i e_i\) | \(\sum_j(T^\top1)_j e_j\) | \(\sum_{ij}T_{ij}r_{ij}\) |

Cặp ở I cũng đã được trọng số theo priors thực thể. Đây là control mạnh hơn ba reader hoàn toàn độc lập; chỉ cô lập **việc chia sẻ trọng số cặp với entity contexts**. Given cùng hidden/weights, T và pair context giống nhau; sau train weights có thể khác vì gradient khác. Khác gradient là hệ quả cần đo của readout, không phải bất công.

Cả hai dùng cùng \(h'_t=h_t+W_o[c_s;c_o;c_r]\), rồi vocabulary head 512→10000 thật của A. \(W_o=0\) lúc khởi tạo; không thêm normalization ngoài đường A. Output ban đầu cần khớp A. Giữ cùng encoder/features, queries, output projection và số tham số; không thêm truth head, UPT hoặc crop mới.

## 4. Loss và trainable scope: một recipe chung

Nguồn giám sát thực có: VSR v105: 256 TRAIN pairs/484 images; 64 held pairs/128 images; original features/prompts của 612 images đã có receipt. COCO references của TRAIN dùng caption loss. Không giả định đã đủ feature/box dev 1.000 hoặc quota; root kiểm inventory/mount trước paired run, không dựng một audit dữ liệu mới.

Trong cặp \(I^+,I^-,q\), cùng assertion q đúng trên ảnh+ và sai trên ảnh−; các predicate khác unknown. Cho \(\ell^\pm\) là mean log-prob toàn predicate+object-tail qua decoder A:

\[
L_{\rm rel}=-\ell^+
+\operatorname{softplus}(\ell^- -\ell^+),
\]
\[
L=L_{\rm SCST,\ reward\ gốc\ A}
 +.1\,\overline{KL(p_A\|p_\theta)}
 +.2\,L_{\rm rel}.
\]

KL teacher A stop-gradient trên COCO reference prefixes chung; normalize theo token rồi theo mẫu. Reward/normalization/gradient clip của SCST dùng đúng implementation A, không đổi reward để giúp J. Câu âm chỉ âm với q, không CE assertion sai làm target dương. Báo predicate-only log-prob để kiểm gain chỉ từ noun.

**Bỏ endpoint loss**, không thêm box-attention loss và không CE warmup riêng. Positive likelihood đã có trong L_rel; không cần một lịch train phụ chỉ vì projection 0. Không tuyên bố tách được tác dụng của L_rel: hai arm đều có nó.

Học P_e/P_r, keys/queries/neutral keys, W_o và **hai decoder layer cuối**. Frozen CLIP, visual projection, Q-Former, prompt fusion, input embedding, vocabulary head và decoder layer đầu. Starting LR: new modules 1e−4, decoder 5e−6; cùng optimizer/settings của A. Đây là recipe khóa trước, không khẳng định tối ưu.

Root preflight kiểm parity lúc 0, finite/gradient sau W_o bắt đầu học, scope frozen, cache binding và hai-GPU global update. Không đòi gradient mọi module khác0 ngay bước 0. Mỗi arm khởi tạo từ cùng A/state digest và cùng data order/RNG protocol.

## 5. Hai arm đầu và dữ liệu đánh giá

Khóa một seed và endpoint 500 updates/arm; không chọn best epoch. Starting global batch 8 caption images+4 VSR pairs/update, microbatch/accumulation do root chốt theo preflight; giữ effective batch và schedule bằng nhau. Với batch này, 500 updates tương đương khoảng 8 lượt qua từng nguồn, không gọi là full training hay hội tụ.

A fixed + I + J được decode bằng cùng tokenizer, beam/length settings gốc. Hai account có thể chạy I/J song song khi root đã xác minh quota; chưa có thời gian chạy được đo, không cam kết 60 phút hoặc mặc định 50 giờ còn lại.

| Kết quả cần có | Cách dùng |
|---|---|
| Free caption trên common dev 1.000 nếu đủ cache, disjoint TRAIN | CIDEr và các metric caption sẵn có; CHAIRs/i, object recall, length |
| Same-assertion ranking trên toàn 64 held | Chỉ báo chuyển giao qua decoder, báo cả hai strata; không thay free-caption outcome |
| Panel 32 pairs, 16/stratum khóa bằng hash, 64 images, 192 captions A/I/J | Chấm q được diễn đạt đúng/sai/không nhắc/uncertain; RC trên ảnh+, FR trên ảnh−, Y=RC−FR; giữ cả coverage |
| Params, device-hours, peak VRAM, captions/s hoặc ms/caption | Tradeoff I/J/A trên cùng máy và decoding; tách chi phí cache/detector khỏi phần caption đã cache |

Panel này là đánh giá câu mới sau train, không mở thêm audit/cohort. Không chọn cases theo J thắng. Dùng GT/labels ở đánh giá, không forward. Nếu bất định giữ bounds; nếu chưa có human adjudication thì ghi model-assisted, không nâng thành gold.

Report paired difference J−I, J−A và CI trên cùng ảnh; semantic bootstrap theo cặp/strata. Nếu dev thiếu tensors hoặc có overlap TRAIN thì khóa tập eligible trước run, báo số loại/giữ; không so A 1.205512599 từ 1.000 ảnh với một cohort khác mà không recompute A.

Các dev/held đã dùng nhiều vòng và có thể từng được A thấy ở giai đoạn khác: gọi đúng là development/adaptation-heldout được tái sử dụng, không độc lập hoặc test mới. Không tune trên 64 held; không dùng thay đổi subgroup để cứu overall. Giữ kết quả frame200 trong lịch sử thử nghiệm.

## 6. Quyết định sau pilot và đầu ra cho bài

Các ngưỡng F−A≥.005,F−C*≥.003,CI>0 và Y gain 4/32 ở proposal F–C* chưa chạy **không áp làm luật công bố cho I/J**. Đổi trước kết quả vì đổi câu hỏi/scope; không thay gate frame200 đã thất bại.

Khóa bảng quyết định theo ba loại bằng chứng, không chọn theo một scalar CIDEr 0.00x:

- **Đáng xác nhận tiếp:** J có tín hiệu tốt hơn I về đúng quan hệ/coverage, không đổi lấy giảm rõ chất lượng caption/recall và overhead chấp nhận được; hoặc có gain caption có CI hỗ trợ kèm semantics không xấu đi. Báo counts/CI thực; chênh nhỏ bất định chỉ gọi tín hiệu thăm dò, chưa gọi cải thiện.
- **Chưa thấy lợi ích riêng:** J≈I hoặc CI rộng/semantics ít được đề cập. Không mở sweep/solver để cứu. Kết luận pilot thiếu bằng chứng cho shared readout; nếu dừng vì ngân sách thì ghi lý do thực dụng.
- **Dừng cấu hình:** J có bằng chứng gây hại chất lượng hoặc hallucination/coverage, chi phí vượt mục tiêu, hay nhánh không được dùng. Technical bug sửa trước khi so kết quả; không đổi method sau nhìn held rồi giữ cùng nhãn thí nghiệm.

Không có threshold phổ quát về “overhead chấp nhận được” hay “đủ đăng” khi chưa biết venue/quota. Ghi trước priority: **độ đúng có coverage > CIDEr tăng rất nhỏ; tổn hại caption rõ ràng không được che bởi assertion score**. Mọi tradeoff cần bảng số cụ thể và root/user quyết định theo tài nguyên.

Bộ kết quả để viết một bản thảo vừa mức: bảng A/I/J chất lượng+semantics+cost, paired effect/CI, mô tả cache/split/đánh giá tái sử dụng, vài ca đúng/sai điển hình chọn theo quy tắc và toàn output để kiểm. Một inference ablation tắt nhánh cặp (cùng lúc đặt pair scores s=0 và pair values=0, giữ entity features/priors) kiểm sensitivity; không gọi là chứng cứ nhân quả hoặc pure-interaction ablation vì r_ij cũng chứa endpoints. Nếu đủ quota cho một việc thêm, ưu tiên **lặp I/J với seed thứ 2** hơn thêm kiến trúc; nếu chỉ một seed, trình bày đúng là nghiên cứu exploratory trong cấu hình này.

Nếu J hơn A nhưng không hơn I: chỉ mô tả hiệu quả toàn recipe so A, không quy công cho shared readout; chưa cô lập tác dụng riêng của feature hay relation objective. Nếu J hơn I: có bằng chứng cho delta readout trong recipe chung, vẫn không suy thành đã sửa binding. Nếu chỉ ranking tăng, caption không cải thiện: báo khoảng cách transfer, không gọi caption thành công. Bộ kết quả âm vẫn đủ cho báo cáo trung thực; việc có phù hợp để gửi venue nào chưa xác định.

## 7. Hai rủi ro còn lại và thứ tự công việc

1. **Readout có thể dư thừa:** r_ij đã chứa endpoints; I có thể học cùng thông tin qua c_r. VSR ít mơ hồ làm cơ hội J phát huy nhỏ hơn. Không coi prior của nhánh mới là lỗi có sẵn của A.
2. **Shortcut/overfit:** class/co-occurrence, geometry hoặc context phân biệt ảnh+/−; teacher-forced assertion tăng nhưng caption né quan hệ. Union CLIP có thông tin toàn ảnh, nên ROI không phải chứng cứ cục bộ thuần túy. RC/FR/coverage và caption toàn cục là kiểm chính.

Thứ tự cho root:
1. Chốt đúng I/J ở tài liệu này, kiểm metadata dev/held và quota hiện tại; tái sử dụng code/data helpers, không đọc lại ZIP hoặc tìm papers.
2. Triển khai một reader với switch chọn entity weights; shared pair code, loss, init/schedule. Chạy preflight kỹ thuật trước train.
3. Chạy đúng hai arm 500 updates; xuất checkpoint endpoint, free captions, likelihoods, timing/VRAM/provenance, không chỉ scalar scores.
4. Chấm panel và lập bảng effect/CI/cost; root phản biện kết quả rồi quyết định xác nhận seed 2, viết bản thảo exploratory hoặc dừng. Không tự thêm F, sweep hay lớp pretraining.

**Quyết định chốt: I/J closed-form shared-readout study, một delta nhỏ và một đối chứng mạnh; giữ chuẩn bằng chứng, điều chỉnh tham vọng novelty theo mục tiêu công bố thực tế.**
