# Astra decision — 2026-10-27

Source: thread `01a15544-6a7a-7903-a6e1-23e7b9f6caa6`, turn `01a17b8f-c0ea-7551-b066-80c0d3d5813f`. Design decision only; no ML code/run approved by this document. Paper links below are Astra's citations, not a claim root independently read DLCT.

**Tôi chọn giữ giả thuyết đọc lại patch trước Q-Former và đưa nó tới một pilot có đối chứng. Bản SD-RPR hiện tại chưa đủ để chạy; phần “orthogonal semantic residual” và loss foil nên bỏ.** Đây là ứng viên kỹ thuật để tìm caption gain, chưa phải phương pháp có novelty được xác lập.

Điểm mới so với các thử nghiệm trong dự án là **đường truy cập trực tiếp từ trạng thái decoder tới các token trước khi nén thành 32 Q**, rồi tác động lên phân phối sinh thực tế. Chưa có bằng chứng đường này sẽ hữu ích. Nó cũng không phát minh attention theo bước thời gian: decoder A vốn đã có cơ chế ấy.

**Thiết kế tối thiểu tôi chọn như sau.**

Giữ toàn bộ A đóng băng. Lấy \(h_t\in\mathbb R^{512}\) là đúng trạng thái A đưa vào vocabulary head. Nguồn patch là:

\[
Z_P=\operatorname{Proj}_A(V_{\mathrm{patch}})
\in\mathbb R^{196\times512},
\]

dùng projection gốc đã đóng băng. Như vậy pilot kiểm **bypass Q-Former**, chưa kiểm việc bỏ cả projection \(768\to512\).

Thêm một attention head:

\[
q_t=\operatorname{LN}_0(h_t)W_q,\quad
K=\operatorname{LN}_0(Z)W_k,\quad
V=\operatorname{LN}_0(Z)W_v,
\]

\[
r_t=\operatorname{softmax}(q_tK^\top/\sqrt{128})V,\qquad
\Delta h_t=r_tW_o,
\]

\[
\ell_t=\ell_{A,t}+W_{\mathrm{vocab},A}\Delta h_t.
\]

Ba projection vào 128 chiều, \(W_o:128\to512\); tất cả không bias, LN không affine. Tổng **262.144 tham số mới**. Khởi tạo \(W_o=0\), các ma trận còn lại khởi tạo thông thường: logits ban đầu bằng A.

Không dùng top-K, entropy gate, phép chiếu trực giao hay MLP nhận trực tiếp \(h_t\). Lựa chọn này làm giả thuyết và số tham số rõ hơn; **không chứng minh loại hết shortcut**. Các CLIP patch vẫn chứa ngữ cảnh toàn ảnh, không phải bằng chứng cục bộ độc lập.

**Loss phải tối ưu phân phối sau khi cộng residual.**

Tôi chọn SCST với reward đã pin của dự án, cộng KL neo A:

\[
L=-\mathbb E\!\left[
\operatorname{sg}(R(y_s)-R(y_g))
\sum_{t\le EOS}\log p_\theta(y_{s,t}\mid y_{s,<t},I,P)
\right]+0.1L_{\mathrm{KL}},
\]

trong đó \(y_s\) là multinomial temperature 1, \(y_g\) là greedy của mô hình hiện tại. KL lấy mean valid token giữa \(p_A\) và \(p_\theta\) **trên cùng sampled prefixes**; các prefix và A được detach.

Nguồn giám sát là COCO references qua reward hiện có và phân phối A cho regularization. Không cần coi từ foil hoặc caption của ảnh ngẫu nhiên là sai. KL giảm áp lực rời A nhưng **không bảo đảm** giữ CIDEr hay độ trôi chảy. Không thêm CE warm-up trong pilot này; cũng không kết luận CE là nguyên nhân các lần regression trước.

**Đối chứng quyết định là hai nhánh train có cùng module và loss:**

| Nhánh | Nguồn \(Z\) |
|---|---|
| P — treatment | 196 projected patches trước Q-Former |
| Q — control | 32 visual queries của A |

Cùng kích thước feature 512, tham số, initialization, optimizer, dữ liệu, số updates và seed. A gốc là baseline thứ ba, không cần train lại.

Đối chứng này trả lời câu hỏi thực dụng: **đọc lại token trước Q có ích hơn thêm cùng một readout trên Q sẵn có không?** Nó tránh confound vừa đổi nguồn vừa bỏ patch loss của bản Argon. Permutation lúc inference chỉ được xem là sensitivity test phụ.

**Pilot đề xuất:** 2.000 ảnh hash-first từ TRAIN hợp lệ, hai epochs, global batch 8 → 500 updates mỗi nhánh; AdamW \(10^{-4}\), weight decay .01, clip 1, một seed, end-only. Baseline/reward/evaluator dùng đúng nguồn và IDF đã pin, không dựng IDF riêng cho 2.000 ảnh. Atlas 20 cases chỉ minh họa, không làm train set hay gate.

Trên dev1000 hiện tại:

- P−A ≥ .01 CIDEr, paired 95% CI lower > 0;
- P−Q paired CI lower > 0;
- CHAIRs/i không xấu, object recall không giảm quá .005; báo độ dài câu.

Đây là development evidence trên tập đã dùng nhiều lần, chưa là xác nhận độc lập. Nếu P và Q cùng tăng tương đương, kết quả ủng hộ việc thêm readout/adaptation, **không ủng hộ riêng bypass Q**. Nếu P hơn Q nhưng không hơn A, chưa đạt mục tiêu. Nếu thất bại, dừng recipe này; không suy ra trần của mọi post-hoc alignment.

Không nhận ETA 3–5 phút. Có thể đặt cap chuẩn bị chung 30 phút và train+end-eval 45 phút mỗi nhánh trên T4×2; đây là **giới hạn ngân sách**, cần kiểm throughput sớm, không phải dự báo runtime.

**Về paper, tôi chưa coi đây là đóng góp mới.** RSTNet đã đề xuất điều chỉnh đóng góp visual/language trên decoder; DLCT đã khai thác kết hợp grid/region và cross-attention giữa các nguồn. Tôi kiểm abstract chính thức RSTNet và đọc phần method DLCT; chưa hoàn thành novelty search. [RSTNet](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_RSTNet_Captioning_With_Adaptive_Attention_on_Visual_and_Non-Visual_Words_CVPR_2021_paper.html), [DLCT](https://ojs.aaai.org/index.php/AAAI/article/download/16328/16135).

Phần có thể đáng phát triển tiếp là bằng chứng rằng **một readout nhỏ trước Q cải thiện sinh tự do trên baseline mạnh, vượt đối chứng cùng capacity đọc Q**. Có gain mới xét tiếp cơ chế ngữ nghĩa và khoảng cách với prior. Vocabulary vẫn cố định 10.000 từ; không claim open-vocabulary, bảo toàn toàn bộ fine cues hay không distribution drift.

Tôi chốt **đáng hoàn thiện thiết kế pilot hai nhánh này**, chưa launch hoặc viết code ML. Chưa có kết quả để gọi nó là phương pháp tốt hơn A hay nội dung paper đã khả thi.
