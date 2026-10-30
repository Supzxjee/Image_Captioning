# Quyết định sau C0 — 30/10/2026

Đọc AGENTS.md, RESEARCH_STATE.md, C0_RESULT.json và báo cáo root; không đọc ZIP/chạy model. State đã được root cập nhật cùng lúc: C0 hoàn tất, commit5efa385 và registryR213, không còn trạng thái chờ run.

**Chốt: dừng training C/G/C0 đợt này. Giữ C0 làm continuation baseline có gain, không nhận là semantic-alignment contribution mới. Nếu đầu tư thêm trước hạn, đề xuất duy nhất là một evaluation lớn A/C0/C đã khóa; cần user chốt ngân sách trước triển khai.**

## Bằng chứng và attribution

- C0−A CIDEr+.025229,paired95%CI[+.006253,+.043992].
- C−C0−.005063,CI[−.014020,+.003438].
- G−C0−.001206,CI[−.008235,+.005985].

Gain vẫn có khi keep=negative=0: chưa quy được gain C/G cho hai proposed alignment losses. CI chứa0 chưa chứng minh preservation luôn vô dụng, gây hại hoặc tương đương. C0 cũng chưa tách được SCST khỏi rolloutKL/exposure, nên không đổi tên recipe thành contribution mới.

CHAIRs A/C0/G52/1000,C51; mentions A/C0/G53,C52. C0 CHAIRi thấp hơn A do denominator1632→1646, không do giảm false mentions. Chênh1caption của C chưa chứng minh reduction. Supported-mass separation C cao hơn C0 một chút chưa thay caption endpoint utility.

C0 có recall1498/3253 so A1490/3253 và CIDEr tốt hơn trong run này. Cần bảo tồn kết quả tích cực ấy, đồng thời giữ giới hạn one seed/reuseddev. Mục tiêu gốc “CIDEr tăng và hallucination giảm, kèm contribution căn chỉnh” chưa đạt.

## Bước tiếp theo được đề nghị

Không đề nghị C00 bỏ KL, tăng keep/UL, train thêm ảnh hoặc một kiến trúc khác trước31/10. Các bước ấy dễ thành thêm control hoặc tuning mà chưa giải quyết mục tiêu contribution.

Nếu còn tài nguyên, trao đổi với user để khóa **một evaluation A/C0/C**:
1. Dùng nguyên checkpoints/decoding/evaluator, không tune hoặc chọn epoch.
2. Ưu tiên phần val có original caches, không giao TRAIN và không nằm trong dev1000. Nếu val5000 bao gồm dev1000 thì phần thêm dự kiến4000IDs; root xác nhận manifest một lần trước run.
3. Các tập val/test đã dùng trong lịch sử phải khai báo; không gọi phần thêm là blind independent test nếu từng được xem. IDs và contrasts khóa trước inference.
4. Primary: C−C0 về factuality và tradeoff CIDEr/recall; C−A về hai mục tiêu gốc. C0−A xác nhận continuation quality. Không cần G trong main comparison.
5. Report full CHAIR80, absolute false-caption/mention counts, recall/length và paired CIs. Phần ảnh thêm báo riêng; pooled5000 là secondary nếu chứa dev1000 đã dùng để chọn.
6. Ước tính cost từ timing inference đã đo, kiểm quota hiện tại và user chốt cap. Chưa có approval mở rộng; không tự launch.

Đánh giá lớn không làm weights ít hallucination hơn. Nó chỉ thu hẹp bất định; chênh nhỏ có thể vẫn chưa rõ hoặc đảo dấu. Đây không phải hứa cứu C.

## Khóa cách đọc kết quả trước evaluation

- Nếu C giảm hallucination so C0 với quality tradeoff hợp lý và tốt hơn A trên cả hai trục: có căn cứ phát triển positive preservation trong phạm vi thử nghiệm, vẫn ghi post-pilot selection và one seed.
- Nếu C0/C đều có quality gain nhưng không có lợi ích riêng của preservation: giữ performance result, không nhận method attribution.
- Nếu hallucination vẫn không giảm hoặc effect bất định: kết thúc claim dual-goal/method-specific đợt này; không dùng tập lớn làm vòng tuning tiếp.
- Muốn claim đồng thời cải thiện hai mục tiêu so A phải có evidence cả hai, không chỉ point estimate hoặc denominator CHAIRi. Không gọi CI chứa0 là equivalence; margin thực dụng nếu dùng phải chốt trước inference.

## Thời hạn và bài báo

Không tự thay mục tiêu user thành chỉ tăng CIDEr. Hiện có một kết quả empirical về native continuation và hai proposed grounding objectives chưa cho added benefit. Chưa có positive-method result đạt yêu cầu gốc.

Nếu bắt buộc có bản thảo31/10, phương án trung thực là bài nghiên cứu so sánh/negative findings kèm continuation baseline gain, **nếu user chấp nhận scope ấy**. Nếu giữ yêu cầu contribution căn chỉnh mới đã chứng minh và hallucination giảm, cần trao đổi lại thời hạn; không thể bảo đảm một method mới thắng trong một ngày.

**Khuyến nghị riêng: không thêm training; chỉ đề xuất một evaluation A/C0/C để chốt uncertainty nếu được user duyệt. Không đổi tên C0 thành novelty.**
