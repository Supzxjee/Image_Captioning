# Phản biện bản Argon sau C/T — 26/10/2026

Bản nhận: `brainstorm_c_l_p_sau_c_t_26_10_2026.md`. SHA256: `deca66566bcf8bbae64ea46d05a23e26e4e588d67303dfec1d168792b8b6eef1`.
Đã đọc toàn văn; bản nguyên trạng giữ trong ignored `review/`. Chưa chạy mô hình,
chưa duyệt CEV-PG hoặc bốn nhánh GPU. Đã gửi Astra tóm tắt cơ chế và phản biện cụ thể
để ghép vào quyết định phương pháp đang làm, không yêu cầu đọc ZIP lại.

## Phần có ích

Argon đề xuất kiểm chứng từng token/noun của prompt bằng bằng chứng từ Q-Former,
rồi thêm residual vào grounded prompt memory. Không dùng ảnh che ở inference.
Alpha khởi tạo0 có thể giữ đường A lúc khởi tạo nếu mọi phần khác không đổi.
Đề xuất inventory không GPU là hợp lý: cần biết lỗi CHAIR của A có thật sự trùng với
những lớp được nhắc trong prompt nhưng không được annotation xác nhận hay không.
Đếm nhiễu prompt đơn thuần chưa đủ nếu decoder đã bỏ qua chúng.
Đọc cache metadata và dự đoán lưu sẵn rẻ hơn thêm một lượt train, nhưng full original
prompt JSON phải đúng fingerprint/budget20tokens; không thay bằng UPT text mới.

## Những kết luận cần sửa trước khi thành protocol

1. **Chẩn đoán nhân quả:** silhouette/context có thể giải thích một phần p00, nhưng
   p00 trung bình không chứng minh Bayes-optimal fusion hay stale loss gây mất24
   correct objects. C cũng giảm và T−C CI cắt0; nguyên nhân chưa được cô lập.
   CE trên5references không bảo đảm tạo phân phối đều, và chưa được xác nhận là
   nguyên nhân regression. C0 chỉ kiểm chế độ continuation CE+KL đang xét.
2. **Nhãn:** không xuất hiện trong COCO instances/captions không đồng nghĩa verified
   absence. Caption không liệt kê mọi vật; COCO giới hạn80classes và có thể thiếu
   instance. Cross-image sampled negatives cũng cần chính sách xác nhận/unknown.
   Phân biệt annotation-relative rate với visual falsity, không gọi weak negatives
   là những false positives thực sự đã biết.
3. **Toán residual:** Pmod=Pvis+tanh(alpha)(2g−1)Wmod LN(h) không bảo đảm g thấp sẽ
   giảm xác suất đúng noun. Alpha có thể âm, Wmod không bị ràng buộc dấu và Jacobian
   của decoder có thể đảo tác động. Đây là modulation học được, không là phép chứng
   minh suppression. Sau khi mở base, đặtalpha0 không trả lại original A.
4. **Số tham số:** Wp,Wh128×512 và Wmod512×512 đã393.216weights, chưa MLP/LN/bias;
   vì vậy claim~0,20M cho toàn module sai. Từ layers hiện tại512D/FFN2048/vocab10k,
   lastQlayer +lastDecoderlayer +vocabhead đã13.538.064parameters, chưa nhánh prompt;
   claim~9Mbase không đúng với chính tập mở được mô tả. Đây là đếm theo công thức
   source, không import/chạy model local; actual trainable count phải xác nhận khi build.
5. **Shortcut:** contextual CLIP token còn mang sentence/position context. Cross-image
   same-word sampling không tự chứng minh text-only verifier không đạtloss dướilog2.
   Cần test text-only/shuffled-image phù hợp và kiểm tính hợp lệ nhãn.
6. **Cơ chế:** Normal thắng Inverted gate có thể chỉ do can thiệp OOD. Đó là usage
   sensitivity, chưa chứng minh grounding đúng. Normal=Zero-Mod/Inverted cũng không
   chứng minh mọi gain là ngẫu nhiên: có thể base đã thích nghi hoặc greedy không đổi.
7. **Đối chứng/chi phí:** đổi reference selection,KL,LR,trainable set và thêm verifier
   đồng thời cần matched control. Bốn nhánh ngay trước feasibility chưa là minimum
   decisive experiment. Thời gian3phút của C/T không là ETA module mới; staging mới
   có thể mất lại19–63phút. Các threshold mới là đề xuất chưa khóa, không áp vào C/T cũ.
8. **Giới hạn suy luận:** vài continuation failures không chứng minh mọi CE adaptation
   đều thất bại, A chạm trần kiến trúc, hoặc A vượt một baseline không prompt chưa
   được đánh giá cùng protocol. Không dùng kết quả này để tuyên bố giới hạn tối ưu.

## Tính mới cần kiểm chứng

A đã có `prompt_visual_gate` theo token/feature, nhìn cả prompt và retrieved visual
information (`captioning/models.py`). Vì vậy khác biệt ứng viên là explicit verification
supervision và cách tác động tới memory, không phải lần đầu có gate ở từng token.
`captioning/evidence_routing.py` cũng đã có verifier và abstention trong lịch sử dự án.
Cần phân biệt kỹ thuật và kết quả với những nhánh này trước khi đặt claim mới.

Có các primary works thật để đối chiếu, nhưng chúng không xác nhận tính mới CEV-PG:

- [M3ID, CVPR2024 / bản tác giả](https://arxiv.org/abs/2403.14003): đã khai thác
  chênh lệch phân phối có/không conditioning ảnh để tác động sinh câu. Không được gọi
  riêng ý tưởng tăng visual reliance là đóng góp mới. Chỉ đối chiếu cơ chế, không
  suy kết quả LVLM lớn sẽ chuyển sang decoder Q-Former nhỏ của chúng ta.
- [DiCO, BMVC2024, tác giả](https://arxiv.org/abs/2408.14547): direct CLIP-based
  optimization và neo mô hình. Không gộp mọi DPO/DiCO thành phương pháp bắt buộc che ảnh.
- [CHAIR-DPO, bản tác giả2025](https://arxiv.org/abs/2508.20181): CHAIR-based preference
  optimization tồn tại. Sự tồn tại paper không làm automatic preference labels của
  chúng ta hợp lệ; screen30pairs trước vẫn fail.
- [Geigle et al., EMNLP2024](https://aclanthology.org/2024.emnlp-main.159/): nghiên cứu
  grounding objectives và hallucination trong open caption generation. Cần đánh giá
  câu tự sinh, không suy caption gain từ grounding score đơn lẻ.

Đã xác minh metadata/abstract và phần nội dung M3ID qua nguồn tác giả; không tuyên bố
đã đọc fulltext tất cả bài hoặc kiểm xong nearest-prior của CEV-PG/CSA-Net.

## Khuyến nghị hiện tại

Giữ A. Tiếp tục trao đổi Astra về một ứng viên có nguồn giám sát hợp lệ, flow/loss và
matched pilot cụ thể. Có thể giữ inventory prompt–error làm feasibility, nhưng báo
đúng weak-label/association scope. Chưa chạy C0/CEV-PG hoặc tự chuyển sang SCST/new
query projection. Chờ quyết định phương pháp đã xử lý các phản biện trên.
