# CSA + SCST + fixed reranking: kết quả Karpathy TEST5000

Ngày kiểm tra: 05/11/2026. **Hoàn tất đủ 5.000 ảnh TEST**, không huấn luyện hoặc chọn hệ số trên TEST. Người dùng xác nhận output ở `ducanh2403/2-vi-cai-kia-qua-nhieu/166`. [Protocol](RERANK_TEST_PROTOCOL.md), [số liệu và khoảng tin cậy](RERANK_TEST_RESULT.json), [mã kiểm tra metadata/text](audit_rerank_test_result.py).

**Bước chọn lại caption cải thiện đồng thời CIDEr, CHAIR và recall trên cùng checkpoint.** Kết quả cùng chiều với VALID164; có thể chốt cấu hình này để viết báo cáo. Không tự mở thêm lượt train, hệ số hoặc nhánh mới.

## So sánh chính: cùng mô hình, cùng tập ảnh, cùng 5 ứng viên/ảnh

| Thước đo | Chọn theo decoder | Chọn lại bằng CLIP + đối tượng |
|---|---:|---:|
| CIDEr ×100 ↑ |108,1053|**117,0633**|
| BLEU-4 ↑ |0,348912|**0,364142**|
| ROUGE-L ↑ |0,545620|**0,567807**|
| CHAIRs ↓ |5,56%|**4,40%**|
| CHAIRi ↓ |4,0955%|**2,9568%**|
| Recall đối tượng ↑ |41,3946%|**44,6399%**|
| Caption bị CHAIR đánh dấu ↓ |278|**220**|
| Lượt nhắc đối tượng sai ↓ |295|**234**|
| Cặp lớp đối tượng/ảnh khôi phục đúng ↑ |6.518|**7.029**|
| Số từ trung bình |8,4278|9,2188|
| Caption không quá 2 từ ↓ |401|**61**|

Chênh lệch chính: **+8,9581 điểm CIDEr**, CHAIRs giảm **1,16 điểm phần trăm**, CHAIRi giảm **1,1387 điểm phần trăm**, recall tăng **3,2453 điểm phần trăm**. Số caption bị CHAIR đánh dấu giảm 58/278, tương đương khoảng 20,9%. Có 2.971 caption thay đổi; sửa được 94 trường hợp bị đánh dấu trước đó, nhưng phát sinh 36 trường hợp mới. Vì vậy không được nói phương pháp loại bỏ hoàn toàn ảo giác.

Bootstrap ghép cặp theo ảnh, 10.000 lần, seed 20261105; tính lại tỷ lệ từ số đếm ở mỗi lần lấy mẫu. Khoảng 95% cho chênh lệch: CIDEr **[+7,6964; +10,2277] điểm**, CHAIRs **[−1,62; −0,70] điểm phần trăm**, CHAIRi **[−1,4585; −0,8303]**, recall **[+2,8735; +3,6200]**. Các khoảng này có điều kiện trên mô hình và tập ảnh hiện tại, không bao gồm biến thiên giữa seed huấn luyện hoặc ảnh hưởng của việc dự án đã xem TEST trước đây.

## Pipeline được đánh giá

Ảnh → ROI/đặc trưng ảnh đã cache → **CSA encoder–decoder có căn chỉnh ngữ nghĩa kế thừa** → checkpoint đã học XE 15 epoch rồi SCST + CE một epoch → beam search sinh 5 caption → điểm decoder + điểm CLIP ảnh–câu + độ nhất quán đối tượng detector → chọn một caption.

Hai cột dùng cùng checkpoint của version161, SHA `58ce5dc0e30ac26add9bac8e24a2e9e03b91532bfdeece588997395bf9cfc16b`; beam5, max20. Cột trái chọn hạng đầu của decoder, cột phải dùng công thức đã chốt từ VALID164: **0,3 min–max(log-prob decoder) + 0,4 min–max(CLIPScore) + 0,3 độ nhất quán đối tượng chuẩn hóa**; detector confidence ≥0,5, penalty1, hòa chọn beam sớm hơn. CLIP `openai/clip-vit-base-patch16`, revision `57c216476eefef5ab752ec549e440a49ae4ae5f3`, model/processor theo cùng bộ SHA của VALID164. Caption tham chiếu và CHAIR chỉ được dùng để đánh giá sau khi chọn, không tham gia chọn câu. Không dùng oracle hoặc BLIP ở lượt TEST này.

Reranking không cập nhật trọng số nên không tạo ra mức giảm loss huấn luyện. Lượt này không chạy teacher-forced NLL; không có số loss mới để so sánh.

## Đối chiếu với các mốc cũ

- VALID5000: cùng bước reranking tăng CIDEr 106,3103 → 115,2416; caption bị CHAIR đánh dấu 260 → 206. TEST5000 ở bảng trên có xu hướng tương tự. Không trộn hai tập thành một điểm chung.
- CSA10 XE trên cùng TEST5000 trước đây đạt CIDEr 101,9207, 309 caption bị đánh dấu. Pipeline cuối đạt 117,0633 và 220, tương ứng **+15,1427 điểm / giảm 89 caption**. Chênh lệch này gồm **thêm 5 epoch XE, SCST và reranking**; không phải tác động riêng của căn chỉnh hoặc đối chứng cùng ngân sách huấn luyện.
- Bảng NoCSA10/CSA10 TEST đã có ở [kết quả đối chứng kiến trúc](PLAIN_VS_CSA10_TEST_RESULT.md). Giữ bảng đó riêng để tránh quy toàn bộ mức tăng của pipeline cuối cho CSA.

## Kiểm tra và bảo toàn

ZIP SHA `3983dfdf20a62fba0ad8791b453431cf92526bb7034ebf08f8c3094e75dc6e15`, CRC PASS; đọc một lần rồi cache text có kiểm soát. Runtime source `19e7dfe1fb0b9de9f6d047c644a3b429289bb6f4`. **95 kiểm tra metadata/text/count PASS**, bao gồm 26 source/asset pins, đủ 5.000 TEST ID và 25.000 ứng viên, tách ID khỏi VALID, checkpoint/cache identity, ánh xạ token–caption, SHA các file beam/score, tái tính phép chọn cho toàn bộ ảnh, trung bình CIDEr theo ảnh, số đếm CHAIR/recall và số trường hợp sửa sai/phát sinh sai. Log người dùng có 7 test PASS, không traceback.

Cache TEST được khôi phục bằng pooling CPU từ H5 gốc, **không chạy lại trích xuất đặc trưng bằng mô hình**. Receipt báo khớp nguyên byte với ba SHA đã chốt từ version143:

| File | SHA256 |
|---|---|
|test_regions.npz|7fb385e1a03c5241c4d65cc240d0434e041284a9a829198c5167f0725be57b1a|
|test_metadata.json|4271b46b7ab67117433bfdf84f6eb9ba66515ada9b3c4ae4d2e507bc2cede018|
|vocabulary.json|924be210aa0daa8b0968a6d69ad2c66b3de603f49c35f4aa67247c75d6173fc7|

143 là nguồn tham chiếu identity; output hiện tại là **166**, không phải đã tải lại thành công output143. Metadata/vocabulary được kiểm SHA từ text lưu trong ZIP; NPZ/checkpoint/JPEG/H5/detector binaries không nằm trong ZIP, kiểm tra các binaries đó là bằng chứng runtime do người dùng chạy.

CPU restore 165,14 giây; hai worker decode 222,82/221,48 giây; hai worker chấm điểm 88,63/88,27 giây. Đây là thời gian từng pha, không gồm toàn bộ setup/download/staging/metrics, không phải số giờ GPU bị tính phí.

Giữ toàn bộ `/kaggle/working/csa_ce161_fixed_rerank_test5000_v2` của **Save Version166**, gồm score/beam/cache receipts và metric outputs; trọng số vẫn là checkpoint161, không có checkpoint mới được train. Kết quả thành công này thay thế trạng thái V2 chưa chạy và lỗi tải cache ở lượt trước, không xóa lịch sử lỗi.

## Giới hạn và kết luận sử dụng

Tác giả kiểm metadata, text, phép chọn và số đếm; không chạy lại Torch, mô hình hoặc scorer chính thức trên máy local. BLEU/ROUGE và điểm CIDEr từng ảnh là kết quả scorer người dùng chạy. CHAIR đánh giá đối tượng thuộc bộ lớp COCO, không chứng minh mọi quan hệ/thuộc tính đúng; thiếu detector có thể là bỏ sót/top-K. TEST đã được xem ở các vòng trước của dự án: công thức lượt này được chốt trên VALID trước TEST, nhưng không mô tả đây là TEST chưa từng tiếp xúc. Hiện chỉ có một seed huấn luyện.

Kết luận phù hợp: **pipeline kế thừa căn chỉnh CSA, kết hợp SCST + CE và chọn câu bằng CLIP/đối tượng, đạt cải thiện chất lượng caption và giảm ảo giác đối tượng trong thiết lập thực nghiệm này**. Chưa có bằng chứng cho một module căn chỉnh mới, SOTA hoặc cải thiện tương đương ở mọi dữ liệu. Khuyến nghị chốt cấu hình, bảng kết quả và bắt đầu viết bài/báo cáo.
