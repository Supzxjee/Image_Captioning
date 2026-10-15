# Pilot học chấm caption: train1024 / tune256 / validation512

**Mục tiêu:** kiểm tra liệu một bộ chấm được học từ dữ liệu có chọn caption tốt hơn fixed reranking, và thông tin vùng ảnh có ích hơn chỉ học lại các điểm/global embeddings hay không. Chưa có kết quả pilot; người dùng tự chạy notebook Kaggle.

## Đối chứng và phương pháp

Giữ nguyên Q-Former gốc32q/2l epoch10, prompt cache gốc và beam5. Q-Former/CLIP đóng băng. Cùng candidates, cùng train/tune/evaluation images cho bốn chính sách:

| Chính sách | Đầu vào bộ chấm |
|---|---|
| Fixed | decoder/CLIP/OCC=.3/.4/.3, không học |
| Scores | Học từ6 đặc trưng điểm: decoder, CLIP, OCC, average logprob, độ dài, rank |
| Global | Cùng6 điểm + CLIP image/caption global embeddings và tương tác |
| Regional | Cùng global + attention từ token caption tới196 cached visual patches |

Bộ chấm thêm residual `0.2*tanh(head(features))` vào điểm fixed. Lớp cuối khởi tạo0: cả ba mô hình bắt đầu đúng lựa chọn fixed. Điểm fixed lưu float64 riêng để tránh đảo tie do làm tròn float32. Hidden128, attention4 heads. Regional attention dùng patch contextual từ CLIP cache, **không phải crop độc lập hay chứng minh đúng subject/object instance**. Đây là pilot kiểm tra một đường sử dụng bằng chứng ảnh, chưa phải tuyên bố phương pháp mới.

## Tách dữ liệu và supervision

- Seed42 chọn1.024 ảnh `train/restval` để học,256 ảnh train/restval khác để chọn epoch. Không gradient trên256 ảnh này.
- 512 ảnh `val` để đánh giá pilot sau khi đã khóa epoch. Không dùng `test`. Validation này là subset của nguồn đã xem, không phải đánh giá độc lập mới.
- Mỗi ảnh có tối đa5 candidates và5 caption tham chiếu. CIDEr từng candidate tính trên cùng reference corpus trong từng cohort để giữ IDF cố định.
- Loss so sánh cặp có CIDEr khác nhau: weighted softplus + regularization residual. Preference CIDEr là supervision yếu về khớp reference, **không được gọi là nhãn sai quan hệ đã xác minh**. Mô hình không nhận reference, CIDEr target hoặc nhãn đúng/sai làm input khi chấm caption.
- AdamW lr1e-3, weight_decay1e-4, batch8,10 epochs; chọn epoch tốt nhất trên tune256 theo CIDEr, tie giữ epoch sớm. Epoch0 cũng được xét, tránh buộc chọn checkpoint kém hơn khởi tạo fixed.

## Điều kiện mở rộng đặt trước

Nhánh regional phải đạt đồng thời trên validation512:

1. CIDEr ít nhất fixed +0.005.
2. CHAIRs và CHAIRi không cao hơn fixed.
3. Object Recall không giảm quá0.01 tuyệt đối.
4. CIDEr cao hơn cả scores-only và global-only.

Nếu chỉ nhánh scores hoặc global tốt hơn, ghi nhận đúng nhánh đó; không quy cho căn chỉnh vùng. Gate chỉ quyết định có đáng mở rộng, không chứng minh ưu thế tổng quát từ một seed/512 ảnh. Không tự mở full training hoặc dò lại theo test.

## Chạy Kaggle

Dùng [notebook đầy đủ](LEARNED_CAPTION_RANKER_PILOT.ipynb), Internet ON, GPU T4x2. Cùng5 dataset của validation/test trước:

- `vuthetam/mscoco-2014`
- `ducanh2403/qformer-epoch10`
- `ducanh2403/prompt-cache`
- `ducanh2403/visual-cache`
- `ducanh2403/objectdetectionecache`

Đường dẫn rõ đầu cell; không cần output validation hoặc cache UPT. Hai GPU sinh candidates/trích CLIP features theo hai shard độc lập. Ba head nhỏ học tuần tự trên GPU0; Q-Former và CLIP không được cập nhật.

Ghi feature cache riêng khoảng1–2GB ở working, không copy toàn bộ HDF5. Review ZIP loại cache lớn và checkpoint Q-Former; giữ3 head nhỏ, cohort/protocol, candidates, histories, predictions và caption/CHAIR metrics.

Kết thúc gửi `/kaggle/working/learned_caption_ranker_1024_512_review.zip`. Summary phải có `PILOT_COMPLETE`, fit1024/tune256/validation512 và `test_used=false`. Mở notebook không có nghĩa thí nghiệm đã chạy xong.

## Kiểm tra trước bàn giao

13 tests PASS: giữ baseline lúc khởi tạo kể cả tie rất gần; padding không gây NaN; gradient tới nhánh patch sau cập nhật; padded caption tokens không tác động kết quả; reference target không là input; split train/tune/val không trùng và loại test; các kiểm tra HOI baseline/journal/reuse/locked-test cũ vẫn qua. Syntax các script hợp lệ.

Đây là kiểm tra logic local với tensor tổng hợp và test hiện có. Chưa chạy training pilot hoặc inference GPU; chưa có CIDEr/CHAIR mới. Trợ lý không tự chạy notebook Kaggle.
