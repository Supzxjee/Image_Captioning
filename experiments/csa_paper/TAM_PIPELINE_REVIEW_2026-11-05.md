# Đối chiếu pipeline của Tâm với CSA hiện tại

05/11/2026. Phạm vi: đọc báo cáo tuần12 của Vũ Thế Tâm và bốn hình pipeline, đối chiếu mã CSA hiện tại; đề xuất kế thừa, **chưa duyệt triển khai hoặc GPU**. Nguồn local: `C:/Users/leduc/Downloads/Báo cáo kết quả nghiên cứu tuần 12.docx`. Các kết quả của Tâm là số liệu báo cáo, chưa có code/checkpoint/split/metric-replay để kiểm chứng độc lập.

## Bằng chứng trong báo cáo

Kho offline chứa caption TRAIN, vector CLIP ViT-H/14 1024 chiều và objects/attributes/triplets từ FACTUAL + SimpleNLG. Ảnh truy hồi top10 bằng CLIP; tùy cấu hình BLIP ITM xếp lại trước khi chọn K. Facts được gom và loại trùng, mã hóa bằng CLIP ViT-B/32; visual patches làm Q, fact embeddings làm K/V trong cross-attention, AddNorm/FFN tạo fused memory cho decoder4lớp. ITM ở đây chọn **ngữ cảnh trước sinh câu**, khác ITM chọn caption đầu ra mình đã thử.

| Cấu hình của Tâm | CIDEr | CHAIRs | CHAIRi |
|---|---:|---:|---:|
| Baseline |110.74|10.18%|7.25%|
| Text RAG / CLIP |120.27|5.22%|3.55%|
| Text RAG / ITM |120.67|5.16%|3.53%|
| Fact RAG / CLIP |120.05|5.28%|3.60%|
| Fact RAG / ITM |121.95|4.96%|3.42%|
| Fact RAG / ITM + Fact Memory |120.23|5.30%|3.60%|
| Fact RAG / ITM + Context Memory |122.18|5.10%|3.46%|

**Sửa câu trả lời trước:** 121.95 là Fact RAG chính, không phải số cao nhất của báo cáo; cao nhất là122.18. Thêm Context Memory chỉ +.23CIDEr và xấu đi cả hai CHAIR; thêm Fact Memory giảm1.72CIDEr. Chỉ một seed, chưa có khoảng tin cậy. Không lấy +11.21 của Fact RAG so baseline để dự đoán mức tăng trên CSA của mình. Báo cáo không nêu đủ split/epoch/SCST/decoding/evaluator để xếp hạng công bằng với TEST117.06 của mình.

## Điểm khác biệt hữu ích

`model.py:SemanticAggregation.forward(bank)` tạo prototype từ bank danh/động từ đơn, không nhận ảnh. CSA hiện tại **đã căn chỉnh theo ảnh** ở Alignment/Fusion phía sau; không được nói CSA chỉ dùng bank chung nên không có căn chỉnh. Tâm bổ sung cụm có cấu trúc được truy hồi riêng cho ảnh (ví dụ `man riding motorcycle`), cung cấp tổ hợp ngữ nghĩa mà từ rời `man`, `ride`, `motorcycle` không thể hiện đầy đủ. Facts vẫn là giả thuyết lấy từ ảnh khác, không phải bằng chứng đã xác nhận cho ảnh query.

Ưu tiên nghiên cứu tiếp là **truy hồi cụm mô tả theo ảnh trước khi sinh**, giữ backbone/căn chỉnh CSA và reranking cuối làm đối chứng hiện có. SmallCap cũng mô tả sinh caption có điều kiện theo ảnh và caption truy hồi: https://openaccess.thecvf.com/content/CVPR2023/html/Ramos_SmallCap_Lightweight_Image_Captioning_Prompted_With_Retrieval_Augmentation_CVPR_2023_paper.html . Đây là hướng có tiền lệ, không phải phát minh mới.

Sơ đồ ứng viên: ảnh → hai đường (CSA tạo Vo/Vr; CLIP truy hồi caption TRAIN → lọc/loại trùng → cụm văn bản và embeddings) → một khối cross-attention bổ sung vào visual memory → decoder → 5caption → reranking cố định hiện tại. Vùng ảnh làm Q, cụm truy hồi làm K/V; phép chiếu và cầu nối mới phải được huấn luyện. Vị trí cụ thể lên Vo/Vr, gate, loss, checkpoint và ngân sách **chưa chốt**. Không nối embedding mới vào checkpoint cũ rồi coi là mô hình đã biết sử dụng.

## Trình tự tối thiểu nếu người dùng duyệt

1. Làm cache truy hồi nhỏ trên TRAIN và kiểm tra ngữ cảnh đúng ảnh; không cần ngay ViT-H/14/FACTUAL/full-bank. Dùng đúng cặp CLIP image/text encoder cùng không gian và revision, không ghép vector các backbone khác nhau. Cụm từ nhiều từ phải giữ như một đơn vị; không tách hết rồi mất lợi ích định thử.
2. Datastore chỉ từ Karpathy TRAIN của dự án, loại mọi caption cùng imageID khi query TRAIN. Không đưa VALID/TEST captions vào kho; caption tham chiếu của query không dùng chọn ngữ cảnh hoặc inference. Lưu nguồn imageID để audit. Không hard-delete object vì thiếu detector: có thể bỏ sót/topK.
3. Matched continuation: CSA hiện tại vs CSA + ngữ cảnh truy hồi, cùng khởi tạo chung, cohort/thứ tự/update/giải mã. Đánh giá output decoder trước rerank và cùng fixed rerank sau đó; trước hết dùng VALID, không tuning trên TEST166. Ngữ cảnh rỗng phải có fallback giữ đường ảnh. Cache/fusion tăng latency cần đo thực tế; chưa có dự toán GPU mới.
4. Nếu có tín hiệu jointCIDEr/CHAIR/recall mới thử ITM **lọc nguồn truy hồi**, vì Tâm báo +1.90 nhưng đây không phải lợi ích đã chứng minh cho CSA. Không ưu tiên thêm Fact Memory (kết quả âm) hoặc Context Memory (+.23 nhưng CHAIR xấu hơn).

Khuyến nghị: giữ nguyên kết quả117.06 đã chốt để viết bài, xem retrieval là phần mở rộng riêng có ranh giới và đối chứng. Chưa thay notebook, weights, hệ số hoặc chạy mô hình. Chưa biết retrieval có tăng điểm trên CSA hay không.
