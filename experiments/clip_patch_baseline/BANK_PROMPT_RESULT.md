# Bank + prompt trên CLIP patch — kết quả 10 epoch XE

Lượt chạy hoàn tất, người dùng xác nhận Save Version **9**; tên notebook/tài khoản chưa được xác nhận trong câu trả lời. ZIP SHA `035c9b5b95cb9a105fd72bf20566e9ef1ef013c8a4781a9e8f44a508d13e3d72`; source `f5f6eb0`. ZIP được đọc một lần, text cache tại `../.review_cache/clip_patch_bank_prompt_xe10_v1/archive_text.json`. Audit `review_bank_prompt.py`: **1.934 kiểm tra PASS**, không tải trọng số/Torch hoặc chạy lại scorer local.

113.287 ảnh TRAIN, 566.747 caption/epoch, 10 epoch, 177.110 update, 5.667.470 lượt caption, global batch32. Khởi tạo mới, không SCST/rerank/TEST. Mã core decoder và lịch học giữ nguyên baseline199; SHA core ban đầu khớp, GPU đã qua kiểm tra khởi tạo bộ nhớ bằng baseline, mask/empty slots, gradient DDP và beam. Byte seals của dữ liệu/caption/vocabulary/VALID references và patch cache khớp baseline.

## So sánh trên cùng VALID5000

Mỗi cấu hình chọn checkpoint bằng CIDEr cao nhất trong 10 epoch theo quy tắc đã khóa. Điểm caption nhân100, CHAIR/Recall tính phần trăm; NLL không nhân100.

| Chỉ số | Baseline199, epoch10 | Bank + prompt, best epoch9 | Bank + prompt, epoch10 |
|---|---:|---:|---:|
| BLEU-1 | 77,2054 | 77,7238 | 77,6653 |
| BLEU-4 | 37,0724 | 37,4403 | 37,6069 |
| ROUGE-L | 57,3740 | 57,6802 | 57,5182 |
| CIDEr | 119,0942 | 120,5804 | 120,4308 |
| CHAIRs | 5,38% | 4,58% | 4,42% |
| CHAIRi | 3,7842% | 3,1826% | 3,0746% |
| Recall | 43,0500% | 43,6527% | 43,3137% |
| VALID NLL | 2,101613 | 2,100591 | 2,106570 |
| Caption bị CHAIR đánh dấu | 269/5000 | 229/5000 | 221/5000 |
| TRAIN online NLL tại checkpoint | 1,838866 | 1,827451 | 1,801271 |

TRAIN online NLL là trung bình trong quá trình cập nhật của epoch, không so trực tiếp với VALID NLL cố định trọng số. METEOR/SPICE và TEST chưa chấm cho cấu hình mới.

## Diễn giải

Best epoch9 tăng CIDEr **1,4862 điểm**, giảm CHAIRs **0,80 điểm phần trăm** (khoảng14,87% tương đối), CHAIRi giảm0,6016 điểm phần trăm, NLL chỉ giảm0,001023. Câu dài gần như tương đương: 9,4154→9,4262 từ, không có câu tối đa2từ; không có dấu hiệu giảm ảo giác đơn thuần bằng caption ngắn hơn trong các thống kê này. Recall tăng0,6027 điểm phần trăm là chỉ số phụ.

Đối chiếu theo ảnh:148 ảnh mất cờ CHAIR,108 ảnh mới bị gắn cờ, giảm ròng40. Đây là thay đổi cờ đối tượng theo CHAIR, không xác nhận148caption đã đúng hoàn toàn hoặc quan hệ đã chính xác. Bootstrap cặp2.000 mẫu/seed42 trên caption cố định: CI95 chênh CIDEr [-0,0504;+3,0267] điểm, CHAIRs [-1,42;-0,20] điểm phần trăm. CIDEr còn chứa0, bằng chứng chắc hơn ở giảm CHAIR; bootstrap có điều kiện trên một seed/VALID đã tái sử dụng/checkpoint được chọn, không điều chỉnh selection và không thay thế nhiều seed hoặc TEST.

Epoch10 vẫn hơn baseline tại cùng endpoint về CIDEr và CHAIR, nhưng VALID NLL xấu hơn và CIDEr thấp hơn epoch9. Giữ **epoch9 làm best**, không đổi sang epoch10 chỉ vì CHAIR tốt hơn sau khi xem kết quả.

34.800.888 tham số học so với27.461.904 baseline (+7.338.984, khoảng26,72%). Worker train+VALID+checkpoint25750,23s (7h09m10s) so với15923,06s (4h25m23s); stage patch26m58s và semantic56,53s, chưa gồm setup/scoring/billing. Build Torch cu128→cu130 khác nhau; cùng phiên bản công khai2.11.0 và core init nhưng không tuyên bố train bitwise giống nhau.

## Kết luận và bước tiếp theo

Kết quả VALID ủng hộ **thêm bank + prompt kết hợp** giúp giảm ảo giác đối tượng và có mức tăng CIDEr quan sát được. Chưa tách riêng M2/M3, ngữ nghĩa chữ, liên kết vùng hay lợi ích thêm tham số. Không so VALID120,58 với TEST120,51 của baseline để kết luận; hai tập khác nhau.

Ưu tiên bảo toàn `best.bin` epoch9 SHA `7c11be52fce9cd90dadc55fa514d4017d540f40f630232a68bfe4306c754158b` và `checkpoint.bin` epoch10 SHA `82f195d565ed628ce77c1d5d963574dc6e32d76d2336cfb3ab04f6a3099cff49`. Chọn phương pháp/checkpoint/beam cố định rồi chấm TEST5000 như baseline199, đủ8chỉ số; đây là đề xuất, chưa tạo hoặc chạy notebook mới. Không tự tăng epoch/SCST/đổi selector hoặc sửa Word.

Kết quả đã ghi vào Sheet Kết quả mới A129:L131; xác minh lại giá trị và format qua API, chưa kiểm tra giao diện trực quan.
