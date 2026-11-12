# Bản thảo phương pháp và kết quả

Mô hình nhận đặc trưng patch CLIP của ảnh và các prompt đối tượng–quan hệ do UPT tạo cho từng ảnh. Mỗi đối tượng được gắn với một vùng; các quan hệ giữ chỉ mục của hai đối tượng đầu cuối. Tọa độ ROI chuẩn hóa được chiếu lên lưới patch 14 × 14 theo phần diện tích giao nhau. Bộ mã hóa kết hợp bank ngữ nghĩa với đặc trưng ảnh và bổ sung prompt gắn vùng trước khi bộ giải mã Transformer sinh câu. Khi suy luận, beam search tạo năm ứng viên. Bộ rerank chọn câu theo tổng có trọng số của điểm decoder, CLIPScore và nhất quán đối tượng (0,3 / 0,4 / 0,3).

Mô hình được huấn luyện bằng cross-entropy trong 10 epoch, với batch toàn cục 32 và 177.110 update. Checkpoint epoch 9 được chọn theo CIDEr trên VALID. Trên MS COCO Karpathy TEST 5.000 ảnh, CLIP patch baseline, CSA top-1 và CSA + rerank lần lượt đạt CIDEr 120,51, 121,64 và 125,23; CHAIRs tương ứng là 5,16%, 3,96% và 3,32%. Khi cố định checkpoint và bộ ứng viên, rerank tăng CIDEr 3,59 điểm (KTC 95% [2,37; 4,82]) và giảm CHAIRs 0,64 điểm phần trăm (KTC 95% [−1,04; −0,24]). Khoảng tin cậy dùng 10.000 lượt lấy mẫu lại theo cặp ảnh. So sánh với baseline phản ánh mức cải thiện của cả hệ thống. Kết quả đến từ một seed huấn luyện.

## Ví dụ định tính

| COCO ID | Prompt và vùng đã ghi nhận | CSA top-1 | CSA + rerank | Cờ CHAIR |
|---|---|---|---|---|
| 157102 | Person và scissors, với hai quan hệ holding và cutting with. | a little boy that is holding a toothbrush | a little boy that is sitting at a table | Bỏ cờ toothbrush. Câu mới lược mất hành động với kéo trong caption tham chiếu. |
| 165029 | Person và snowboard được liên kết bằng quan hệ holding. | a man and a woman standing next to each other | a man and a woman holding a skateboard | Phát sinh cờ skateboard. Caption tham chiếu gọi vật đó là snowboard. |

Hai ca được chọn để minh họa một lần bỏ cờ và một lần phát sinh cờ, không dùng để ước lượng tần suất. Tọa độ ROI và chỉ mục hai đầu quan hệ được lưu trong CSA_TEST_QUALITATIVE_CASES.json. Cần đối chiếu ảnh gốc trước khi dùng hình minh họa cuối cùng. Hai ví dụ mô tả đầu vào của mô hình và kết quả chọn câu; tác động riêng của prompt gắn vùng cần một đối chứng có kiểm soát. CHAIR đo ảo giác đối tượng và không chấm độ chính xác của quan hệ.

Phân tích bổ sung: ngưỡng rerank 0,06 tăng 0,11 CIDEr trên TEST và giảm thêm một caption có cờ, đồng thời Recall giảm 0,10 điểm phần trăm. Pilot SCST + XE 500 update giảm 0,78 CIDEr top-1 trên VALID và giảm CHAIRs 0,12 điểm phần trăm. Giữ cấu hình rerank cố định làm kết quả chính.
