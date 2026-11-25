# Beam SCST một epoch — seed 43, output v199

Ngày kiểm tra: 26/11/2026. Nguồn người dùng xác nhận: `ducanh3/notebook7bc83c4c76/199`.

**Hoàn tất một epoch và VALID5000: CIDEr tăng rõ, nhưng ảo giác đối tượng cũng tăng.** Đây là kết quả chưa rerank, không phải TEST.

| Thang đo | CE seed 43, v179 | Sau beam SCST | Chênh lệch |
|---|---:|---:|---:|
| CIDEr | 121.7778 | 129.3895 | +7.6117 |
| BLEU_1 | 78.0365 | 80.5946 | +2.5580 |
| BLEU_4 | 38.0981 | 40.3305 | +2.2325 |
| ROUGE_L | 58.1284 | 59.7202 | +1.5918 |
| CHAIRs | 4.1800 | 5.0800 | +0.9000 |
| CHAIRi | 3.0048 | 3.3759 | +0.3710 |
| Recall | 42.4787 | 45.7559 | +3.2772 |
| NLL | 2.1553 | 2.9461 | +0.7908 |

Điểm caption nhân 100; CHAIR/Recall là phần trăm, chênh lệch là điểm phần trăm. NLL không nhân 100. METEOR và SPICE chưa được chấm ở lượt này.

CIDEr tăng 7,6117 điểm; BLEU-4 tăng 2,2325 điểm. CHAIRs tăng 0,90 điểm phần trăm, tương ứng 209 → 254 caption bị gắn cờ (+45, khoảng 21,53% tương đối). CHAIRi tăng 0,3710 điểm phần trăm. Recall tăng 3,2772 điểm phần trăm nhưng không bù cho việc tăng CHAIR trong mục tiêu giảm ảo giác. NLL tăng 2,1553 → 2,9461 là sự đánh đổi với độ khớp caption tham chiếu; riêng con số này không chứng minh lỗi huấn luyện.

Huấn luyện hoàn tất 113.287 ảnh TRAIN / 3.541 update, batch cuối 7 ảnh. Thời gian worker huấn luyện 11.249,12 giây (3 giờ 7 phút 29 giây), không gồm setup, chấm VALID hoặc thời gian tính quota. Peak allocated được báo cáo 5,08 GiB; không phải toàn bộ bộ nhớ hệ thống/GPU.

Khởi đầu Full seed43 best CE epoch8 của v179, SHA `a51e7ec21be68aa1cda9e0805bd6248add6199914a5c262fce138988066051a5`. Cơ chế beam5/train mode, baseline trung bình reward của năm beam cùng ảnh, pure SCST/CIDEr-D; mô hình căn chỉnh giữ nguyên. Dùng gradient accumulation local4×4 trên hai GPU để đạt global32. Preflight trên Kaggle báo PASS beam/EOS/autograd/tail7/core và hai prompt adapter; sáu update kiểm tra bị bỏ, không khởi tạo huấn luyện thật.

## Checkpoint cần giữ

- `pilot_checkpoint.bin`: endpoint để sinh caption, SHA `3b03acc29f025b07dc358846b30618ae9c83f8634bc12bf7905e9d908b971f24`.
- `checkpoint.bin`: trạng thái optimizer/RNG/cursor để resume, SHA `5b8d33fc0b26f2c77ad1cee4a58d3e4a718f13107a2d2bd9a0884c427faed88c`.
- Giữ toàn bộ Output v199; review ZIP không chứa các file trọng số. Hai SHA khác nhau là hai dạng đóng gói, không phải hai mốc update khác nhau.

## Kiểm tra và giới hạn

Đối chiếu 38 source SHA với manifest V2; hai rank cùng identity; 142 dòng log tăng đều tới endpoint; receipt3541/113287; source179; các seal caption/reference/scoring; 5.000 ID duy nhất và đủ hai rank; 25.010 caption tham chiếu (4.990 ảnh có5,10 ảnh có6); CIDEr trung bình từng ảnh; NLL từ tổng loss/token hai worker; CHAIR từ caption và số đếm gốc đều khớp. CHAIR:254 caption/262 nhắc sai/7.761 nhắc đối tượng; recall7.288/15.928.

Agent chỉ đọc cache text/metadata và tính lại phép tổng hợp, không chạy Torch, mô hình, scorer, Java hoặc Kaggle. SHA trọng số là receipt của lượt người dùng chạy, không phải kiểm tra lại binary local. Feature cache được kiểm tra header/size/map ở runtime, không hash lại toàn bộ nội dung patch.

Một seed, một epoch và VALID tái sử dụng chưa đủ kết luận khái quát hay kết quả TEST. Mức tăng này thuộc cả recipe mới; so với SCST cũ còn khác thời lượng, optimizer rate và policy mode, nên chưa tách riêng đóng góp của beam baseline. CHAIR chỉ phản ánh đối tượng, không đo đầy đủ lỗi thuộc tính/hành động/quan hệ.

**Khuyến nghị:** giữ checkpoint này như ứng viên tăng chất lượng. Bước có ích tiếp theo là kiểm tra rerank cố định hiện có trên VALID v199 để xem giảm CHAIR được bao nhiêu trong khi giữ CIDEr. Chưa tự động học epoch2, chấm TEST hay đổi trọng số rerank.

Sheet: `Kết quả mới!A442:E459`, đã ghi và đọc lại đủ90 ô; đã kiểm tra bố cục Google Sheets và tăng chiều cao riêng các hàng mới để chữ không bị cắt.
