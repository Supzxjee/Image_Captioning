# M2-only hoàn tất CE10 — 24/11/2026

Nguồn hoàn tất: **ducanh2403/2-vi-cai-kia-qua-nhieu/245**, tiếp tục đúng latest v243. Đủ177.110update/5.667.470captionevents/10epoch; best **epoch9** theo maxVALID CIDEr. [Số liệu và kiểm tra](FULL_XE10_RESULT.json). Không SCST/rerank/TEST.

## Bảng thành phần trên VALID5000

Các điểm caption nhân100; CHAIR/Recall là phần trăm. Mỗi cấu hình dùng checkpoint được chọn bằng cùng quy tắc maxVALID CIDEr sau lịch10epoch. METEOR/SPICE chưa được đo ở lượt này.

| Thang đo | Baseline cùng decoder | M2-only | M2 + M3 zero-text | M2 + M3 prompt đúng |
|---|---:|---:|---:|---:|
| Best epoch | 9 | 9 | 8 | 9 |
| Tham số (triệu) | 29,55 | 49,24 | 53,18 | 53,18 |
| BLEU-1 | 77,87 | 77,76 | 77,79 | **77,92** |
| BLEU-4 | 37,89 | 37,49 | 37,88 | **37,94** |
| ROUGE-L | **57,90** | 57,84 | 57,72 | 57,83 |
| CIDEr | 119,14 | 119,93 | **120,30** | 120,30 |
| CHAIRs (%) | 5,70 | 5,24 | 4,96 | **4,00** |
| CHAIRi (%) | 3,99 | 3,75 | 3,50 | **2,79** |
| Recall (%) | 42,64 | 42,33 | 42,47 | **42,70** |
| NLL | 2,19418 | 2,17416 | **2,16391** | 2,16421 |

CIDEr zero-text120,304444 và full120,295535 cùng làm tròn120,30; chênh−0,008909 không được mô tả là tăngCIDEr hoặc bằng nhau về thống kê.

## Diễn giải theo từng phép so sánh

**Baseline → M2-only:** CIDEr tăng0,7856 điểm; CHAIRs giảm0,46 điểm phần trăm, CHAIRi giảm0,2481điểm phần trăm. Caption có đối tượng ảo285→262 (giảm ròng23), lượt nhắc đối tượng ảo293→273. Tuy nhiên BLEU-4 giảm0,4008điểm, Recall giảm0,3014điểm phần trăm; không kết luận cải thiện đồng loạt. Đây là bằng chứng quan sát cho toàn bộ bổ sung bank+căn chỉnh M2, có tăng dung lượng tham số.

**M2-only → M3 zero-text:** CIDEr tăng0,3784; CHAIRs giảm0,28điểm phần trăm. Cặp này thêm cả adapter M3 và tín hiệu ROI/hình học, nên không gọi là riêng tác dụng ROI hoặc riêng cấu trúc adapter.

**Zero-text → prompt đúng ảnh:** CIDEr gần bằng nhau về số điểm quan sát, CHAIRs giảm0,96điểm phần trăm (4,96%→4,00%), CHAIRi giảm0,7042điểm phần trăm. Giữ adapter/ROI/hình học và cùng số tham số khai báo giúp phép so sánh này tập trung vào chữ prompt M3. Kết quả ủng hộ mục tiêu giảm ảo giác đối tượng; chưa phải phép thử tương đương hoặc ý nghĩa thống kê. Đối chứng TEST đã có cho cặp zero-text/full trước đó; **M2-only hiện mới có VALID**.

**M2-only → full prompt:** CIDEr tăng0,3695; CHAIRs giảm1,24điểm phần trăm (5,24%→4,00%, giảm23,66% tương đối), CHAIRi giảm0,9538điểm phần trăm. Caption có đối tượng ảo262→200. Đây là đóng góp gộp của M3 (adapter+vùng/hình học+chữ), không phải chỉ chữ.

Tại cùng endpoint10, M2-only CIDEr119,54/CHAIRs4,68%/CHAIRi3,37%. Checkpoint9 vẫn được giữ cho bước tiếp theo vì quy tắc chọn là CIDEr VALID; không đổi sang epoch10 vì thấy CHAIR thấp hơn. Các đối chiếu endpoint10 được giữ đầy đủ trong JSON, không thay bảng primary bằng checkpoint chọn sau khi xem hallucination.

## Kiểm tra và giới hạn

ZIP138files, SHA256 `aa7143617080860b59f997a6b9eff9b660f5f26ff3c9a30c7bd1e2079780e5db`, CRC/source42SHA/evaluationseals khớp. Identity, tám history cũ và byteprefix1.480dòngtraininglog v243 được giữ nguyên; đủ1.780dònglog tới update177110, LR cuối1e-5. Kiểm50.000captionVALID/10epoch,10.000captionCHAIR ở best9/end10, các tổngCIDEr/NLL/meanwords và đếmCHAIR/Recall khớp. APIresume đúngv243 được ghi trongreceipt; 42nguồn training không thay đổi. Không nạp checkpoint hoặc chạyTorch/model/scorer local.

Best9 SHA `3fe31a921742e841c0d3863aac9ab9f00b0cd6b46289f3cf36029ac1fa168226`; final10 SHA `d26af3af3fb8be53920f13ac062e1d3274b02afd3c5b67ef2cec6c5041472c8b`. Binary không nằm trongreviewZIP nên các SHA này là receipt, phải kiểm lại binary trước inference. Resume worker7654,85giây (khoảng2giờ08phút) chỉ là worker tiếp tục, không gồm run1/setup/tải/checkCPUquality hoặc quota tính phí.

Mộtseed cho baseline/M2-only/zero-text, reusedVALID dùng chọncheckpoint; không coi là TEST độc lập hoặc kết luận thống kê. M2-only49.239.288tham số so với baseline29.553.424 và full53.183.736; chưa đối chứng dung lượng bằng nhau. Zero-input có cột trọng số chữ không hoạt động. Historicalfullseed42 khác I/Ostaging; CHAIR chỉ kiểm tra đối tượng, không chứng minh tính đúng của thuộc tính/hành động/quan hệ. Chuỗi so sánh có thể có tương tác; không coi mức tăng là các đóng góp tuyến tính độc lập.

**Khuyến nghị tiếp theo:** chấm TEST5000 của đúng best9v245 với cùng beam/scorer trướcSCST/rerank, để hoàn tất bảng thành phần trên cùngTEST. Chưa tự chuẩn bị/chạy TEST, train thêm, thay kiến trúc hoặc sửaWord từ lần nhậnZIP này.
