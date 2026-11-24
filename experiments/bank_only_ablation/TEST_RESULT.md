# M2-only TEST v248 và bảng ablation bốn cấu hình — 24/11/2026

Output do người dùng xác nhận: **ducanh2403/2-vi-cai-kia-qua-nhieu/248**. Checkpoint cố định từ **v245, best epoch9**, SHA `3fe31a921742e841c0d3863aac9ab9f00b0cd6b46289f3cf36029ac1fa168226`. [Số liệu và kiểm tra](TEST_RESULT.json).

## Bảng TEST5000

Tất cả cấu hình seed42, cùng lịch10epochCE và quy tắc chọn maxVALIDCIDEr; cùng decoder parallel3lớp/dropout0,2, cohort5000TEST, beam5/max20 và bộ chấm. Các hàng M3 đều có M2; zero-text giữ adapter/ROI/hình học nhưng đặt chữ M3 bằng0 trong train và TEST. Lấy top1 trướcSCST/rerank. Điểm caption nhân100; CHAIR là phần trăm; in đậm theo giá trị thô tốt nhất.

| Thang đo | Baseline cùng decoder | M2-only | M2 + M3 zero-text | M2 + M3 prompt đúng |
|---|---:|---:|---:|---:|
| BLEU-1 | **77,75** | 77,52 | 77,49 | 77,63 |
| BLEU-4 | 37,73 | **38,01** | 37,68 | 37,74 |
| METEOR | 28,45 | **28,85** | 28,70 | 28,83 |
| ROUGE-L | 57,93 | 57,95 | 57,74 | **57,99** |
| CIDEr | 119,96 | 121,41 | 121,18 | **121,64** |
| SPICE | 21,38 | **21,91** | 21,69 | 21,85 |
| CHAIRs (%) | 5,22 | 4,70 | 4,52 | **3,96** |
| CHAIRi (%) | 3,69 | 3,29 | 3,15 | **2,80** |

Thông tin bổ sung: Recall42,9696/43,4523/43,3380/43,4904%; NLL2,214722/2,190687/2,181292/2,182139. Số caption chứa đối tượng ảo261/235/226/198, số lượt nhắc đối tượng ảo269/242/230/205. Tham số được học29.553.424/49.239.288/53.183.736/53.183.736; best epoch9/9/8/9. NLL thấp nhất thuộc zero-text; không coi mọi đại lượng đều tốt nhất ở full.

## Diễn giải

**Baseline → M2-only:** thêm bank và căn chỉnh bank–ảnh tăng1,4439CIDEr,0,2798BLEU-4,0,3995METEOR,0,5313SPICE; CHAIRs giảm0,52điểm phần trăm, CHAIRi giảm0,3907điểm phần trăm. Caption có đối tượng ảo261→235, Recall tăng0,4827điểm phần trăm. BLEU-1 giảm0,2262điểm. Đây là lợi ích quan sát của cả bổ sungM2, có thêm19.685.864tham số; không tách riêng tác dụng bank khỏi dung lượng mô hình.

**M2-only → M3 zero-text:** CIDEr giảm0,2259, BLEU-4/METEOR/SPICE cũng giảm; CHAIRs giảm0,18điểm phần trăm (235→226captionảo). Vì vậy không mô tả chuỗi thêm thành phần là tăng mọi chỉ số hoặc cho rằng adapter+ROI/hình học tự làm tăngCIDEr trênTEST. Đây cũng không phải phép so sánh chỉ riêng ROI vì adapter được thêm cùng lúc.

**Zero-text → chữ prompt đúng ảnh:** cùng53.183.736tham số khai báo và giữ adapter/ROI/hình học, CIDEr tăng0,4582, tám chỉ số chính cùng cải thiện theo hướng mong muốn; CHAIRs giảm0,56điểm phần trăm (4,52%→3,96%, giảm12,39% tương đối), captionảo226→198. NLL hơi tăng0,000847. Kết quả ủng hộ lợi ích của chữ prompt M3 đối với ảo giác đối tượng, chưa phải ý nghĩa thống kê hoặc bằng chứng đúng quan hệ/hành động.

**M2-only → full prompt:** CIDEr tăng0,2323, CHAIRs giảm0,74điểm phần trăm (4,70%→3,96%, giảm15,74% tương đối), CHAIRi giảm0,4931điểm phần trăm. Captionảo235→198, giảm ròng37; không gọi đây là37caption được sửa vì chưa đối chiếu từng trường hợp. Recall43,4523%→43,4904% gần như giữ nguyên về số điểm quan sát. Full thấp hơn M2-only0,2665BLEU-4,0,0160METEOR và0,0578SPICE, nên đóng góp nổi bật của M3 là kiểm soát ảo giác và tăng nhẹCIDEr, không phải thắng toàn bộmetric.

**Baseline → full:** CIDEr tăng1,6762 và CHAIRs giảm1,26điểm phần trăm (giảm24,14% tương đối). Không chuyển các mức tăng nối tiếp thành tỷ lệ đóng góp nhân quả độc lập: các module và huấn luyện có thể tương tác.

## Kiểm tra nguồn và giới hạn

ZIP75files, SHA256 `8ba5abcbda684ddab4387c6e75576919a7a318e8e6be61291e4bb94c57d1816c`, CRC và33sourceSHA khớp notebook đã giao. Đúngv245best9/seed42/159399updates; guard8ảnh/GPU xác nhận49.239.288tham số, khôngM3, bankfinite/L2, memoryđúngM2 và bất biến với promptgiả, memorydelta0. Đủ5000caption/25000beam; token→text/top1/logprob, cohort/reference và worker+scoringseals đều khớp. ĐếmCHAIR/Recall, trung bìnhCIDEr/SPICE và NLL có trọng số khớp; METEOR corpus không bị thay bằng trung bình per-image. Bộchấm/Java/CoreNLP có cùng digest ở cả bốn cấu hình.

Stage559,19s; worker lâu nhất237,67s gồmdecode+teacherNLL, không gồmsetup/tải/CPUscoring hoặc quota tính phí. Binarycheckpoint/patchNPY không nằm trong ZIP; xác nhận qua source/runtime/SHAreceipts, không loadbinary hoặc chạyTorch/model/scorer local. Kết quả các cấu hình trước được dùng lại từ audit đã lưu.

Mộtseed cho chuỗi ablation này, khác tổng tham số giữa baseline/M2/full, text0 có cột trọng số đầu vào không hoạt động, historicalfullseed42 khác I/Ostaging; bestepoch khác nhau theo cùng tiêu chíVALID. TEST đã được dùng trong phát triển, không gọi là finaluntouchedholdout. Chưa kiểm định ý nghĩa thống kê, equivalence/noninferiority hoặc đánh giá đúng thuộc tính/hành động/quan hệ từCHAIR. Kết quả đủ để ghi bảng thành phần trên cùngTEST; phần độổnđịnh/thống kê còn là giới hạn trước khi nâng mức khẳng định.

Giữ nguyên mô hình/checkpoint/rerank đã chốt, không tự mở thêm huấn luyện, seed, scorer hay sửaWord trong lần nhậnZIP này.
