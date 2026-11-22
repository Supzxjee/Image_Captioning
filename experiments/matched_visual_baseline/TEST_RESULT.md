# Baseline cùng decoder — TEST5000 hoàn tất

22/11/2026. Người dùng xác nhận output TEST **ducanh3/notebook7bc83c4c76/189**; checkpoint giữ nguyênv188/best9/SHA1eade1fc… . [Số liệu/audit](TEST_RESULT.json), [protocol trước chạy](TEST_PROTOCOL.md). Chưa sửa báo cáo Word hoặc chạy mô hình/scorer local.

## So sánh chính

Cả hai cùng seed42, lịch10epochCE, bestepoch9 chọn bằngVALID, cùng decoder parallel3lớp/dropout0.2. Dùng5000KarpathyTEST, beam5/max20, top1 trướcSCST/rerank. Điểm nhân100, CHAIR/Recall là phần trăm; deltaCHAIR là điểm phần trăm.

| Thang đo | Baseline thị giác | Bank + prompt | Full − baseline |
|---|---:|---:|---:|
| BLEU-1 | 77.7462 | 77.6252 | -0.1210 |
| BLEU-4 | 37.7282 | 37.7415 | +0.0133 |
| METEOR | 28.4455 | 28.8290 | +0.3835 |
| ROUGE-L | 57.9251 | 57.9911 | +0.0660 |
| CIDEr | 119.9613 | 121.6375 | +1.6762 |
| SPICE | 21.3792 | 21.8527 | +0.4735 |
| CHAIRs (%) | 5.2200 | 3.9600 | -1.2600 |
| CHAIRi (%) | 3.6854 | 2.8017 | -0.8837 |
| Recall (%) | 42.9696 | 43.4904 | +0.5208 |
| NLL | 2.214722 | 2.182139 | -0.032584 |

CIDEr tăng1.6762điểm, SPICE tăng0.4735, METEOR tăng0.3835; BLEU4 gần như không đổi và BLEU1 giảm0.1210. Không mô tả là mọi thang đo đều cải thiện.

CHAIRs5.22→3.96%, giảm1.26pp/tương đối24.14%; caption có đối tượnghallucination261→198, giảm ròng63. Hallucinatedmentions269→205; CHAIRi3.6854→2.8017%, giảm0.8837pp/tương đối23.98%. Objectmentions7299→7317, recalledunique6766→6848 với cùng15746referenceunique; Recall tăng0.5208pp. Meanwords baseline9.1536→full9.2696. Mức giảm ảo giác quan sát không đi kèm giảm recall hoặc giảm tổng số lượt nhắc đối tượng.

Full + rerank cố định đã có: CIDEr125.2307/CHAIRs3.32%/CHAIRi2.2655%. Đây là cấu hình hậu xử lý riêng; không dùng mức tăng so baseline đó làm đóng góp riêng của học căn chỉnh. Bảng chính trên giữ cả hai trước rerank.

## Kiểm tra thực tế và nguồn

- Exactbest9v188 tải bằng API theo runtime receipt; model strict state/first GPU beam pathPASS trên haiGPU Torch2.11.0+cu128, không đổi CUDA build. Không training/rerank/tuning trong lượtTEST.
- ZIPCRC/SHA72files;32sourceSHA khớp notebook, worker/scoring seals khớp.5000top1 khớp beam đứngđầu, đủ25000candidates/token→text/scoreorder.25010referencecaption events, cohort vàdecodepolicy giữ nguyên. Evaluatorfiles/JRE/CoreNLPdigests trùng fullseed42.
- Số đếmCHAIR5000caption, CIDEr/SPICE per-image means vàNLL theo token tái tổng hợp đạt. METEOR và các corpusn-gram dùng output scorer đã lưu; không chạy lại scorer local hoặc lấy trung bìnhMETEORper-image để thay corpusmetric.
- StageTEST536.31s; worker lâu nhất175.62s gồmdecode+teacherNLL. Không cộngnhầm thànhGPUbilling hoặc runtime toànnotebook. Đã giữcaption/5beam/per-imagemetrics để tái sử dụng; binarycheckpoint không nằm trongreviewZIP nên chưa load/hash lại local.

## Diễn giải

Kết quả TEST cùng chiều vớiVALID: whole bank+prompt thêm chất lượngcaption và giảmobjecthallucination so baseline cùngdecoder. Đây là đối chứng tốt hơn baseline lịch sửkhácdecoder, nhưng không đủ tách ngữ nghĩa khỏicapacity hoặc riêngM3: full53.18triệu so baseline29.55triệu parameters; mới một matchedbaseline seed. Các fullseed43/44 đã có không thay thế baseline43/44.

Baseline bốn lớp lịch sử CIDEr120.5145 vẫn được giữ làm mốc hệ thống khác; baseline mới119.9613 không thay thế hoặc xóa điểm cũ. Chọn baseline này vì cùngdecoder theo quyết định trước chạy, không chọn vì nó thấp hơn sau khi xemTEST. Fullseed42 dùnglocalFP16stage lịch sử, baseline mới trainH5trực tiếp; không cam kếtbitwise trajectories. TEST đã từng được dùng trong quá trình phát triển dự án; không gọi là untouchedholdout. Không kiểm địnhsignificance trong lầnreviewnày. CHAIR không chứng minh mô tả hành động/quan hệ đúng.

## Lưu kết quả

Sheet **Kết quả mới!A291:D309** đã ghi/đọc lại, kiểmmetadata định dạng; chưa xemhìnhnative. Dùng cặp baseline/fullcùngdecoder trướcSCST/rerank cho bảng so sánh chính; giữ các cấu hìnhrerank/SCST riêng. Có thể bắt đầu viết kết quả bài báo; đối chứng textM3 và matchedmultiseed là phần cần bổ sung nếu muốn khẳng định đóng góp riêng/độổnđịnh, không tự mở thêm training bằng việc nhậnZIP này.
