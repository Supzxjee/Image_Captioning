# SCST epoch 2 — VALID/TEST + fixed rerank, output v202

Đã kiểm tra ngày 26/11/2026. Người dùng xác nhận nguồn `ducanh3/notebook7bc83c4c76/202`. Lượt học và đánh giá đã hoàn tất; CIDEr tăng tiếp, đồng thời CHAIR tăng. Không kết luận cải thiện mọi mặt.

## Huấn luyện và nguồn trọng số

- Khởi đầu đúng full checkpoint v199 SHA `5b8d33fc0b26f2c77ad1cee4a58d3e4a718f13107a2d2bd9a0884c427faed88c`; hai rank khôi phục Adam180 state, mọi step3541, RNG, epoch2cursor0. Không khởi tạo mới Adam hoặc dùng trạng thái preflight.
- Học thêm đủ113.287 ảnh theo cache gốc, tổng7.082 update/226.574 image events cho hai epoch; tail7 hợp lệ. Model53.183.736 tham số, M2/M3 giữ nguyên, SCST thuần/beam5/FP32/global32 và hai tối ưu đã kiểm tra.
- Endpoint export SHA `5373f5800109e8d24d56968a222f1b9424ceb119497a8e530212f76026de37cb`; full checkpoint học tiếp SHA `f21a72b23c5f138550840a4065e9f94a94ff36042b8eaad9d87930d0124529e4`. Giữ toàn bộ Output v202; review ZIP không chứa các binary này.
- Train worker11.405,21 giây ≈3giờ10phút05giây, chưa gồm setup/VALID/TEST/scorer/quota. Epoch1 worker11.249,12giây; lượt mới dài hơn khoảng156giây. Hai epoch có chính sách/caption khác nhau, không phải benchmark cùng công việc; không dùng cặp thời gian này để xác nhận hay bác bỏ mức tăng tốc pilot.

## TEST5000

Cùng seed43, cohort/reference bytes, evaluator assets và fixed selector. Caption metrics nhân100; CHAIR là phần trăm. Checkpoint cuối epoch2 được định trước, không chọn dựa vào TEST.

### Top1 beam search, chưa rerank

| Thang đo | CE179 | SCST epoch1 | SCST epoch2 |
|---|---:|---:|---:|
| BLEU_1 | 77,76 | 80,20 | 81,44 |
| BLEU_4 | 37,79 | 40,26 | 40,36 |
| METEOR | 28,81 | 28,82 | 29,12 |
| ROUGE_L | 57,91 | 59,20 | 59,35 |
| CIDEr | 121,98 | 129,61 | 132,55 |
| SPICE | 21,83 | 22,66 | 23,06 |
| CHAIRs | 4,06 | 5,06 | 5,42 |
| CHAIRi | 2,80 | 3,34 | 3,56 |
| Recall (%) | 43,71 | 46,60 | 47,26 |
| NLL | 2,1781 | 2,9789 | 4,0050 |

### Rerank cố định0,3decoder +0,4CLIP +0,3object

| Thang đo | CE179 | SCST epoch1 | SCST epoch2 |
|---|---:|---:|---:|
| BLEU_1 | 77,91 | 80,87 | 82,06 |
| BLEU_4 | 37,42 | 39,62 | 40,04 |
| METEOR | 29,41 | 29,17 | 29,34 |
| ROUGE_L | 58,25 | 58,99 | 59,48 |
| CIDEr | 124,36 | 130,18 | 132,85 |
| SPICE | 22,57 | 23,25 | 23,55 |
| CHAIRs | 3,66 | 3,68 | 4,04 |
| CHAIRi | 2,45 | 2,48 | 2,69 |
| Recall (%) | 44,93 | 46,88 | 47,33 |
| NLL | 2,1781 | 2,9789 | 4,0050 |

## Diễn giải

**Epoch2 so với epoch1:** top1 CIDEr129,6063→132,5474 (+2,9411); sau rerank130,1785→132,8530 (+2,6745). Cả sáu điểm chất lượng chính đều tăng theo điểm quan sát ở cả hai policy. Tuy nhiên CHAIRs top1 5,06%→5,42%, rerank3,68%→4,04% (cùng tăng0,36điểm phần trăm, thêm18caption bị gắn cờ). CHAIRi sau rerank2,4847%→2,6880% (+0,2033điểm phần trăm). Không gọi epoch2 là tốt hơn toàn diện hoặc giữ nguyên hallucination.

**Rerank trong epoch2:** CIDEr132,5474→132,8530 (+0,3056), CHAIRs5,42%→4,04% (−1,38điểm phần trăm), CHAIRi3,5580%→2,6880%. Caption bị cờ271→202; 94hết cờ,25cờ mới, giảm ròng69. Có2.833caption đổi hạng. Đây là thay đổi do scorer xác định, chưa phải94lỗi được người chấm xác nhận đã sửa. BLEU4 giảm40,3559→40,0413; không mô tả rerank tăng mọi chỉ số.

**So với CE cùng rerank:** CIDEr124,3639→132,8530 (+8,4891), nhưng CHAIRs3,66%→4,04% (+0,38điểm phần trăm), CHAIRi2,4509%→2,6880%; METEOR29,4109→29,3439 giảm nhẹ. CE+rerank vẫn là mốc ít ảo giác hơn. Không ghép baseline seed42/decoder khác vào phép so sánh này.

**NLL tăng:** TEST2,9789→4,0050, VALID2,9461→3,9777. Điều này cho thấy khả năng gán xác suất cho caption tham chiếu theo teacher forcing kém đi, dù CIDEr của câu tự sinh tăng. Phù hợp với đánh đổi có thể xảy ra khi tối ưu reward; chưa xác định nguyên nhân cụ thể hoặc khẳng định không có overfitting.

## VALID5000

Top1 CIDEr129,3895→133,1704 (+3,7809), BLEU4 40,3305→41,0269. CHAIRs5,08%→5,60%, CHAIRi3,3759%→3,6454%; NLL tăng như trên. VALID cũng cho cùng chiều đánh đổi, không chỉ xuất hiện ở TEST. Không có VALID rerank trong lượt này.

## Kiểm tra và giới hạn

180ZIPfiles đọc một lần và lưu cache; SHA ZIP `f58a78380edb6749a327ed4d42ba90413de5eba86e9bb79e2f4165e05c942fb0`. 80sourceSHA khớp notebook; hai runtime input seals khớp; identity/parent/Adam/RNG hai rank,143log rows và full endpoint verified. Đủ5.000VALID +5.000TEST không giao nhau;25.000TESTbeam/CLIPpairs và replay5.000lựa chọn đều khớp. CIDErmean, CHAIRcounts, Recall và NLL được tính lại từ số đã lưu; METEORscalar/per-image và basic-score artifacts khớp. SPICEaggregate có trong output hoàn tất, không có per-imageSPICE để tính lại. Bộchấm/Java/CoreNLP và CLIPassetcontract khớp epoch1. [Audit](audit_run1.py), [số liệu](RUN1_RESULT.json), [inventory](RUN1_ZIP_MANIFEST.json).

Không load checkpoint hoặc chạy Torch/mô hình/scorer local. Binary/JPEG/CLIP được xác nhận qua receipt của lượt người dùng chạy. Một seedSCST; TEST đã dùng trong phát triển; chưa kiểm định thống kê cho chênh lệch epoch1/2. CHAIR chỉ đo đối tượng, không chứng minh thuộc tính/hành động/quan hệ đúng. Lượt này không tách riêng đóng góp M3 hay cơ chế tối ưu tốc độ.

**Khuyến nghị:** giữ epoch2+rerank như ứng viên CIDEr cao hơn, giữ epoch1+rerank và CE+rerank làm mốc CHAIR thấp hơn. Báo đầy đủ đánh đổi; chưa tự học epoch3, đổi reward hay chỉnh selector dựa vào TEST.
