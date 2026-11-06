# Kết quả đầy đủ trên caption VALID5000 đã lưu

06/11/2026. [Audit32checks và số liệu](PROMPT_SCRATCH_QUALITY_RESULT.json). ZIP SHA `b92492b79ce3f782e827db99d6d1b604942ef1ddc09f548876026220feb71510`. Nguồn weights không đổi: CONTROL178/PROMPT58. Không hỏi lại sốversion CPU vì toàn bộ số liệu/caption/provenance cần thiết đã có trongZIP và payload Git.

**Cập nhật kết luận từ lượt train:** trước đó chỉ thấy PROMPT không tăng CIDEr. CHAIR mới bổ sung bằng chứng rằng nhánh PROMPT giảm ảo giác đối tượng trên VALID5000, với recall tăng nhẹ. Đây là đánh đổi chất lượng, chưa phải cải thiện đồng thời mọi độ đo.

| Độ đo | CONTROL10 (best) | PROMPT9 (best đã chọn) | PROMPT10 (endpoint) |
|---|---:|---:|---:|
| CIDEr ×100 ↑ | 99,4587 | 98,8453 | 97,9000 |
| BLEU-4 ×100 ↑ | 32,9878 | 32,3083 | 32,1692 |
| ROUGE-L ×100 ↑ | 53,2696 | 52,9931 | 53,0370 |
| CHAIRs % ↓ | 6,60 | 4,40 | 3,78 |
| CHAIRi % ↓ | 4,8236 | 3,1451 | 2,7585 |
| Object recall % ↑ | 39,7664 | 40,5387 | 40,0239 |
| Caption bị CHAIR đánh dấu ↓ | 330 | 220 | 189 |
| Mention bị đánh dấu ↓ | 346 | 230 | 199 |
| Tổng object mentions | 7.173 | 7.313 | 7.214 |
| Số cặp ảnh–lớp đối tượng đúng được nhắc | 6.334 | 6.457 | 6.375 |
| Số cặp ảnh–lớp đối tượng tham chiếu | 15.928 | 15.928 | 15.928 |
| Số từ/caption | 8,1054 | 7,7278 | 7,7134 |
| NLL VALID (từ lượttrain) ↓ | 2,448211 | 2,516285 | 2,502108 |

## So sánh chính: best9 so với best10

Hai checkpoint được chọn bằng maxCIDEr trong10epoch **trước khi xemCHAIR**. Giữ CONTROL10 vàPROMPT9 làm so sánh chính, không đổi lựa chọn sangPROMPT10 vìCHAIR thấp hơn.

PROMPT9: ít hơn110captionhalu (−33,33% tương đối), CHAIRs−2,20điểm phần trăm, CHAIRi−1,6786điểm phần trăm; recall+0,7722điểm phần trăm. Sửa212caption màCONTROL bịCHAIR đánh dấu, đồng thời phát sinh102caption mới bị đánh dấu; net−110. Có4.206caption khác nhau.

CIDEr−0,6134điểm; paired95%CI trên chênh CIDEr×100 là[−2,2165;0,9653]. Khoảng có0: chưa thấy chênh có dấu cố định trong bootstrap này, **không có nghĩa đã chứng minh tương đương hoặc không suy giảm**. CHAIRs paired95%CI[−2,94;−1,5595]điểm phần trăm. Đây là2000bootstrap trên mộtseed, corpusIDF và checkpoint đã cố định, trên VALID đã dùng nhiều lần; không phải biến thiên giữa các lầntrain hoặc TEST độc lập.

Caption ngắn hơn nhưng objectmentions tăng140 và số cặp ảnh–lớp đối tượng đúng tăng123. Do đó không thể giải thích toàn bộ lợi ích chỉ bằng việc bớt nhắc đối tượng. Đây là kết luận theo CHAIREvaluator/COCOaliases/GT cache, chưa phải người đánh giá từng ảnh.

## So sánh phụ: cùng endpoint10

PROMPT10 ít hơn141captionhalu (−42,73% tương đối), CHAIRs−2,82điểm phần trăm, recall+0,2574điểm phần trăm. Sửa227/phát sinh86; net−141;3.817caption đổi. CIDEr−1,5587điểm, paired95%CI[−2,9707;−0,0761], cho thấy đánh đổi CIDEr rõ hơn trong đánh giá cố định này. Không dùng endpoint như checkpoint được chọn sau khi thấyCHAIR.

## Kiểm tra và giới hạn

32checks metadata/source6pins17bf52c/replay/captioninput/CHAIRcounts/per-imagealignment/pairedstatisticsPASS. CIDEr cả3mô hình khớp số đã lưu từtrain; replay_mismatches=[]; đủ5000ID/25010PTBrefs. Provenancepycocoevalcap1.2/nltk3.9.2/numpy2.1.3, CHAIRcacheSHA73ce1122. Caption đầu vàoCHAIR trùng sealedpayload, counts khớp overallmetrics, per-imageCIDEr trung bình khớp corpusCIDEr; pairedCI vàrepair/introduction được tác giả kiểm lại bằngNumPy trên số liệu đã có.

Tác giả **không chạy scorer/model/checkpoint/inference/GPU**; evaluator do người dùng chạyCPU, không sinh caption mới hoặc dùngTEST. Không có bằng chứng quan hệ/thuộc tính đúng hơn chỉ từCHAIR. Chưa kiểm textzero/shuffle/off: lợi ích của nội dung ngữ nghĩa riêng với lợi ích adapter/capacity/ROI vẫn chưa tách được. Prompt thêm4.468.736tham số, không phải đối chứng tham số khớp. Không ghi là đã tăngCIDEr hay đã chứng minh đóng góp căn chỉnh nhân quả.

## Khuyến nghị

Giữ PROMPT9 theo quy tắc đã khóa và giữ endpoint10 để báo cáo đánh đổi. Chưa tự mở rộngXE/SCST hoặc chạyTEST. Bước nhỏ hợp lý tiếp theo là kiểm tra ảnh hưởng của textzero/shuffle/off trên tập VALID cố định có phân tầng relation/object/empty, để biết mô hình có thực sự dùng nội dung prompt trước khi chốt mô hình và đánh giáTEST. Bước này cần người dùng chốt phạm vi/ngân sách trước; chưa có notebookGPU mới và chưa hỏi thêmquota trong lượt review.

CachedZIPtexts: `.review_cache/prompt_scratch_quality_valid5000/archive_text.json` ngoài repo. Không cần đọcZIP lại khi tiếp tục từ báo cáo này.
