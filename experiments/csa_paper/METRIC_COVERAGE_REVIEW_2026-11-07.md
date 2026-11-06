# Rà soát độ phủ8chỉ số — 07/11/2026

Phạm vi: bản Word ngày06/11 và các mốc đối chứng chính trong báo cáo/kết quả CSA hiện tại. Không phải kiểm kê toàn bộ thí nghiệm lịch sử hoặc Q-Former. Chỉ đọc DOCX, RESULT.json/md và textcache; không chấm lại hoặc mở ZIP lần nữa.

## Những gì đã đo và những gì mới chỉ thiếu trên Word

CONTROL và PROMPT hiện tại, cả normal và fixed_rerank, đã đủ BLEU1/BLEU4/METEOR/ROUGE-L/CIDEr/SPICE/CHAIRs/CHAIRi trên TEST5000. Output chấm CPU185/62, caption184/61, weights180/60. Nguồn: PROMPT_TEST_EXTRA_RESULT.json.

Bản Word `docs/weekly_reports/2026-11-06/Bao_cao_nghien_cuu_06_11_2026.docx` vẫn là bản viết trước lượt chấm bổ sung: bảng TEST chỉ cóCIDEr/CHAIRs/i/recall/flags; BLEU4/ROUGE ở đoạn văn, chưa cóB1/METEOR/SPICE. Với4cấu hình chính chỉ cần cập nhật bảng báo cáo, không chạy lại.

## Các mốc đối chứng TEST còn thiếu kết quả

| Cấu hình | Cohort | Đã lưu trong8chỉ số | Còn thiếu |
|---|---|---|---|
| NoCSA, XE10 | TEST5000 | BLEU4, ROUGE-L, CIDEr, CHAIRs/i | BLEU1, METEOR, SPICE |
| CSA, XE10 | TEST5000 | BLEU4, ROUGE-L, CIDEr, CHAIRs/i | BLEU1, METEOR, SPICE |
| CSA, XE15 + SCST/CE, trước rerank | TEST5000 | BLEU4, ROUGE-L, CIDEr, CHAIRs/i | BLEU1, METEOR, SPICE |
| Cùng CSA15+SCST/CE, sau rerank | TEST5000 | BLEU4, ROUGE-L, CIDEr, CHAIRs/i | BLEU1, METEOR, SPICE |

Nguồn: PLAIN_VS_CSA10_TEST_RESULT.json (`comparison/plain`, `comparison/csa`); RERANK_TEST_RESULT.json (`metrics/decoder_rank0`, `metrics/clip_object_fixed`). Mốc117.0633CIDEr là dòng thứ4, không phải PROMPT mới.

Textcache local xác nhận có5000caption độc nhất và5000metadata córefs cho mỗiNoCSA10/CSA10 (`review/plain_self_test_loaded.json`, `review/csa_epoch10_karpathy_test_loaded.json`: test_chair_input.json/test_metadata.json). Cache TEST166 có5000caption mỗi policy và5000references tại `rerank_test_v2_review_text`. Vì thế có thể chuẩn bị chấm bổ sung từ câu đã lưu, không cần train/inference. Bước này chưa được chuẩn bị/chạy trong lượt rà soát.

## Những bảng phụ chưa đủ nhưng chưa cần ưu tiên

- Zero/shuffle của PROMPT trên TEST5000: còn thiếuBLEU1/METEOR/SPICE. Đã có5chỉ số còn lại; chỉ cần bổ sung nếu trình bày như một bảng ablation đủ8metric.
- Các mốc VALID5000 sauXE/sauSCST/trước-sau rerank: các RESULT liên quan cũng chưa lưuBLEU1/METEOR/SPICE. Giữ riêng vớiTEST; không lấy điểmTEST điền vàoVALID.
- Các hàng lịch sử CSA15+SCST không cùng lịchhuấn luyện vớiC/P hiện tại; bảng mở rộng chỉ là mốc tham khảo, không coi là đối chứng riêng tác độngprompt.

## Khuyến nghị

Ưu tiên bổ sung3metric còn thiếu choNoCSA10 vàCSA10, rồi hai policy củaCSA15+SCST nếu giữ mốc cũ trong bài. Không cần chấm lại toàn bộpilot. Cập nhật Word bằng bảng8metric hiện có là công việc dàn trang riêng; lượt này chưa sửaWord/Sheet và chưa khởi độngrun nào.
