# Bank + prompt — TEST5000, checkpoint epoch9

**Hoàn tất đủ8chỉ số trên5000KarpathyTEST.** Người dùng xác nhận lượt chấm version10, notebook kltnnnnnff theo mạch phiên hiện tại. Trọng số lấy đúng `ducanh2403/kltnnnnnff/9/best.bin`, epoch9/159399updates, SHA `7c11be52fce9cd90dadc55fa514d4017d540f40f630232a68bfe4306c754158b`; không dùng endpoint10. Source đánh giáaae9afc, ZIP SHA76346b7a718b0a0549d5c230dea59b05e7a5f2974952b967cbd99e0d76c095aa.

ZIP đọc1lần, text cache `../.review_cache/bank_prompt_test5000_v1/archive_text.json`. Audit `review_semantic_test.py`: **5.067 kiểm tra PASS** về source24pins, checkpoint/runtime/beam evidence,5000IDs, caption/token, rawrefs/PTB, stage/scorerseals, CIDEr/SPICE saved-score means, CHAIR counts và NLL. Không chạy localTorch/mô hình/scorer hoặc mở binarycheckpoint/features. BLEU/METEOR/ROUGE là điểm runtime báo, không tái chấm local.

## So trực tiếp baseline199 trên cùng TEST

Cùng5000IDs/rawreferences, TRAIN vocabulary, toàn bộ patch/caption stage byte seals, beam5/max20/rawlogprob/rank0/no lengthnormalization. Bộchấm1.2 và evaluatorasset hashes giống nhau. Cả hai từ đầu10XE, checkpoint maxVALIDCIDEr: baseline chọnepoch10, bank+prompt chọnepoch9; không SCST/rerank/tuning trong lượtTEST. Tham số và chi phí tăng ở mô hình mới; comparison là M2+M3 kết hợp.

| Chỉ số | Baseline199 | Bank + prompt best9 | Chênh lệch mới−baseline |
|---|---:|---:|---:|
| BLEU-1 | 77,1623 | 76,9530 | −0,2093 |
| BLEU-4 | 37,1675 | 36,6553 | −0,5122 |
| METEOR | 28,7868 | 28,6547 | −0,1320 |
| ROUGE-L | 57,5850 | 57,3237 | −0,2613 |
| CIDEr | 120,5145 | 120,4520 | −0,0625 |
| SPICE | 21,6475 | 21,6008 | −0,0467 |
| CHAIRs | 5,16% | 4,16% | −1,00điểm% |
| CHAIRi | 3,5958% | 2,9299% | −0,6658điểm% |
| Recall | 43,7254% | 43,8651% | +0,1397điểm% |
| NLL | 2,121834 | 2,123528 | +0,001694 |

Các metriccaption nhân100; CHAIR/Recall là%; NLLkhông nhân100. CaptionbịCHAIRđánhdấu258→208, giảm50/5000; khoảng19,38%tươngđối. Mentionhallucination264→214, objectmentions7342→7304; recalledunique6885→6907. Độdài9,3906→9,4242từ, khôngcaptionrỗng/tốiđa2từ. Giảmảo giác không đi kèm câu ngắn hơn trong thống kê trung bình này; không phải bằng chứng nhân quả về cách mô hình sửa câu.

## Bằng chứng và giới hạn

Theoảnh:139ảnhmấtcờCHAIR,89ảnhmớibịgắn, net−50; không gọi139caption đã hoàn toàn đúng vì CHAIR chỉ kiểmtraảo giácđối tượng, không kiểm chứng quan hệ/hành động. Bootstrap cặp2000mẫu/seed42 trên caption cốđịnh và referenceIDF: CIDErCI95[−1,5025;+1,4061]điểm; CHAIRsCI95[−1,60;−0,40]điểm%. Đây là bootstrap cóđiềukiện trên1seed/checkpointđãchọn, không điều chỉnh nhiều so sánh, không thay thế độbiếnthiên giữa lầntrain.

**Không có tăngCIDEr trênTEST.** Chênh−0,0625rất nhỏ ở pointestimate, nhưng chưa phải phép kiểm tươngđương/noninferiority. BLEU/METEOR/ROUGE/SPICE vàNLLđều hơi kém, cần báo đầyđủ. VALIDtăngCIDEr1,4862 chưa lặp lại ởTEST. Kết quảủnghộ giảmảo giác đối tượng với điểmCIDEr quan sát gầnbaseline, không ủnghộ tăng mọi chỉ số.

Đã có cơ chếbank–patch vàprompt–patch học bằngXE, nhưng comparisonchưa táchM2/M3, ngữ nghĩa chữ, geometry mask hoặc lợiích thêm7,34Mtham số. Không nói riêngprompt gây ra toàn bộ lợiích. Nghiêncứu có các vòng xemTEST trước đó; ghi rõpriorprojectTESTexposure, không tuyên bố TESThoàn toànchưatừngdùng. Lượt này giữ checkpoint/decoder theoVALID và khôngtuningTEST.

Runtime mới train/infer2.11.0+cu130; baseline traincu128/infercu130, khôngbitwise-equivalence. GPU firstbeam-pathguard PASS; stageTEST656,92s (~10m57), worker123,02s (~2m03, gồmverify/load/beam/NLL), khônggồmsetup/CPUmetrics/billing.

## Quyết định

Giữ hai kết quảTEST như hiện tại. Nhánh mới là ứngviên giảmảo giác, không đổi sangepoch10 hoặc chọn cấu hình theoTEST này. Khuyếnnghị bước nghiêncứu kế tiếp là ablation trênTRAIN/VALID để táchbank vàprompt trước khi khẳngđịnh đónggópcủaM3; chưa tự tạo/chạy thêmthínghiệm hoặc sửaWord. Cả trọng sốnguồnversion9 và caption/scoreroutputversion10 cần bảo toàn.

Kết quả được ghi tại Sheet Kết quả mới A134:L136, đọc lại giá trị và format qua API; không kiểm tra giao diện trực quan.
