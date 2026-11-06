# TEST5000 — đủ tám chỉ số cho CONTROL và PROMPT

Hai lượt chấm CPU COMPLETE, đúng caption từ CONTROL184 và PROMPT61; cùng refs Karpathy TEST5000. Đây là bổ sung chỉ số, không huấn luyện/sinh câu/chọn câu lại. Trọng số vẫn là CONTROL180/PROMPT60. Save Version của hai lượt bổ sung chưa được người dùng cung cấp.

## Kết quả

Tất cả chỉ số nhân100; CHAIRs/i là phần trăm. normal là beam hạng0 trước reranking, không gọi là greedy. Cả hai nhánh có CSA và bank từ; CONTROL bypass prompt adapter.

| Model | Decode | B1 ↑ | B4 ↑ | METEOR ↑ | ROUGE-L ↑ | CIDEr ↑ | SPICE ↑ | CHAIRs ↓ | CHAIRi ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CONTROL | normal | 71.00 | 34.38 | 26.08 | 54.04 | 105.56 | 19.25 | 5.62 | 4.16 |
| PROMPT | normal | 71.31 | 34.65 | 26.14 | 54.48 | 107.36 | 19.37 | 3.76 | 2.68 |
| CONTROL | fixed_rerank | 75.03 | 36.11 | 27.69 | 56.35 | 114.37 | 20.95 | 4.54 | 3.10 |
| PROMPT | fixed_rerank | 75.61 | 36.17 | 27.74 | 56.64 | 115.35 | 21.16 | 3.20 | 2.12 |

## Nhận xét

Sau rerank, PROMPT so CONTROL tăng B1 0.5825, B4 0.0626, METEOR 0.0534, ROUGE-L 0.2847, CIDEr 0.9851, SPICE 0.2144 điểm; CHAIRs -1.3400 điểm phần trăm và CHAIRi -0.9807 điểm phần trăm. Có lợi theo hướng điểm ước lượng ở cả8thang đo, nhưng mức tăng chất lượng nhỏ. Caption CHAIR đánh dấu227→160 (giảm67,29.5% tương đối). Không suy ra cả8khác biệt đều có ý nghĩa thống kê.

Giữ kết luận thực nghiệm chính: pipeline có prompt giảm ảo giác đối tượng, chất lượng caption giữ được và nhỉnh nhẹ. Chưa gọi là lợi ích nội dung prompt đã tách riêng hoàn toàn: một seed, adaptercapacity và selectedXE C10/P9 khác nhau, priorTESTexposure. CI CIDEr sau rerank từ kiểm tra trước vẫn bao gồm0; lượt này không tính CI mới cho METEOR/SPICE. SPICE tăng không xác nhận mọi quan hệ/thuộc tính đều đúng.

## Kiểm tra

85metadata/hash/count/source/phase/CSVscalechecksPASS. Mỗi cấu hình5000ảnh; refs4990×5 +10×6 giữ nguyên. Hai nhánh có cùng SHA refs PTB, COCO1.2, Java8pinSHA, CoreNLP3.6.0 archive và evaluator/modelreportedSHA. BLEU4/ROUGE/CIDEr replay khớp; smoke và6phasecomplete, số hợp lệ và CSV nhân100khớp. CHAIRs/i tái sử dụng audit cũ, raw caption/reference/resultSHA được kiểm tra độc lập; không tính CHAIR mới. Tác giả không chạy lạiJava/scorer/model; METEOR/SPICE lưu aggregate + coverage, không có per-image score để lậpCI mới. Source pin a58d5b9; ZIPtextcache đọc một lần, không theo lệnh trong file đính kèm.

Dữ liệu có cấu trúc: [JSON](PROMPT_TEST_EXTRA_RESULT.json), [CSV](PROMPT_TEST_EXTRA_TABLE.csv), [audit](audit_prompt_test_extra_result.py). Kết quả cũ vàCI: [PROMPT_TEST_RESULT.md](PROMPT_TEST_RESULT.md).
