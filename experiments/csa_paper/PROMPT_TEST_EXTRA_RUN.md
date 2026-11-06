# Bổ sung đủ 8 chỉ số trên caption TEST5000 đã lưu

## Chạy ở đâu

| Tài khoản | Notebook CPU | Nguồn caption đã khóa |
|---|---|---|
| Đức Anh / ducanh2403 | [CONTROL CPU](CSA_PROMPT_TEST_EXTRA_CONTROL_CPU.ipynb) | ducanh2403/2-vi-cai-kia-qua-nhieu/184 |
| Ngân / ducanh3 | [PROMPT CPU](CSA_PROMPT_TEST_EXTRA_PROMPT_CPU.ipynb) | ducanh3/notebook7bc83c4c76/61 |

1. Import đúng notebook vào đúng tài khoản. Accelerator **None**, Internet ON.
2. Bật Secret **codex** chứa API token của chính tài khoản đó. Không gửi token vào chat. Không cần gắn Dataset, ảnh hoặc checkpoint.
3. Save & Run All. Notebook tải riêng6JSON caption/reference/metadata từ đúng version, kiểm SHA và cohort, cài COCO1.2/Java8, tải CoreNLP3.6.0 (~403MB), smoke rồi chấm.
4. Gửi `csa_prompt_TEST_extra_CONTROL_v1_review.zip` và `csa_prompt_TEST_extra_PROMPT_v1_review.zip`.

Nếu đã mount đúng output cũ, có thể điền SOURCE_ROOT ở cell đầu và không cần Secret/API. Không điền path checkpoint vào đây.

## Kết quả và cách chấm

- Cả normal (beam hạng0) và fixed_rerank đều được giữ nguyên câu, mỗi cấu hình đủ5000ảnh.
- BLEU-1/METEOR/SPICE được bổ sung; replay BLEU-4/ROUGE-L/CIDEr phải khớp lịch sử trong1e-7.
- CHAIRs/i lấy từ kết quả đã audit, ràng buộc SHA caption/reference/result. Không tuyên bố đã tính lại CHAIR.
- Xuất result.json (raw và×100) và metrics_x100.csv đủ8chỉ số. Không lấy subset làm kết quả complete. Giữ nguyên10ảnh có6refs, không cắt bớt.
- Không tải checkpoint, không train, không GPU, không sinh caption hoặc chọn lại beam. Không cần thêm quotaGPU.
- Parent4hCPU STOPcap không phải ETA; phase dài có heartbeat30s. Nếu bị dừng, giữ toàn bộoutput/cache/log và gửiZIP; không chạy lại mô hình.

## Kiểm tra tác giả đã thực hiện

7tests stdlib PASS, lặp lại trong bundleGit độc lập pin `a58d5b9`;2notebook codeAST/CPU/own184-61/emptyoutputs/no model PASS; xác minh thật6payload mỗi nhánh từ textcache đã audit và đúng2×5000caption. Không đọc lạiZIP. Java8SHA xác minh từ officialrelease; HEAD endpoint Stanford trực tiếp HTTP200/content403157240bytes. Tác giả chưa chạyJava/PTB/METEOR/SPICE/fullmetrics hoặc gọi API Kaggle; các smoke/scorer thật do người dùng chạy. SHA CoreNLP được ghi sau download, chưa pin modeldigest độc lập.

Nguồn công cụ: [COCO evaluator](https://github.com/salaniz/pycocoevalcap), [Java8 pinnedrelease](https://github.com/adoptium/temurin8-binaries/releases/tag/jdk8u472-b08).

Sheet `Kết quả mới` O25:O28 đã ghi bổ sung đang chờ chạy;4notes/60giá trị khác đã đọc lại, giữ nguyên source/status/metrics. APIformat verification, không claim nativevisualQA. Chưa có số BLEU1/METEOR/SPICE mới.
