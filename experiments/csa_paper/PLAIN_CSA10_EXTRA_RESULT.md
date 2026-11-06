# NoCSA10 / CSA10 — đủ8chỉ số trênKarpathyTEST5000

07/11/2026: COMPLETE; người dùng xác nhận output chấm bổ sung SaveVersion186 (notebookducanh2403/2-vi-cai-kia-qua-nhieu theo ngữ cảnh tiếp tục, chưa tải xác minh quaAPI). Không phải checkpoint186: weightsNoCSA10 từtrain45,CSA10 từ139. Giữ đúngcaption archived đãaudit; khôngtrain/sinhcâu/chọncâu lại.

## Bảng chính

Thang×100, CHAIRs/i là%. Cảhai exact10XEepochs/beam5/max20; điểmquality↑, CHAIR↓.

| Model | BLEU1 | BLEU4 | METEOR | ROUGE-L | CIDEr | SPICE | CHAIRs | CHAIRi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NoCSA10 | 35.39 | 15.08 | 16.77 | 40.33 | 56.92 | 11.64 | 3.92 | 4.43 |
| CSA10 | 70.35 | 33.34 | 25.25 | 53.46 | 101.92 | 18.66 | 6.18 | 4.61 |

CSA tăng44.9987điểmCIDEr,18.2539BLEU4,8.4832METEOR và7.0146SPICE. Cả6metricquality tăng nhưngCHAIRs tăng2.26điểmphầntrăm,CHAIRi tăng0.1833điểmphầntrăm. CaptionCHAIRđánh dấu196→309. Vìvậy không nói riêngCSA đã giảmảo giác. NoCSA có1853caption≤2từ (37.06%),CSA388 (7.76%); recallđối tượng24.0633%→41.4327%. LowerCHAIRs củaNoCSA phải xét cùng mô tả ngắn/coverage thấp. Một seed, encodercapacitykhác; khôngequivalence/sota/causalpromptclaim.

## Đặt cạnhpipelinehiện tại

PROMPT+SCST/CE+rerank hiện tại đạt115.3522CIDEr/3.20%CHAIRs, nhưng lịchhuấn luyện khác CSA10. Khôngquy toàn bộchênhlệch cho riêngprompt. Đối chứngprompt phù hợp hơn là CONTROL/PROMPT mới, bảngđủ8metric ở [PROMPT_TEST_EXTRA_RESULT.md](PROMPT_TEST_EXTRA_RESULT.md).

## Kiểm tra

43storedmetadata/hash/source/cohort/smoke/phase/CSVscale/replaychecksPASS. SnapshotđúngSHA512496B,5000ảnh/25010refs,5sourcepinsGitkhớp; PTBrefs bằng nguyên nội dung vớiCONTROL/PROMPT hiện tại. CùngCOCO1.2/Java8/CoreNLP3.6.0 vàreportedjar/modelSHA như185/62. Cả2×5000caption/mỗiảnh1câu, mỗi3phaseđủcohort. BLEU4/ROUGE/CIDEr replaykhớp; CHAIRs/i táidùngauditcũ sauSHAđúng, khôngchấmCHAIRmới. Tác giả chỉđọcZIPmộtlần rồitextcache; khôngchạyJava/scorer/model. METEOR/SPICE chỉlưuaggregate+coverage, khôngcóperimageCI ởlượtnày.

[JSON](PLAIN_CSA10_EXTRA_RESULT.json), [CSV](PLAIN_CSA10_EXTRA_TABLE.csv), [audit](audit_plain_csa10_extra_result.py), [nguồn/đối chứngcũ](PLAIN_VS_CSA10_TEST_RESULT.md).

Cònthiếu3metric choCSA15+SCST trước/sau rerank(TEST166), nếugiữmốccũtrongbài. Không mở thêmrun tronglượtreviewnày. Word06/11 vẫn chưa cập nhậtbảng8metric; cập nhậtsốliệulàbướcsoạnthảoriêng.
