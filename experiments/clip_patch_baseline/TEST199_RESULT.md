# Baseline CLIP patch — TEST5000 hoàn tất, 10/11/2026

Nguồn chấm: notebook **kltnnnnnff, Save Version8**, người dùng xác nhận. Owner của notebook chấm chưa được cung cấp, không tự dựng link. Checkpoint huấn luyện: **ducanh2403/2-vi-cai-kia-qua-nhieu/199**, epoch10, SHA5061e305d98edf54930a200090a41c983c354d1ba2dfcf700b715cf20a5b80b7. Source chấmfc17e99; ZIPSHA trong TEST199_RESULT.json. Đủ5000ảnhTEST, haiworker2500/rank, cùng toàn bộ ảnh và caption tham chiếu nguyên văn với các kết quảTEST cũ. Không dùngTRAIN/VALIDfeature trong lượt này.

## Mô hình được chấm

CLIP ViT-B/16 frozen/cache →196patch sau bỏCLS →Linear/LayerNorm512 →TransformerDecoder chuẩn4lớp/8heads/FFN2048 →vocab10000. Học mới10epochXE/113287TRAIN/global32, chọnepoch10 bằngVALID trướcTEST. Khôngbank/prompt/detector/ROI/SCST/objectreward/retrieval/rerank. TEST dùngbeam5/max20token, chọnrank0 theo rawlogprob, khônglengthnormalization/tuning. ReferenceTEST chỉ dùng chấm vàteacherNLLsau khi sinhcaption.

## Tám thang đo, cùng checkpoint và caption

Các điểm ngôn ngữ hiển thị×100, CHAIR theo phần trăm.

| Thang đo | TEST5000 |
|---|---:|
| BLEU-1 | 77.16 |
| BLEU-4 | 37.17 |
| METEOR | 28.79 |
| ROUGE-L | 57.58 |
| CIDEr | 120.51 |
| SPICE | 21.65 |
| CHAIRs | 5.16% |
| CHAIRi | 3.60% |

Recall **43.73%**, teacher-forced NLL **2.1218**. CHAIRs258/5000caption; CHAIRi264/7342objectmentions. Recall6885/15746uniqueobjectsupports. Captiontrungbình9,3906từ, khôngcâu rỗng hoặc≤2từ. NLL khác vớiCIDEr và cácsplitkhác nhau, khônggọi TEST120,51 là tăng từVALID119,09do huấnluyệnthêm.

## So với các hệ thống cũ trên cùng TEST

| Cấu hình | CIDEr×100 | CHAIRs | CHAIRi |
|---|---:|---:|---:|
| Đối chứng ROI cũ, XE10 | 56.92 | 3.92% | 4.43% |
| Căn chỉnh bank cũ, XE10 | 101.92 | 6.18% | 4.61% |
| CONTROL cũ, SCST+CE+rerank | 114.37 | 4.54% | 3.10% |
| PROMPT cũ, SCST+CE+rerank | 115.35 | 3.20% | 2.12% |
| CLIP patch mới, XE10 | 120.51 | 5.16% | 3.60% |

So vớiPROMPTcũ, CIDEr+5,1622điểm nhưng CHAIRs+1,96điểmphầntrăm (thêm98captionflag), CHAIRi+1,4776điểmphầntrăm. So vớiCONTROLcũ, CIDEr+6,1473điểm nhưng CHAIRs+0,62điểmphầntrăm. So với căn chỉnhbankcũXE10, CIDEr+18,5938điểm/CHAIRs−1,02điểmphầntrăm. Đây là so sánh hệ thống với cùngreference, **không phải đối chứng cô lập cho lợi ích bank/prompt**: kiến trúc/đặc trưng/lịchhọc/trainingstage/selector khác nhau. Khôngkếtluậnbank/promptvôích hay cải tiến thànhcông mọi mục tiêu. Đối chứngROIcũ có1853caption≤2từ; CHAIRs thấp hơn khôngđồngnghĩacaption tốt hơn.

## Kiểm tra và môi trường

5056assertions vềsource/seal/cohort/token-caption/phases/counts/arithmetic PASS (chủ yếu5000captiontokenreplay, khôngphải5056thựcnghiệmđộclập). CIDEr vàSPICE pointkhớpmeanđiểmtừngảnh; CHAIRkhớpđếm lại; NLLkhớpweightedtokens haiworker. METEORperimageđủ5000finite, khôngépMETEORcorpus bằngmeanperimage. BLEU/METEOR/ROUGE giữđiểmdoofficialscorer ởruntime; tácgiảkhôngchạy lại neural/metrics/checkpoint/featurepayloadlocal. TừgốcTRAINvocabSHAkhớp199, fullcaptionstreamsealskhớp.

TrainTorch2.11.0+cu128, inference2.11.0+cu130; strictload vàhaiGPUbeampathrescoringthựcPASS, khácbuildđượcghi. Khôngđãchứngminhbitwiseoutputcrossbuild. pycocoevalcap1.2, Java8pinned, CoreNLP3.6modelarchiveSHAđượcghi nhưngkhôngpinnedđộclập, cùngquytrìnhcũ. Mộtseed, TESTđãđượcdùngtrướctrongdựán; lượt nàykhôngtune theoTEST.

Stage559,94s(~9phút20giây), workermax104,24s(~1phút44giây)gồmcheckpoint/stageverify,beamvàNLL. SPICElog7,494phút. Cácphầnkhôngphải toànthờigian/billing; khônggán4,4htraintrướcđâychoTESTinference.

## Quyết định tiếp theo

Giữcheckpoint199 và toànbộoutputchấmkltnnnnnff/v8; khôngtrain/sinhcaptionlại. Baseline120,51 là mốc mới cho các so sánh tiếp. Khuyến nghị pháttriểnbank/promptcănchỉnhtrênchínhbackbonepatchnày,vớibudgetkhớp vàchọnbằngVALID, xem cógiữCIDErtrongkhi giảmảo giác. ROIadaptercũ khônggắntrựctiếpvàopatchmàchưachốt mapping. Chưa có chấp thuận hay triểnkhai thínghiệmmới ởlượt review này. KhôngsửaWord.
