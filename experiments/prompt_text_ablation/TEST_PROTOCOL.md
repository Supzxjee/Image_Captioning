# TEST của zero-text M3 — 23/11/2026

Người dùng đồng ý chấm TEST sau kết quả ablation VALID của v193. Chuẩn bị notebook local, người dùng tự chạy Kaggle; không inference/Torch/scorer/API thật local. Không train, rerank, chỉnh ngưỡng hoặc chọn lại checkpoint bằng TEST.

Khóa best.bin của `ducanh3/notebook7bc83c4c76/193`, epoch8/seed42, SHA `0aa81a0c0168c64987b77e0c86203fa5c4f7ccb858de806ed8842130ce37e1e0`. Snapshot cursor epoch9/0,141688updates/4533976captionevents,history8; không dùng finalepoch10/SHA80998038 hoặc run1best7.

Giữ nguyên ZeroTextCaptioner và mọi sourcecore/adapter của lượt train. Kiểm checkpoint identity method `full_zero_prompt_text_xe10_v1`, prompt_text_mode zero, seed42, toàn bộsourceSHA. Nạpbankbuffer từ checkpoint; giữ M1/M2/ROI/geometry, chỉ zero M3object_text/relation_text như khi train. Không chuyển sang CSAPatchCaptioner bình thường dù strictstatekeys giống nhau. Guard đầu mỗiGPU dùng8ảnhTEST: thaytext thànhNaN trên dictcopy không đổi memory, kiểm zeroedtext/ROI và beamrescore; không thayweights hoặc captionselection.

Ba dataset: visual-cache, vuthetam/mscoco-2014, csa-prompt-test-cache-v181. TEST181 vẫn cần để giữ cùng ROI/masks/geometry với fullmodel; kiểmcache/seals trướcstaging. Không cần bank130/prompt175trainDataset hoặc Datasetcheckpointmới. APIexactv193 chỉ tảibest.bin bằngpinnedCLI, Secretscodex/KAGGLE_API_TOKEN có quyềnnguồn; cóCHECKPOINT_ROOT nếugắnđúngfile, luônverifySHA. Khônglộtoken hoặc fallbacklatest.

CùngcohortKarpathy5000TEST,vocabulary/patch extraction,FP32,T4×2,beam5/max20/rawlogprob/no lengthnormalization và bộchấm của fullseed42. ReferenceTEST chỉ chấmmetric/NLL sau decode; không dùng chọncaption. Lưu top1 và5beam/ảnh trướcCPUmetrics; basic/METEOR/SPICE/CHAIR vàcombine táchphase, cầnđủ8metrics (+Recall/NLL) mớicomplete. Không tựdecode lại khiCPUmetriclỗi. Cap2h gồmsetup/scoring, khôngETA/billing; chỉstageTEST≈1.4GiB, khôngcopy fullTRAINcache hoặc chạyCLIP/detector lại.

Localchecks: compile/hash/importclosure/undefinedglobals, effectiveAPIroutev193, checkpointmetadata accept/rejectzero mode/source/epoch/cursor, unchangedtrainingmodel andscorecode, guaranteedzero factory andGPUguard hooks, all8metricscompleteness. ActualAPIauthorization/binary/CUDA/Java/scorers chỉ cósauuser-run.

Primarycomparison: zero-textbest8 với correctpromptfullseed42best9, cùngrulemaxVALIDCIDEr sau10epochs, trướcSCST/rerank. Mộtseed, priorTESTdevelopmentexposure và historicalI/Odifference đượcgiữ trongbáocáo. Khôngformalnoninferiority/significance hoặc action/relationhallucinationclaim từCHAIR. ChưatriểnkhaiM2-only/additionalseeds/Word hoặc ghi dòngkếtquảSheet cho notebookchưa chạy.
