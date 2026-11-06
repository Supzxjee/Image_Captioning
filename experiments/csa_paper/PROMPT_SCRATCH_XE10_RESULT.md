# CSA từ đầu, CONTROL so với PROMPT — đủ 10 epoch XE

06/11/2026. CONTROL: `ducanh2403/2-vi-cai-kia-qua-nhieu/178`; PROMPT: `ducanh3/notebook7bc83c4c76/58`, người dùng xác nhận. [Kết quả/208checks](PROMPT_SCRATCH_XE10_RESULT.json), [protocol](PROMPT_SCRATCH_XE_PLAN.md).

**Kết luận hiện tại: PROMPT chưa vượt CONTROL về CIDEr sau10epoch. Chưa đo CHAIR nên chưa kết luận giảm ảo giác.**

| Chỉ tiêu VALID5000 | CONTROL | PROMPT |
|---|---:|---:|
| Epoch được chọn bằng CIDEr | 10 | 9 |
| CIDEr ×100 của checkpoint tốt nhất | 99,4587 | 98,8453 |
| NLL VALID tại checkpoint được chọn | 2,448211 | 2,516285 |
| CIDEr ×100 tại cùng epoch10 | 99,4587 | 97,9000 |
| NLL VALID tại epoch10 | 2,448211 | 2,502108 |
| NLL TRAIN trung bình trong epoch10 | 2,438823 | 2,430591 |
| Số từ trung bình/caption epoch10 | 8,1054 | 7,7134 |
| CHAIR/BLEU/ROUGE/recall | Chưa đo | Chưa đo |

PROMPT−CONTROL: best−0,6134điểmCIDEr; cùngendpoint10−1,5587điểm vàNLLVALID+0,053897. TrainNLL củaPROMPT thấp hơn0,008231 ởepoch10, chưa chuyển thành chất lượngVALID cao hơn. TrainNLL là trung bình online trongepoch/dropoutON, cònVALID là mô hình cuốiepoch/dropoutOFF; không dùng chênh tuyệt đối để khẳng định overfit. Chưa có pairedCI/per-imageCIDEr; không khẳng định ý nghĩa thống kê hoặc phương pháp không thể hiệu quả.

| Epoch | CONTROL CIDEr ×100 | PROMPT CIDEr ×100 |
|---|---:|---:|
|1|27,41|27,33|
|2|35,22|43,32|
|3|46,32|69,67|
|4|70,75|86,84|
|5|83,13|87,85|
|6|88,67|90,30|
|7|91,99|95,23|
|8|95,60|97,05|
|9|98,40|98,85|
|10|99,46|97,90|

PROMPT có CIDEr cao hơn ởepoch2–9 nhưng lợi thế giảm dần và không giữ ởendpoint. Đây là đường học quan sát trong mộtseed, không chứng minh quan hệngữ nghĩa đúng hay hội tụ tốt hơn. Không tự chạy thêm5epochs/SCST hoặc lựa checkpoint theoCHAIR chưa đo.

## Tính hợp lệ và giới hạn

208checks metadata/source/count/token/SHA/lịchtrainPASS. Cảhai:113287TRAIN,566747captionevents/epoch,177110updates,5667470events sau10epoch;5kVALID/25010refs, khôngTEST/SCST;4DDPguards và5neuralfixtures trong logPASS. Same50sourcepins8420162, data/bank/prompt/ref/vocab/software, commoninitSHA `d6dad13dc69bf5fe9355e525e5240b518082d897ca82a53db4f71e37966c9791`. Mỗi epoch hai filecaption có2500ảnh/rank, đủ5000ID, nội dung khớp tokenvocab vàSHA đã seal. Cảhai freshstart/noresume; FP32/T4×2/global32/image-sharing. LỗiCPUruntime trước đó đã qua trong lượtCONTROL này, runtime ghiTorch2.11.0+cu128/CUDA12.8.

CONTROL49.239.288 vàPROMPT53.708.024tham số học, tăng4.468.736. Đối chứng so thêm toàn bộ adapter; chưa tách giá trịngữ nghĩa khỏi capacity, cần zero/shuffle/off nếu phát triển tiếp. Samearchitecture/commoninit không thay thế multiseed hoặc TEST độc lập. CHAIR chỉ chấm đối tượng, không chấm quan hệ/thuộc tính. Không so loss này với QFormer khácvocab/objective.

Thời gianworker quan sátCONTROL21290,6giây≈5,91h/PROMPT18941,2giây≈5,26h, chưa gồm toàn bộsetup/billing; chạy trên hai máy khác nhau, không dùng để khẳng định PROMPT tăng tốc.

## Giữ checkpoint và bước kế tiếp

- CONTROL178: best vàlatest cùngSHA `3dbc24409259f2e1a062c674fd5cec495e12aa168bcc1b747c4c5fc1beec1c46` (epoch10).
- PROMPT58: **giữ cảbest9** SHA `371f8d8ad4635617d0ca2a111aeba1fa81894631475d629ef81c31a71f26dc96` và **latest10** SHA `02a9080c7150afc7dfdd8f88d6dc0f537547abdb6945b9ba1ebe0080f75b0772`.

SHA trọng số trên là receipt trong ZIP, chưa đọc/deserialize binary ở máy tác giả. Người dùng giữ SaveVersion/entireoutput, chưa yêu cầu tạoDataset trọng số mới.

Chuẩn bị notebook CPU `CSA_PROMPT_SCRATCH_QUALITY_VALID5000_CPU.ipynb`: chấm **caption đã lưu** củaCONTROL10/PROMPT9/PROMPT10 trên cùng5kVALID/cachedreferencePTB; replayCIDEr≤1e−7, thêmBLEU4/ROUGE-L/CHAIRs/CHAIRi/Recall vàpairedCI. Không APISecret/AddInput/model/checkpoint/newgeneration/GPU/TEST. Best10/9 khóa trướcCHAIR, không lựa lại theo metric mới. Bản payload583.294bytes được seal từhaiZIP đã audit; không đặt data trong notebook để tránh ngưỡng1MB. 3safe package/synthetic paired/ASTfixturesPASS; actualscorers chỉ do người dùng chạy. Chưa có kết quảCPU, chưa mở rộng train.

[Notebook CPU](CSA_PROMPT_SCRATCH_QUALITY_VALID5000_CPU.ipynb) đã chuẩn bị:7.320bytes,6source/payloadpins17bf52c.3codecellsAST/emptyoutputs/AcceleratorNone/noAPI/sourceSHA/3fixtures từexactGitbundlePASS. [Authorchecks](PROMPT_SCRATCH_QUALITY_AUTHOR_CHECKS.json). Người dùng chọnCPU/InternetON, Save&RunAll rồi gửi `csa_prompt_scratch_quality_valid5000_review.zip`; không cần output178/58 gắnInput vì captions đã khóa trong payload. Tác giả chưa chạy evaluator.

CachedZIPtexts ở `.review_cache/prompt_scratch_xe10` ngoài repo; không cần đọc ZIP lại để tiếp tục phân tích.
