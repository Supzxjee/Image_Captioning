# CSA + prompt: pilot 500 update chưa có cải thiện rõ

05/11/2026. Người dùng xác nhận CONTROL ở `ducanh2403/2-vi-cai-kia-qua-nhieu/172`, PROMPT ở `ducanh3/notebook7bc83c4c76/56`. [Số liệu và kiểm tra](PROMPT_PILOT_RESULT.json), [protocol trước chạy](PROMPT_PILOT_PROTOCOL.md).

**Hai lượt đều hoàn tất 500 update / 4.000 ảnh TRAIN riêng biệt, cùng pool10k, checkpoint161, seed, thứ tự ảnh/caption và CE+0,05SCST.** Hai lần ORIGINAL tái hiện đúng cùng caption, điểm theo ảnh và metrics. Đã đánh giá đủ VALID5000, cả5caption/ảnh; **chưa rerank, chưa TEST**. Không phải một full TRAIN epoch.

## Kết quả VALID5000, chọn beam hạng0

CIDEr, BLEU4, ROUGE-L nhân100; CHAIR/Recall là phần trăm. NLL là CE trên toàn bộ reference VALID, không phải tổng loss CE+SCST.

| Chỉ số | Checkpoint161 đầu vào | CONTROL sau500 | PROMPT sau500 |
|---|---:|---:|---:|
| CIDEr ↑ | 106,3103 | 105,7525 | 105,9776 |
| BLEU4 ↑ | 34,8294 | 34,7194 | 34,6981 |
| ROUGE-L ↑ | 54,4929 | 54,3209 | 54,3284 |
| CHAIRs ↓ | 5,20 | 5,26 | 5,24 |
| CHAIRi ↓ | 3,7369 | 3,8271 | 3,8005 |
| Recall ↑ | 40,5387 | 40,3252 | 40,3252 |
| Caption chứa đối tượng ảo giác ↓ | 260 | 263 | 262 |
| Lượt nhắc đối tượng ảo giác ↓ | 267 | 271 | 269 |
| NLL ↓ | 2,365264 | 2,366776 | 2,366886 |
| Số từ trung bình | 8,4164 | 8,3820 | 8,3730 |

PROMPT hơn CONTROL **0,2251 điểm CIDEr**, giảm ròng **1 caption ảo giác**; recall bằng nhau. Paired bootstrap2.000 lần trên cùng5.000ảnh cho chênh CIDEr95%CI **[−0,2569; +0,6841] điểm**, có0. Về CHAIR, PROMPT sửa16caption nhưng tạo15caption ảo giác mới soCONTROL; McNemar exact p=1,0. Chưa có bằng chứng cải thiện đáng tin cậy; cũng không thể kết luận hai mô hình tương đương chỉ từ một pilot.

So với chính checkpoint đầu, PROMPT giảm0,3327điểm CIDEr, tăng2caption ảo giác; CONTROL giảm0,5578điểm, tăng3caption. CI của các thay đổi cũng chứa0. Giữ161 làm mốc tốt đã biết, không thay bằng một endpoint pilot chỉ vì nhỉnh hơn nhánh tiếp tục học.

## Prompt có được mô hình sử dụng không?

64ảnh được chọn bằng **64COCO ID nhỏ nhất trong VALID**, không ngẫu nhiên và không phân tầng theo quan hệ. Trên cùng checkpoint PROMPT sau500:

| Can thiệp khi suy luận | Caption greedy đổi | Max chênh logit trên64ảnh | ΔNLL so normal, chuẩn hóa token |
|---|---:|---:|---:|
| Text về0, giữ vùng/cặp | 0/64 | 0,002809 | −0,00000194 |
| Roll text giữa các ảnh trong batch, giữ slot/box | 0/64 | 0,002445 | −0,00000696 |
| Tắt hai adapter, giữ base CSA đã cập nhật | 0/64 | 0,085807 | −0,00019472 |

**Có tác động số học**, nên không gọi đường prompt bị đứt hoàn toàn: zero text đổi logits ở63/64ảnh; roll ở61/64. Tuy nhiên tác động text nhỏ và chưa đổi câu ở mẫu này. Không lấy534caption khác giữa hai arm làm bằng chứng riêng cho text: baseCSA đã học thành hai bộ trọng số khác nhau, thêm cả visualpair/geometry/parameters. Không đủ bằng chứng học ngữ nghĩa quan hệ tốt; CHAIR chỉ đo đối tượng.

Actual checkpoint161/CUDA/DDP preflight đều PASS ở cả hai rank, Torch2.11.0+cu128. Closed/empty memory giữ đúng đầu ra; policy replay maxdiff8,82e−6. Ở PROMPT, output projections có gradient ngay bước1; upstreamMLP nhận gradient ở bước2. Hai bước smoke đã bỏ trước worker nạp161 lại. Đây là kiểm tra đường tính toán, không phải kết quả caption có lợi.

## Bằng chứng và giới hạn

- Đọc mỗiZIP một lần, CRC+hash từng file PASS; CONTROL ZIP `bfebb843a9405e627f353b87faba61d4fbac80a47326d5429cb71421bd070141`, PROMPT ZIP `a92a11bb16040ab5ade68ec91c8f6b646fed296b3aa9070d2e07893fd90a178f`.
- Mã thực chạy cùng `fc54bdea66479e66c8e7662831be167325ffef8e`,25Gitpins. Đối chiếu toàn bộ100.000beam trong4pha, IDs/token–caption/scoreorder, refs/vocab/samplerfileSHA, NLL286.871tokens/pha, CIDEr mean từ điểm theo ảnh và CHAIR từ số đếm đã lưu. 220.253 assertions phần lớn là kiểm từngbeam, **không phải220.253 thí nghiệm hoặc lượt chạy mô hình**.
- Không chạy Torch/mô hình hoặc scorers trên máy tác giả. BLEU/ROUGE/CIDEr scorer gốc là bằng chứng user-run; tác giả không tính lại n-gram scorers. Binary checkpoint, stagedarrays và TRAIN references không có trong ZIP; hash kiểm thực của chúng dựa vào receipt/runtime.
- Worker CONTROL770,23giây/PROMPT727,21giây gồm nạp/reward/training/hai lầnVALID (PROMPT thêmprobe); không phải pure train time, không gồm toàn bộ notebooksetup và không phải GPUbilling. Không suy ra PROMPT nhanh hơn chỉ từ một lượt.
- Mộtseed/pilotngắn/VALID đã dùng nhiều lần. Không so trực tiếp105,98VALID rank0 với117,06TEST sau rerank. CI theo ảnh không bao gồm biến thiên huấn luyện giữa seeds.

## Quyết định tiếp theo

**Chưa mở rộng số ảnh/epoch hoặc tự tăng learning rate.** Giữ checkpoint161. Nếu tiếp tục vòng đã chốt, chấm **cùng quy tắc rerank cố định** cho các bộ5ứng viên đã có; không sinh lại caption, không huấn luyện lại, không chỉnhweights theo kết quả này. Rerank có thể thay điểm nhưng không tự chứng minh prompt đóng góp; vẫn phải soPROMPT vớiCONTROL cùng quy tắc.

Checkpoint cần giữ: CONTROL172 `c8d26b5ec872a8c537764f7367750166845fd8abdabbdcf5109167d9f746248f`; PROMPT56 `831fbc7e781dae6320ee330b23b4bb05541c199726aee1bb081f2ce0588994d8`. Giữ cả output/candidatefiles; số version đã ghi, chưa cần tạo Dataset mới chỉ để review.
