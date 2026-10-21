# CSA hai pha: chạy hợp lệ, chưa vượt baseline A

Audit2026-10-22: **CSA_TWO_PHASE_SAVED_ARTIFACT_AUDIT_PASS**. Quyết định theo gate đã khóa: **dừng recipe CSA hiện tại, chưa fulltrain/tune/thêmseed tự động**. ZIP1.369.525bytes SHA256 `0c0730930856b102a8358d63207653df334e513920ec53b8b7d73d2f3e93ed26`.

| Hệ thống | CIDEr | CHAIRs | CHAIRi | Object recall | Caption có ảo giác/1000 |
|---|---:|---:|---:|---:|---:|
| Original checkpoint |1,189066 |5,8% |3,5452% |45,6809% |58 |
| A: huấn luyện tiếp, không CSA |1,205513 |5,2% |3,2475% |45,8039% |52 |
| CSA bật, kết quả chính |1,199998 |5,1% |3,1117% |45,9268% |51 |
| CSA tắt sau huấn luyện, chẩn đoán |1,212948 |4,8% |2,9625% |46,2650% |48 |

CHAIR chỉ đánh giá đối tượng COCO, không chấm quan hệ. Off là baseline đã học chung với CSA rồi bỏ token bổ sung, **không phải A hoặc nhánh thắng được chọn trước**.

## So sánh theo từng ảnh

Paired bootstrap20.000lượt,1000ảnh dev,seed42. Chưa đo biến thiên giữa seed hay điều chỉnh việc dev đã dùng nhiều lần.

| Chênh lệch CIDEr | Giá trị | Khoảng95% |
|---|---:|---|
| CSA bật − A |−0,005515 |[−0,024690; +0,013729] |
| CSA tắt − A |+0,007435 |[−0,012888; +0,028660] |
| CSA bật − tắt |−0,012950 |[−0,029562; +0,002663] |

Cả ba khoảng đều chứa0. Chưa chứng minh CSA bật kém hơn A hoặc tắt tốt hơn A. Nhưng cả bật và tắt **đều không đạt** gateΔ≥.01 và cận dưới95%>0 soA. CSA bật tốt hơn original theo point estimate cũng không thay thế đối chứng A mạnh hơn.

CHAIRs giảm52→51 khi bật,52→48 khi tắt; CHAIRi/recall theo chiều tốt. Chưa có review người mù mới hoặc bằng chứng độ ổn định cho các thay đổi nhỏ này. Bật/tắt làm267caption thay đổi; thay đổi câu không tự chứng minh thông tin CSA hữu ích.

## Kiểm toán thực nghiệm

-12source nhúng, source tác giả và protocol hashes khớp notebook phát hành. Baseline model/tokenizer/checkpoint/SCST/objectreward/visualreader nguồn không đổi giữa commitA vàCSA; run_pilot khác logging hook2dòng đã kiểm.
- Train10k/dev1000,targets train,initialbaseline hash,original predictions/per-imageCIDEr/CHAIR và evaluator reference-token hash giống A. Runtime2ranks/cả2pha và optimizerbaseline pha2 giống A.
- Pha1:1250CEsteps, chỉ23.161.644coreparams cập nhật;33.520.400baselineparams frozen,base/bank state bất biến.
- Pha2:1250jointsteps,baseLR5e-6/coreLR2e-5,CE200 rồi CE+.05SCST(CIDErD+.5object_count_v2). Loss/gradient hữu hạn; lịch học đúng; base/core params cập nhật; hai ranks đồng bộ.
- Phase1final-full hash bằng phase2initial-full; checkpoint chuyển pha giống hai ranks; optimizer bắt đầu rỗng. Samplerorders pha1=pha2,không trùng/thiếu10000IDs; CPU/CUDARNG-start hashes lặp lại đúng mỗi rank giữa hai pha.
- Historical A không lưu actualorder/RNG-start; đối chiếu A dựa source/seed/samplerpolicy, không tuyên bố đã so byte thực tế của thứ tựA. Stage layouts khác và tensor/checkpoint bytes không có trongZIP; worker đã kiểmhash, audit chỉ xác nhận evidence nguồn/log/receipts.
- Clipactive pha1:10/2500ranksteps;pha2:2500/2500. Clip chung có thể ảnh hưởng scale cập nhật baseline khi thêm core, không tự suy ra đây là nguyên nhân kết quả.

## Tài nguyên và giới hạn

Preparation3835,08s (~63,92phút);pha1training122,40s (~2,04phút);pha2training519,70s (~8,66phút). Chưa cộng bootstrap/fullIDF/generation vào trainingtime. Peak PyTorchallocated1,3336GiB/GPU,reserved1,3770GiB. Staging vẫn là phần tốn thời gian lớn; chưa chứng minh mức tối ưu I/O nào so lần trước.

CSA dùng nhiều hơn A1250core-onlysteps và23Mparams. Một seed,dev tái sử dụng và ngân sách sàng lọc2500updates chưa đủ kết luận hội tụ hay bác bỏ toàn bộ cơ chế CSA. Kết quả cụ thể: **chưa có bằng chứng lợi ích khi đưa semantic memory CSA vào decoder lúc suy luận trong cấu hình này**.

Không chạy lại metric/model locally. Phân tích dùng saved artifacts và cached A; [PILOT_RESULT.json](PILOT_RESULT.json) lưu số đầy đủ. Đã chuẩn bị [brief cho một lượt Astra Ultra](ULTRA_RESEARCH_REVIEW_BRIEF.md); chưa tự gửi hoặc đổi mức suy luận.
