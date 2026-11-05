# Cache prompt full Karpathy — hoàn tất tại version175

06/11/2026. Nguồn `ducanh2403/2-vi-cai-kia-qua-nhieu/175`, người dùng xác nhận. ZIP SHA256 `c843884773bffe56a2ae66ef4ffc0f62002881602ade439707d56a276850775c`. [Kết quả và inventory](FULL_PROMPT_FRESH_V175_RESULT.json).

| Chỉ tiêu | TRAIN | VALID |
|---|---:|---:|
| Ảnh trong mapping | 113.287 | 5.000 |
| Object slots giữ lại | 335.723 | 14.690 |
| Liên kết quan hệ đã map | 54.574 | 2.319 |
| Ảnh có prompt đối tượng | 110.434 | 4.875 |
| Ảnh có prompt quan hệ | 28.190 | 1.214 |
| Ảnh không có prompt được map | 2.853 | 125 |

`FULL_PROMPT_TEXT_CACHE_COMPLETE`, `run_status.complete=true`. Toàn bộ **410 cụm văn bản duy nhất** được mã hóa mới thành vector CLIP512D, float32/L2, frozenCPU, max13tokens, không truncation; không dùng lại169. Nhiều ảnh dùng chung cùng cụm text/vector nhưng **mapping instance/ROI/S/O vẫn riêng từng ảnh**. Không có TEST/caption training/caption checkpoint. Giữ bank danh từ/động từ130 riêng; cache prompt này không thay bank.

Log người dùng ghi mapping270,525giây, cache67,296giây, trong đó pha CLIP14,190giây (không cộng lặp14giây vào67giây; không phải thời gian toàn notebook/billing). Trạng thái `models_executed=false` trong cache_plan là kế hoạch trước encoding; manifest.new_encoding.model_executed=true và log410/410 xác nhận CLIP thật đã chạy do người dùng.

51 kiểm tra metadata/source/hash PASS: đủ ID độc nhất/disjointTRAIN–VALID/noTEST; đúng28sourcepins7dfb63a; cùngCLIP revision và8assetSHA; không lỗi loadweights;6payload có trong ZIP khớp receipt; bản sao mapping và receipt đồng nhất. ZIP nhỏ **không có `prompt_links.jsonl`, `prompt_texts.json`, `prompt_embeddings.npy`**, nên tác giả chưa trực tiếp kiểm toàn bộ arrays/records. Không khẳng định vectors bit-identical169 hoặc quan hệ đã đúng ngữ nghĩa. Tỷ lệ map91,42%/91,81% là ghép được với ROI, không phải precision quan hệ; chỉ khoảng một phần tư ảnh có prompt quan hệ. Ảnh thiếu mapped prompt phải được mask nhánhprompt, không tự gán nhãn âm.

## Bước kế tiếp

[Notebook lưu Dataset175](SAVE_CSA_PROMPT_FULL_V175_DATASET.ipynb): CPU/InternetON/Secret`codex`, không AddInput. Chỉ tải exact175 theo14tênfile; kiểm hash toàn bộ payload, vectors410×512/dtype/norm, cohort và trạng thái rồi copy nguyênbyte, tạo Private Dataset mới `csa-prompt-full-karpathy-v175`. Không chạy lại CLIP, không caption checkpoint hoặc train. Metadata license`other`, API chỉ báo thành công khi responseOK; chia sẻ quyền Dataset cho Ngân sau processing.

2fixtures mới về transportPASS (copy nguyênbyte, không ghi đè, dữ liệu bị sửa/NaN/versionkhác bị chặn),4regressiontests của transport169/161PASS. Notebook15,1KB/6codecellsAST/embeddedsource+APIpreflight+uploadscriptAST/14filecontract/CPU/private/emptyoutputs PASS. Chưa tác giả chạyAPI hoặc tạoDataset; fullpayload sẽ được xác minh lúc người dùng chạy bước lưu. Khi có pathDataset, chuẩn bị đối chứng CSA và CSA+prompt từ cùng freshinit, bank130/ROI134; chưa tự chốt sốepoch/GPUbudget hoặc mởTEST.
