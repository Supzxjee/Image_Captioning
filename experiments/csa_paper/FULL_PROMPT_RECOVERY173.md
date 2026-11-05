# Phục hồi cache full từ Output173

05/11/2026. Người dùng gửi log và xác nhận SaveVersion173 của `ducanh2403/2-vi-cai-kia-qua-nhieu`; phiên đã kết thúc, còn Output. Không chạy notebook/API/model bởi tác giả.

## Bằng chứng từ log

19safe tests và exact169 base318×512 PASS. Mapping **113287TRAIN+5000VALID=118287ảnh hoàn tất**,410unique texts,252,71giây của pha mapping (không toàn notebook/billing). TRAIN giữ335723object slots/54574relations; VALID14690/2319, cùng countsVALID như167. Đây là coverage, không precision quan hệ. Dừng tại `inspect`, trước CLIP cho textmới: `row != old_records[i]` trảTrue nhưng exception không ghiCOCO ID/trường/giá trị. **Chưa xác định được trường lệch từ log; không khẳng định nguyên nhân là roundoff.**

## Sửa và điều kiện cho tiếp tục

- Bổ sung chẩn đoán toàn bộ15krecord cũ, ghiCOCO ID/path/old/new, sốexact/safe/unsafe, firstunsafe và ví dụ vào `comparison_diagnostic.json`; không dừng ở thông báo chung đầu tiên.
- Các trường ID/instance/S/O/ROI/text/sourcebbox/score/confidence/threshold/counters/structure vẫn giữ chính xác cảtype. Chỉ ba nhóm **computed**: objectIoU, normalizedroi_bbox, relationgeometry được nhận khácFP64 **≤1e−12 và float32 hữu hạn giống từngbyte**. Không hạIoU, không sửa box/record hoặc thay prompt. Nếu có bất kỳ lệch ngữ nghĩa/float32input nào thì dừng trước encoding, xuấtdiagnostic để xem tiếp.
- Đây là sửa policy so metadata, ghi sau failure; không diễn đạt như15kJSONrecord vẫnbit-exact nếu đã nhận tinycomputeddrift. Báo riêng `old169_records_exactly_preserved`, `old169_records_safe_derived_drift`, `old169_records_model_inputs_preserved`. Reuse318vectors vẫnbyte-exact. Chưa có bằng chứng actual173 sẽ đạtguard mới.

## Notebook người dùng cần chạy

`CSA_PROMPT_FULL_CACHE_RECOVER173_CPU.ipynb`: AcceleratorNone/InternetON, gắn **chỉ** Dataset `csa-prompt-pilot-inputs-v169`, bật API Secret `codex` của chủOutput173. Tải exact173 nămmappingpayload+receipt+source receipt; mockpreflightCLI2.2.5 phải gửiversion_label=v173. Kiểmsource gốc e294c5c/27pins và toàn bộpayloadSHA rồi copynguyênbyte sangoutputv2.

Không map lại118287ảnh, không cầnROI134/UPT/detector ở bước phục hồi, không xóaOutput173. Nếu guard đạt, chỉ encode textmới (410−318=92 khi xác nhận318subset), frozenCLIPCPU, rồi tạo receiptCOMPLETE. Nếu guard không đạt, giữdiagnostic, không bypass hoặc tạo fullcache giả.

Giữwholeoutput `/kaggle/working/csa_prompt_full_karpathy_cache_v2`; gửi `csa_prompt_full_karpathy_cache_v2_review.zip` kể cả khi dừng. NoXE/SCST/TEST/GPU. Rootreceipt gồm9payload nền +comparisondiagnostic khi có; dùng danh sách trong receipt khi lưuDataset sau đó.
