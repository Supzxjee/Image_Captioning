# Prompt UPT có thể nối với ROI hiện tại để tiếp tục pilot CSA

05/11/2026. Audit người dùng chạy **hoàn tất TRAIN10k + VALID5000**, không TEST/huấn luyện/mô hình. Nguồn người dùng xác nhận: `ducanh2403/2-vi-cai-kia-qua-nhieu/167`. [Số liệu kiểm tra](PROMPT_MAPPING_RESULT.json), [mã kiểm metadata/text](audit_prompt_mapping_result.py), [thiết kế](REGION_LINKED_PROMPT_CSA_PROPOSAL.md).

**Khuyến nghị dùng tiếp các ROI đã gắn được prompt để thử nghiệm nhỏ**, chưa cần trích xuất lại đặc trưng hoặc hạ ngưỡng IoU. Đây là kết luận về tính khả thi của dữ liệu, chưa phải kết quả caption hay giấy xác nhận mọi quan hệ UPT đúng.

| Số liệu | TRAIN10k | VALID5000 | Tổng |
|---|---:|---:|---:|
| Đối tượng đầu vào |41.878|20.674|62.552|
| Đối tượng map được trước cap |31.970|15.906|47.876|
| Tỷ lệ map đối tượng trước cap |76,34%|76,94%|76,54%|
| Đối tượng giữ sau cap8/ảnh |29.670|14.690|44.360|
| Quan hệ đã chọn trong cache |5.246|2.526|7.772|
| Quan hệ map được cả hai endpoint |4.839|2.319|7.158|
| Tỷ lệ map quan hệ |92,24%|91,81%|92,10%|
| Ảnh còn prompt đối tượng |9.752|4.875|14.627|
| Ảnh còn prompt quan hệ |2.480|1.214|3.694|
| Ảnh không còn prompt gắn vùng |248|125|373|

Có **97,51% ảnh** còn ít nhất một prompt gắn vùng, nhưng chỉ **24,63% toàn bộ ảnh** có prompt quan hệ. Không được diễn giải92,10% là tỷ lệ ảnh có quan hệ hoặc độ đúng của quan hệ.

## Những phần bị bỏ

14.120object records không tìm được ROI cùng nhãn/IoU≥.5;556records bị loại do nhiều UPT instances cùng map một ROI. Không có ambiguity đồng hạng được ghi trong summary. Có3.516objects đã map nhưng bị cap8/ảnh loại tiếp; tỷ lệ giữ cuối44.360/62.552=**70,92%**, thấp hơn tỷ lệ map76,54% trước cap. Có614quan hệ không map đủ hai endpoint. Số ảnh có quan hệ giảm4.003→3.694; không có prompt đầu vào99ảnh, và274ảnh khác mất các prompt khi mapping, tổng373ảnh fallback.

Không thể suy ra mọi object không map được đều là UPT sai hoặc detectorCSA sai; chưa có nhãn người xác minh cho hai nguồn. Giữ fallback CSA nguyên bản ở những vùng/ảnh không có liên kết, không gắn ngẫu nhiên hoặc ép sinh object/quan hệ đó. Quan hệ UPT vẫn là dự đoán có ngưỡng exploratory, không phải GT.

## Kiểm tra đã thực hiện

ZIP SHA `58e9b4ae01234c31a9f9025de083d151888a9481337d1aac544b2a8c09818558`; CRC PASS, text được lưu một lần rồi dùng cache. Runtime source `14c5bf0e6b19480d9666db4af517e930967c4ea1`,16Git source/asset hashes khớp. **715 assertion checks PASS**: status/receipt, ba payload text hashes, cohort10k/5k/noTEST/disjoint, TRAIN pool bằng đối chứng SCST cũ và VALID ID set bằng rerank164; partition số đếm/mapping/caps/ratio;40ví dụ có ID/vùng/cặp hợp lệ, tính lại IoU và orderedpairgeometry, kiểm template/threshold; log7testsPASS/đủ15000/no traceback.

40ví dụ được lấy có chủ đích5mẫu/bucket, không phải mẫu ngẫu nhiên đánh giá precision. Chỉ kiểm hình học/text/metadata, **chưa xem ảnh để xác nhận ngữ nghĩa**. File liên kết đầy đủ15000dòng và318text không nằm trong ZIP review; SHA của chúng là bằng chứng worker người dùng, chưa được đọc độc lập ở máy tác giả. Nguyên bản UPT/detector/ROI binaries cũng không có trong ZIP. Khi cache318text phải lấy đúng payload từ output167 và xác minh SHA trước chạy CLIP.

Thời gian CPU87,31giây do worker ghi nhận, không gồm toàn bộ setup/download; không phải dự toán train/GPU. Lượt này không có CIDEr/CHAIR/NLL mới, không mô hình nào được huấn luyện.

## Bước tiếp theo trong phạm vi đã duyệt

Giữ toàn bộ `/kaggle/working/csa_prompt_roi_audit_v1` ở version167. Cần `receipt.json`, `summary.json`, `cohort.json`, `prompt_links.jsonl`, `prompt_texts.json`; không chỉ giữ ZIP nhỏ. Chuẩn bị notebookCPU tải đúng167 hoặc nhận mount bản sao có đúngSHA, kiểm đủ15000links/318text, mã hóa CLIP đóng băng và chạy neural fixture smoke do người dùng thực hiện. Bank danh/động từ CSA vẫn giữ; cache prompt là phần bổ sung. Sau cache/smoke mới tạo matchedCONTROL/PROMPT500trainer. Không tự mở TEST hoặc train full, không cần chạy lại mapping167.
