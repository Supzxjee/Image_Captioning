# Bộ tạo prompt UPT v1 — chuẩn bị trong khi full cache chạy

Đã chạy local từ ZIP native-score 3.374 ảnh. Không chạy notebook/GPU, không train
caption, không đánh giá CIDEr/CHAIR mới. Full cache đang chạy không bị sửa.

## Cách diễn đạt

| UPT verb | Loại vật | Cụm đưa vào prompt |
|---|---|---|
| hold | umbrella | a person holding an umbrella |
| hold | skis / scissors | a person holding skis / scissors |
| ride | bicycle / motorcycle / horse | a person riding a bicycle / motorcycle / horse |
| ride | skateboard / snowboard / surfboard | a person riding a skateboard / snowboard / surfboard |
| ride | skis | a person on skis |

Giữ tên hành động rộng thay vì tự biến `ride surfboard` thành `surf`: chỉ có loại
vật chưa đủ xác nhận chuyển động lướt sóng. `on skis` là cách diễn đạt yếu hơn cho
trường hợp ride skis, không thêm động tác. Bộ ba gốc, index/ID hai instance, bbox,
score/threshold được giữ riêng. Câu chữ không làm bộ lọc chấp nhận thêm quan hệ.
Đây là bảng diễn đạt đầu ra, KHÔNG phải bảng sửa GT hoặc đánh giá tương thích
HICO/V-COCO đã hoàn thiện.

## Chọn và tạo prompt

1. Đọc native score. Full cache chỉ lấy cột hold/ride được ontology hỗ trợ theo
   từng loại vật; chưa dùng các verb khác khi chưa có policy tương ứng.
2. Dùng snapshot ngưỡng đã khóa: holding 0.503703236579895, riding 1.01 (abstain).
   Không refit hay thay ngưỡng theo Dev/test; do đó bản này chỉ chọn holding.
3. Bỏ claim trùng đúng cặp/verb; giữ tối đa 3 quan hệ/ảnh theo score-margin giảm
   dần và tie-break cố định. Margin không phải confidence đã hiệu chuẩn.
4. Giữ mọi ID của các quan hệ được chọn, không gộp hai vật cùng lớp thành một
   instance. Nếu hai cụm chữ giống nhau, câu chỉ viết một lần nhưng các quan hệ
   có ID vẫn lưu riêng; không suy ra số người hay bao phủ toàn bộ nhóm.
5. Có quan hệ: `a photo showing a person holding an umbrella.` Không có quan hệ:
   dùng object fallback, confidence >=0.5, tối đa 8 tên lớp khác nhau. Đây là cấu
   hình kỹ thuật, chưa chứng minh precision của detector. Không có vật qua ngưỡng:
   `a photo.` Quan hệ đã qua ngưỡng vẫn giữ endpoint dù confidence vật <0.5;
   không lén áp một bộ lọc mới vào quan hệ đã khóa.

Không dùng caption tham chiếu, nhãn V-COCO, GT boxes, matched GT IDs hay nhãn
đúng/sai để chọn prompt. `from_audit` chỉ lấy allowlist các trường prediction;
cache thật không có các nhãn đó. `candidate_preview` chỉ để xem dự đoán trước lọc,
không nối vào chuỗi prompt hay đưa vào decoder.

## Kết quả thử local

- 3.374 ảnh: 855 object_relation (25,34%), 2.519 object_only; không có no_objects.
- 958 holding claims qua ngưỡng; giữ 952, bỏ 6 do giới hạn 3 quan hệ/ảnh.
- Riding được diễn đạt trong preview nhưng không được chấp nhận bởi policy cũ.
- Có bản review 50 ảnh, chia theo mode và thứ tự hash cố định. Không phải mẫu
  ngẫu nhiên đại diện, chưa phải người dùng đã chấm đúng.
- Đây là 507 Calibration + 2.867 Development đã xem; không phải tập caption
  validation/test hay full 123.287 ảnh. Không công bố đây là cải thiện caption.

Audit aggregate `audit/upt_prompt_prototype_v1_result.json` giữ hash nguồn,
cấu hình và nhãn kiểm tra SAU chọn riêng. Mẫu số không được thay bằng tập prompt
được giữ để tuyên bố recall nguồn quan hệ tăng.

## Chạy từ cache khi đã lưu

```powershell
python build_upt_prompts.py --cache "PATH/upt_coco_full_cache" --policy audit/upt_prompt_locked_policy_v1.json --output "PATH/new_upt_prompts"
```

Kiểm identity/checkpoint/schema, task-plan hash, vocab, header/ID/CRC các shard và
hash manifest nếu có. Mặc định thiếu shard là lỗi. `--allow-partial` chỉ dành cho
preview từ các shard hoàn tất; journal dở không được dùng. Báo complete theo số
ảnh thực tế, không gọi cache partial là đủ. File Input cũ không bị thay đổi.

Chạy từ kết quả đã có, không cần GPU:

```powershell
python build_upt_prompts.py --audit-zip "PATH/vcoco_upt_native_score_audit_review.zip" --policy audit/upt_prompt_locked_policy_v1.json --output "PATH/new_upt_prompts"
```

Output: `prompt_cache.json` (mapping filename -> prompt + evidence),
`prompt_summary.json`, `prompt_review_50.json`, `prompt_review_50.html`.
Thêm `--review-images "PATH/COCO_IMAGES"` để copy 50 ảnh vào `images/`, xem HTML
offline. Thư mục ảnh hỗ trợ cả file trực tiếp hoặc các thư mục train2014/val2014.
Nếu không truyền tùy chọn này, HTML dùng URL COCO và có thể không tải được trong
trình duyệt. Bản review đã giao đã được sửa dùng 50 ảnh local, kiểm tra giải mã
ảnh thành công và cập nhật ZIP Downloads. Format prompt
có trường `prompt` tương thích đầu vào embedding theo filename; chưa chạy encoder
CLIP hoặc tích hợp sang caption pipeline. Metadata embedding cũ cần đổi đúng variant
trước khi xuất token cache mới, không gắn nhãn UPT thành heuristic V2.1.

25 CPU tests PASS, gồm 9 test mới: sparse vocab adapter, chính sách abstain,
không thêm hành động cụ thể, fallback, chống GT leakage, giữ nhiều instance,
dedup/cap/endpoint, lỗi dữ liệu và cache partial. Chạy thật 3.374 ảnh PASS.
Chưa xác nhận GPU/full-cache end-to-end hoặc mức dùng RAM của 123.287 prompt.

## Kiểm tra caption đã lên kế hoạch, chưa chạy

Giữ nguyên checkpoint SCST+CE+object, ảnh, beam và lựa chọn caption. Ba điều kiện:
prompt đang dùng; UPT object-only; UPT object+relation (cùng detections/fallback).
Trước tiên xem một nhóm ảnh và kiểm tra model có đáp ứng thay prompt; đổi nguồn
prompt ở inference là domain shift, không bảo đảm model đã học cách khai thác.
Nếu phải adaptation, giữ cùng số cập nhật/CE/data giữa các điều kiện. Chọn cấu hình
trên validation; không mở lại caption test để chỉnh. Người dùng tự chạy notebook.
