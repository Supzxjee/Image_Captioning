# Prompt UPT v2: hỗ trợ đầy đủ 116 hành động

Đã mở rộng code, chưa chạy trên full cache thật đang được tạo. Không thay cache,
model, detector hoặc notebook đang chạy. Không có metric caption mới.

## Dùng khi đã có cache

```powershell
python build_upt_prompts.py --cache "PATH/upt_coco_full_cache" --all-actions --relation-threshold 0.5 --review-images "PATH/COCO_IMAGES" --output "PATH/upt_prompts_all_actions_v2"
```

Nếu chỉ muốn preview các shard hoàn tất của cache partial, thêm `--allow-partial`.
Không đọc journal đang ghi. Không ghi đè output cũ. `--review-images` copy 50 ảnh
để xem offline; bỏ tùy chọn này thì HTML dùng ảnh COCO qua mạng.

Ngưỡng 0.5 là cấu hình thử nghiệm đã nêu, có thể đổi bằng `--relation-threshold`;
không có nghĩa xác suất đúng 50% và không được gắn nhãn calibrated. Không tự chọn
ngưỡng theo Dev hoặc caption test. Giữ tối đa3 bộ ba/ảnh, ưu tiên native score cao;
dedup đúng cặp/verb, giữ riêng nhiều instance. Cặp cùng người/vật có thể giữ nhiều
hành động. Không ép hành động chỉ được một nhãn, không chặn riding bằng ngưỡng1.01
cũ trong chế độ này.

## Nội dung mở rộng

- Đọc toàn bộ cột verb hỗ trợ theo loại vật từ cache: 117 nhãn HICO, bỏ đúng
  `no_interaction`, còn116 hành động /520 tổ hợp positive từ600 tổ hợp gốc.
- Bảng116 template explicit ở `captioning/upt_action_text.py`, CSV xem nhanh
  `audit/upt_action_prompt_templates.csv`. Ví dụ `eat apple` → `a person eating
  an apple`, `sit_on chair` → `a person sitting on a chair`, `wear tie` →
  `a person wearing a tie`, `cut_with scissors` → `a person cutting with scissors`.
- `hold`/`ride` giữ alias `holding`/`riding` để tương thích dữ liệu cũ. Các action
  khác giữ nhãn gốc trong field predicate và cụm tiếng Anh ở field text.
- Không suy ra `surf` từ `ride surfboard`, không thêm giới tính, số người, vật mới
  hoặc hành động mới. Với `jump`/`run`, label không chỉ rõ vai trò ngữ pháp của
  vật nên dùng dạng có cấu trúc `a person and a horse (interaction: running)`;
  không bịa `over`, `on` hay `away from`. Template chưa được người dùng chấm chất
  lượng ngôn ngữ; chưa coi mọi diễn đạt là caption chuẩn.
- Object fallback, cap, bbox/ID, review50 và chống GT leakage giữ như v1.
- Nếu truyền `--policy` và không truyền `--all-actions`, chế độ v1 vẫn hold/ride
  theo snapshot cũ. Hai chế độ được ghi variant và policy riêng. Audit ZIP cũ
  chỉ có hold/ride, không đủ để giả lập116hành động và bị từ chối ở chế độ all.

## Kiểm tra đã thực hiện

34 CPU tests PASS: regression v1/cache/audit và4 tests mới. Vocabulary fixture
trích đúng HICO metadata pin `9dd59c1f6ee2ddc011567765d0bc644858391fea`, kèm hash.
Đối chiếu đủ116 labels; thử adapter/render cả600 tổ hợp bằng prediction tổng hợp,
bỏ80 no_interaction, giữ520. Test ngưỡng0.5/0.9, giới hạn3, gerund/preposition,
và chạy đủ pipeline cache nhỏ→JSON→HTML; summary ghi rõ uncalibrated.

Đây là kiểm tra tính đúng code, KHÔNG phải116 quan hệ đã đạt precision/recall hay
full-cache thực tế đã xong. Chưa chạy embedding CLIP hoặc caption inference.
Giữ bản review50 cũ nguyên trạng để không nhầm với kết quả v2.
