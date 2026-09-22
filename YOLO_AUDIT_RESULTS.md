# Kết quả audit cache YOLO MS COCO 2014

Audit ngày 22/09/2026 trên toàn bộ Karpathy split, dùng cache YOLO đã tạo cho đề
tài. Đây là kiểm tra coverage/schema/phân bố, không phải ground-truth accuracy.

## Kết quả chính

| Chỉ số | Kết quả |
|---|---:|
| Ảnh trong JSON/cache | 123.287 / 123.287 |
| Ảnh thiếu hoặc dư | 0 |
| Tổng detection | 867.045 |
| Trung bình / trung vị mỗi ảnh | 7,03 / 5 |
| Ảnh không có detection | 1.052 (0,85%) |
| Số lớp | 80 |
| Confidence trung bình / trung vị | 0,648 / 0,685 |
| Detection confidence < 0,5 | 281.462 (32,46%) |
| Box < 0,1% diện tích ảnh | 52.703 (6,08%) |
| Box > 90% diện tích ảnh | 8.748 (1,01%) |
| Box có aspect ratio > 10 | 1.945 (0,22%) |

Không phát hiện box sai schema, tọa độ đảo, vượt biên hoặc cặp box cùng lớp có
IoU từ 0,8 trở lên. Phân bố train/restval/val/test tương đối đồng nhất: val có
34.521 detection, test có 34.576; số ảnh rỗng lần lượt là 50 và 48.

`person` chiếm 271.047 detection (31,26%), cao hơn nhiều lớp còn lại. Đây là
phân bố tự nhiên đáng lưu ý khi dùng region classification loss vì lớp phổ biến
có thể chi phối gradient.

## Quyết định cho region alignment

Cache đủ sạch để dùng, nhưng không dùng toàn bộ detection làm supervision. Bản
ablation đầu tiên giữ prompt gốc và chỉ chọn region target thỏa:

```text
confidence >= 0.5
0.001 <= bbox_area / image_area <= 0.9
tối đa 10 box confidence cao nhất mỗi ảnh
```

Lý do: lưới CLIP ViT-B/16 chỉ có 14×14 patch. Box dưới 0,1% ảnh nhỏ hơn đáng kể
so với một patch nên vector vùng chủ yếu chứa nền; detection confidence thấp cũng
làm nhãn supervision nhiễu. Box gần toàn ảnh không thể hiện căn chỉnh cục bộ.

Ngưỡng này chỉ tác động `L_region`; không sửa prompt cache, visual cache hoặc tập
test. Metadata của `region_targets_yolo.pt` phải ghi số detection bị lọc để báo
cáo thí nghiệm có thể tái lập.
