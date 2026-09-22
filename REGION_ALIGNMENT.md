# Thí nghiệm căn chỉnh object–region

Thí nghiệm giữ nguyên hai nhánh prompt/ảnh, cross-attention, Gate, decoder và
caption loss. Phần mới lấy patch token CLIP nằm trong từng bounding box YOLO,
mean-pool thành vector vùng rồi phân loại vùng đó bằng các CLIP text prototype
của object label.

```text
L_total = L_caption + lambda * L_region
```

`L_region` là cross-entropy trên cosine similarity giữa region feature và toàn
bộ label prototype. Box nhỏ không chứa tâm patch nào dùng patch gần tâm box nhất.
Confidence YOLO làm trọng số loss; padding region không tham gia loss.

Gate và region loss có vai trò khác nhau: Gate điều tiết lượng thông tin ảnh đưa
vào prompt; region loss trực tiếp ép vùng trong bounding box gần đúng object label.

## Kaggle: tạo region target cache một lần

```python
!python -u build_region_targets.py \
  --source /kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json \
  --dataset-json-path /kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json \
  --base-path /kaggle/input/datasets/vuthetam/mscoco-2014/images \
  --objects-field objects --name-key label --bbox-key bbox --confidence-key conf \
  --max-regions 10 \
  --output /kaggle/working/region_targets_yolo.pt
```

Lưu `region_targets_yolo.pt` thành Kaggle Dataset để những lần train sau không
phải đọc lại kích thước toàn bộ ảnh hay encode label.

## Train ablation đầu tiên

Thêm vào lệnh train hiện tại:

```text
--region-targets-path /kaggle/working/region_targets_yolo.pt
--alignment-weight 0.1
--alignment-temperature 0.07
--max-regions 10
--experiment-name gated_region_align_l01
```

Giữ seed 42, prompt object+relation, visual cache, 10 epoch và decoding giống
bản Gate đối chứng. Không thay confidence threshold trước khi đọc báo cáo audit.
Chạy smoke một epoch trước; log phải hiện riêng `caption` và `align`, cả hai hữu
hạn và giảm hợp lý. So sánh metric với Gate cùng seed, sau đó mới thử lambda 0.05
hoặc 0.2 trên validation; không chọn lambda bằng test.
