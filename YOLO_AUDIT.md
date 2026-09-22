# Kiểm tra thống kê cache YOLO

Mục tiêu của bước này là phát hiện lỗi dữ liệu và chọn một tập ảnh nhỏ cần xem
lại trước khi dùng object label/bounding box cho region alignment hoặc Object
Consistency Checker. Script **không chạy lại YOLO** và không thay đổi cache gốc.

## Chạy trên Kaggle

```python
import os
import subprocess
import sys
from pathlib import Path

REPO = Path('/kaggle/working/Image_Captioning')
DETECTIONS = Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
OUTPUT = Path('/kaggle/working/yolo_audit')

def run(args):
    args = list(map(str, args))
    print('Running:', ' '.join(args), flush=True)
    subprocess.run(args, check=True)

if not REPO.exists():
    run(['git', 'clone', 'https://github.com/Supzxjee/Image_Captioning.git', REPO])
else:
    run(['git', '-C', REPO, 'pull', '--ff-only'])

os.chdir(REPO)
run([
    sys.executable, '-u', 'audit_yolo_detections.py',
    '--source', DETECTIONS,
    '--dataset-json-path', COCO_JSON,
    '--base-path', COCO_IMAGES,
    '--objects-field', 'objects',
    '--name-key', 'label',
    '--bbox-key', 'bbox',
    '--confidence-key', 'conf',
    '--output-dir', OUTPUT,
])
```

`--base-path` làm script đọc kích thước ảnh để kiểm tra box vượt biên và tỉ lệ
diện tích. Bỏ tùy chọn này nếu chỉ muốn kiểm tra nhanh schema, confidence, box
đảo tọa độ và box trùng.

## Ba file kết quả

- `yolo_audit_summary.json`: tổng số ảnh/detection, coverage từng split, phân bố
  confidence, lớp phổ biến và tổng từng loại lỗi.
- `yolo_audit_images.csv`: một dòng cho mỗi ảnh; sắp xếp theo `issues` để lấy
  các trường hợp nghi vấn nhất.
- `yolo_audit_labels.csv`: tần suất từng nhãn.

Các cờ chính:

- `missing_source`: ảnh COCO không có mục trong cache.
- `invalid_bbox`, `nonpositive_bbox`, `out_of_bounds_bbox`: box sai định dạng,
  đảo tọa độ hoặc vượt kích thước ảnh.
- `tiny_bbox`, `near_full_image_bbox`, `extreme_aspect_bbox`: box có hình học
  bất thường.
- `same_label_duplicate_pairs`: hai box cùng lớp có IoU từ 0.8 trở lên.
- `confidence_below_0.25` và `confidence_below_0.50`: detection có độ tin cậy thấp.

## Cách kết luận

Không coi caption COCO là ground truth detection vì caption thường bỏ qua nhiều
đối tượng nhìn thấy. Báo cáo thống kê chỉ phát hiện bất thường, chưa chứng minh
label đúng hay sai. Sau khi có CSV, lấy mẫu phân tầng khoảng 100 ảnh: confidence
thấp, box trùng, box quá nhỏ/lớn, ảnh không có detection và một nhóm bình thường.
Nếu các lỗi tập trung ở một ngưỡng rõ ràng mới thay đổi bộ lọc YOLO; giữ nguyên
cache cho thí nghiệm đối chứng.

Kết quả audit thực tế và quyết định lọc region supervision của đề tài nằm trong
[YOLO_AUDIT_RESULTS.md](YOLO_AUDIT_RESULTS.md).
