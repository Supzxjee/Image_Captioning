# Bộ sinh quan hệ V2: semantic + instance-aware

V2 sửa hai lỗi được xác nhận qua kiểm tra thủ công 120 ảnh COCO:

1. quan hệ `contains/inside` được suy trực tiếp từ hộp bao nên biến hành động hoặc
   quan hệ bề mặt thành câu sai;
2. ID instance bị bỏ trước khi khử xung đột, khiến hai đối tượng cùng lớp tạo câu
   như `surfboard left of person` và `surfboard right of person`.

V2 chỉ dùng ảnh gián tiếp qua YOLO label, confidence và bounding box; không đọc
caption tham chiếu. Mỗi detection giữ `instance_id` trong metadata. Bộ sinh chọn
tối đa một quan hệ cho mỗi cặp instance và một quan hệ mạnh nhất cho mỗi cặp lớp
trước khi chuyển thành prompt chữ.

## Các quan hệ ưu tiên

- người–đồ vật: `holding`, `wearing`, `carrying`, `riding` khi lớp và hình học hỗ trợ;
- vật–bề mặt: `on` cho dining table, bed, couch, bench và chair;
- `inside` chỉ dùng cho whitelist vật chứa hợp lý, ví dụ person/dog trong vehicle
  hoặc food trong bowl/cup;
- nếu không có quan hệ ngữ nghĩa, dùng một quan hệ hình học canonical trong
  `overlapping`, `left of`, `above`, `near`;
- không sinh đồng thời cặp nghịch đảo và không sinh `contains`.

## Chạy smoke trên Kaggle

```python
import json
import subprocess
import sys
from pathlib import Path

REPO = Path('/kaggle/working/Image_Captioning')
DETECTIONS = Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
OUTPUT = Path('/kaggle/working/relation_prompts_v2_smoke')
OUTPUT.mkdir(parents=True, exist_ok=True)

def run(args):
    args = list(map(str, args))
    print('Running:', ' '.join(args), flush=True)
    subprocess.run(args, cwd=REPO, check=True)

run([
    sys.executable, '-u', 'build_relation_prompts.py',
    '--source', DETECTIONS,
    '--output', OUTPUT / 'promptcache_relation_v2.json',
    '--min-confidence', '0.5',
    '--max-objects', '10',
    '--max-relations', '3',
    '--limit', '500',
])

summary = json.loads(
    (OUTPUT / 'promptcache_relation_v2.summary.json').read_text(encoding='utf-8'))
print(json.dumps(summary, indent=2))
```

Kiểm tra thủ công 120 ảnh trên JSON smoke trước. Chỉ khi tỷ lệ `ACCEPT` tăng rõ
so với mốc V1 `61/120` mới tạo cache toàn bộ 123.287 ảnh.

## Tạo JSON và CLIP token cache đầy đủ

```python
OUTPUT = Path('/kaggle/working/relation_prompts_v2_full')
OUTPUT.mkdir(parents=True, exist_ok=True)
PROMPT_JSON = OUTPUT / 'promptcache_relation_v2.json'
PROMPT_TOKENS = OUTPUT / 'prompt_clip_tokens_relation_v2.pt'

run([
    sys.executable, '-u', 'build_relation_prompts.py',
    '--source', DETECTIONS,
    '--output', PROMPT_JSON,
    '--min-confidence', '0.5',
    '--max-objects', '10',
    '--max-relations', '3',
])
run([
    sys.executable, '-u', 'embed_relation_prompts.py',
    '--source', PROMPT_JSON,
    '--dataset-json-path', COCO_JSON,
    '--output', PROMPT_TOKENS,
    '--batch-size', '256',
])
```

Không train ngay sau khi tạo cache. Trước hết chạy lại audit 120 ảnh bằng cùng
seed và cùng danh sách filename của V1. Nếu đạt, thay `--prompt-cache-path` bằng
`prompt_clip_tokens_relation_v2.pt` và chỉ chạy một pilot E2 ba epoch. Mọi cấu
hình khác phải giữ nguyên để đo riêng ảnh hưởng của bộ sinh quan hệ.
