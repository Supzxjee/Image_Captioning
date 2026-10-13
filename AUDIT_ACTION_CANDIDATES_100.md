# Kiểm chứng hành động trên cùng cặp đối tượng: 100 pairs

Sau V3, nhãn heuristic riding/skateboard vẫn được verifier chấp nhận dù ảnh cho thấy cậu bé quỳ cạnh ván. Bản audit này cho nhiều hành động cạnh tranh cùng subject/object, thay vì chỉ predicate swap đơn giản hoặc ảnh đối chứng hoàn toàn khác.

- Dùng pretrained CLIP ViT-B/16 với image/text projection trong cùng không gian similarity; không trộn raw EOS cache với image projection.
- Scoring 2 views: crop vùng chứa cả hai instance (padding 15%) và full image. Letterbox giữ đủ hai instance trước CLIP center crop.
- Candidate set theo object class: riding/sitting on/standing on/kneeling on/kneeling beside/carrying/holding/near, hoặc wearing/adjusting/eating/using khi phù hợp. Object class chỉ tạo lựa chọn, không tự gán hành động.
- Chỉ đề xuất predicate khi 2 views cùng chọn, cosine gap top1-top2 >=0.01 ở cả hai views, và near không đứng đầu. Trường hợp khác abstain. Đây là heuristic chưa calibration, không được coi là xác suất relation đúng.
- Một số quan hệ có thể đồng thời đúng (eating và holding, riding và standing on). Không dùng candidate xếp thấp làm negative nhãn thật khi chưa kiểm chứng.
- Selection gồm các action pairs từ 100 ảnh V3 cùng seed42, rồi bổ sung random validation pairs đến100. Audit chỉ xét các cặp đã có trong cache, không phát hiện mới object/pair và chưa thay cache sản xuất.
- Không dùng reference captions để chấm điểm.

## Chạy Kaggle

Một GPU là đủ, bật Internet. Add Input grounded cache V1 và MS COCO. Không cần checkpoint hoặc visual feature cache. Nếu chưa có CLIP model ở local cache, lần đầu tải model sẽ mất thời gian; có thể thay CLIP_MODEL bằng đường dẫn model Hugging Face đã có trong Input.

```python
import json
import os
import subprocess
import sys
import time
from pathlib import Path

GROUNDED_CACHE = Path('/kaggle/input/notebooks/ducanh2403/build-grounded-relation-v1/grounded_relation_v1_full/grounded_relation_cache.pt')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
CLIP_MODEL = 'openai/clip-vit-base-patch16'  # Can be an existing local HF CLIP directory.
REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '2ac5f22'
OUTPUT = Path('/kaggle/working/action_candidate_audit_100')

print('Same-instance action candidate audit bootstrap started.', flush=True)
for path in (GROUNDED_CACHE, COCO_JSON):
    assert path.is_file(), path
assert COCO_IMAGES.is_dir(), COCO_IMAGES

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'

def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    started = time.monotonic()
    with subprocess.Popen(args, cwd=cwd) as process:
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f'Process still running: {time.monotonic()-started:.0f}s', flush=True)
        if code:
            raise subprocess.CalledProcessError(code, args)

REPO.parent.mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    run(['git', 'clone', 'https://github.com/Supzxjee/Image_Captioning.git', REPO])
assert (REPO / '.git').is_dir(), REPO
run(['git', 'fetch', '--depth', '30', 'origin', 'main'], cwd=REPO)
run(['git', 'checkout', '--detach', COMMIT], cwd=REPO)
run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers', 'Pillow'], cwd=REPO)

run([
    sys.executable, '-u', 'audit_action_candidates.py',
    '--grounded-cache', GROUNDED_CACHE,
    '--dataset-json', COCO_JSON, '--images', COCO_IMAGES,
    '--priority-filenames', REPO / 'audit/action_candidate_priority_100.txt',
    '--model', CLIP_MODEL, '--pair-count', '100', '--min-gap', '0.01',
    '--seed', '42', '--output', OUTPUT,
], cwd=REPO)

summary = json.loads((OUTPUT / 'candidate_summary.json').read_text(encoding='utf-8'))
print('\nACTION CANDIDATE AUDIT SUMMARY')
print(json.dumps(summary, indent=2, ensure_ascii=False))
print('Download ZIP:', str(OUTPUT) + '.zip')

```

Output ZIP: `/kaggle/working/action_candidate_audit_100.zip`.

- `review.html`: full image có subject/object boxes, pair crop và bảng score các candidate.
- `candidate_records.json`: lựa chọn, scores, boxes của từng pair.
- `manual_review.csv`: các cột human_old_status, human_proposed_status, human_supported_predicates, notes để đánh giá correct/incorrect/unclear.
- `candidate_summary.json`: decision distribution và số đề xuất khác heuristic cũ, không phải accuracy.

Gửi ZIP và Kaggle code version. Chỉ tạo full relation cache sau khi kiểm tra chất lượng đề xuất và mức abstention. 5 tests candidate/crop/abstention đã PASS; chưa chạy model trên Kaggle. CLIP không được bảo đảm phân biệt chính xác các hành động gần nhau; nếu audit không tốt cần predictor quan hệ được huấn luyện với nhãn phù hợp.

API reference: https://huggingface.co/docs/transformers/model_doc/clip
