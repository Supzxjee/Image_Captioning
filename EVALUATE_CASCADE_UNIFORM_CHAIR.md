# Đánh giá CHAIR cho E2 không Soft Selector

Cell này sử dụng caption test uniform đã sinh, không inference và không train lại.
Add Input output của notebook `e2_no_soft_selector_32q_2l_test` và MS COCO 2014.
Chọn accelerator `None`, bật Internet để clone evaluator.

```python
import json
import os
import re
import subprocess
import sys
from pathlib import Path

CHAIR_REPO = Path('/kaggle/working/CHAIR-metric-standalone')
OUTPUT = Path('/kaggle/working/chair_e2_no_soft_selector_test')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
CHAIR_COMMIT = '4087a26211aa2339b9a76307cb8f0321ef691d0a'


def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def coco_id(filename):
    match = re.search(r'_(\d{12})\.jpg$', filename)
    if not match:
        raise ValueError(f'Không đọc được COCO id: {filename}')
    return int(match.group(1))


# 1. Chọn caption bằng manifest do notebook uniform test tạo ra.
manifests = []
for search_root in ('/kaggle/input', '/kaggle/working'):
    for root, directories, files in os.walk(search_root):
        directories[:] = [name for name in directories
                          if name not in {'images', 'checkpoints', '.git'}]
        if 'uniform_test_manifest.json' in files:
            manifests.append(Path(root) / 'uniform_test_manifest.json')

candidates = []
print('Uniform manifests:')
for manifest_path in sorted(manifests, key=str):
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except Exception as error:
        print(' -', manifest_path, '| bỏ qua:', repr(error))
        continue
    prediction_path = manifest_path.parent / manifest.get('predictions', '')
    print(' -', manifest_path, '| selector =', manifest.get('cascade_selector_mode'),
          '| predictions =', prediction_path)
    if (manifest.get('model') == 'e2_no_soft_selector'
            and manifest.get('cascade_selector_mode') == 'uniform'
            and prediction_path.is_file()):
        candidates.append(prediction_path)

assert len(candidates) == 1, (
    f'Cần đúng một caption file E2 uniform, tìm thấy {len(candidates)}: {candidates}')
PREDICTIONS = candidates[0]
print('Selected uniform predictions:', PREDICTIONS)
assert COCO_JSON.is_file(), COCO_JSON


# 2. Đổi eval_id sang COCO image id cho CHAIR.
predictions = json.loads(PREDICTIONS.read_text(encoding='utf-8'))
coco_data = json.loads(COCO_JSON.read_text(encoding='utf-8'))
test_id_map = {
    eval_id: coco_id(image['filename'])
    for eval_id, image in enumerate(coco_data['images'])
    if image['split'] == 'test'
}
assert len(predictions) == 5000
assert len(test_id_map) == 5000
prediction_ids = [int(item['image_id']) for item in predictions]
assert len(set(prediction_ids)) == 5000
assert set(prediction_ids) == set(test_id_map)

chair_input = [
    {
        'image_id': test_id_map[int(item['image_id'])],
        'caption': item['caption'],
    }
    for item in predictions
]
assert all(isinstance(item['caption'], str) and item['caption'].strip()
           for item in chair_input)

OUTPUT.mkdir(parents=True, exist_ok=True)
CHAIR_INPUT = OUTPUT / 'e2_no_soft_selector_chair_input.json'
CHAIR_RESULT = OUTPUT / 'e2_no_soft_selector_chair_output.json'
CHAIR_INPUT.write_text(
    json.dumps(chair_input, ensure_ascii=False, indent=2), encoding='utf-8')


# 3. Cố định phiên bản evaluator CHAIR.
if not CHAIR_REPO.exists():
    run(['git', 'clone',
         'https://github.com/Maxlinn/CHAIR-metric-standalone.git', CHAIR_REPO])
run(['git', 'fetch', 'origin'], cwd=CHAIR_REPO)
run(['git', 'checkout', '--detach', CHAIR_COMMIT], cwd=CHAIR_REPO)
run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85',
    'nltk', 'tqdm',
])
run([
    sys.executable, '-c',
    "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')",
])


# 4. Chạy CHAIR trên đúng 5.000 caption uniform.
CHAIR_CACHE = CHAIR_REPO / 'chair.pkl'
assert CHAIR_CACHE.is_file(), CHAIR_CACHE
run([
    sys.executable, '-u', CHAIR_REPO / 'chair.py',
    '--cap_file', CHAIR_INPUT,
    '--image_id_key', 'image_id',
    '--caption_key', 'caption',
    '--cache', CHAIR_CACHE,
    '--save_path', CHAIR_RESULT,
], cwd=CHAIR_REPO)

result = json.loads(CHAIR_RESULT.read_text(encoding='utf-8'))
metrics = result['overall_metrics']
assert len(result['sentences']) == 5000
print('\nE2 WITHOUT SOFT SELECTOR CHAIR TEST')
print(json.dumps({
    'CHAIRs': float(metrics['CHAIRs']),
    'CHAIRi': float(metrics['CHAIRi']),
    'Recall': float(metrics['Recall']),
}, indent=2))
print('Result:', CHAIR_RESULT)
```

Chỉ so sánh với E2 đầy đủ sau khi cả caption metrics và CHAIR của uniform đã hoàn
thành. Nếu uniform giữ lợi thế caption nhưng CHAIR xấu hơn rõ rệt, không thay E2
đầy đủ chỉ dựa trên CIDEr validation.

