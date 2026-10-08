# Đánh giá CHAIR cho E2 Cascade Alignment

Cell này dùng caption test E2 đã sinh; không inference và không train lại. Add Input
output của notebook `qformer_cascade_alignment_32q_2l_test` và MS COCO 2014. Chọn
accelerator `None`, bật Internet để clone evaluator.

```python
import json
import os
import re
import subprocess
import sys
from pathlib import Path

CHAIR_REPO = Path('/kaggle/working/CHAIR-metric-standalone')
OUTPUT = Path('/kaggle/working/chair_e2_cascade_test')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
CHAIR_COMMIT = '4087a26211aa2339b9a76307cb8f0321ef691d0a'

# Nếu tự dò không thấy, bấm Copy Path tại file caption E2 rồi dán vào đây.
# Ví dụ: '/kaggle/input/.../test_5000_captions_h1_2_gated.json'
PREDICTIONS_PATH = ''


def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def coco_id(filename):
    match = re.search(r'_(\d{12})\.jpg$', filename)
    if not match:
        raise ValueError(f'Không đọc được COCO id từ filename: {filename}')
    return int(match.group(1))


# 1. Chọn đúng caption test E2 bằng metric đi kèm. Kaggle có thể đổi tên thư
# mục mount, nên không dựa vào tên experiment trong đường dẫn.
prediction_name = 'test_5000_captions_h1_2_gated.json'
all_predictions = []
if PREDICTIONS_PATH:
    direct_path = Path(PREDICTIONS_PATH)
    assert direct_path.is_file(), direct_path
    all_predictions = [direct_path]
else:
    for search_root in ('/kaggle/input', '/kaggle/working'):
        for root, directories, files in os.walk(search_root):
            # COCO's image folders contain >120k files and can make a recursive
            # search take many minutes. No evaluation JSON is stored below them.
            directories[:] = [name for name in directories
                              if name not in {'images', 'checkpoints', '.git'}]
            if prediction_name in files:
                all_predictions.append(Path(root) / prediction_name)
all_predictions.sort(key=str)
candidates = []
print('Test prediction files:')
for path in all_predictions:
    metrics_path = path.parent / 'test_5000_metrics_h1_2_gated.json'
    metrics = None
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        except Exception as error:
            print('-', path, '| metrics không đọc được:', repr(error))
            continue
    print('-', path, '| metrics =', metrics)
    if (isinstance(metrics, dict)
            and abs(float(metrics.get('CIDEr', -1)) - 1.194745300990926) < 1e-9
            and abs(float(metrics.get('Bleu_4', -1)) - 0.37114510544349455) < 1e-9):
        candidates.append(path)

assert len(candidates) == 1, (
    f'Cần đúng một caption file có metric E2, tìm thấy {len(candidates)}: '
    f'{candidates}. Hãy Add Input output của notebook E2 test đã hoàn thành.')
PREDICTIONS = candidates[0]
print('Selected E2 predictions:', PREDICTIONS)
assert COCO_JSON.is_file(), COCO_JSON


# 2. Đổi eval_id trong prediction sang COCO image id mà CHAIR yêu cầu.
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
assert set(prediction_ids) == set(test_id_map), (
    'Prediction ids không khớp đúng test split trong dataset_coco.json')

chair_input = []
for item in predictions:
    caption = item.get('caption')
    assert isinstance(caption, str) and caption.strip(), item
    chair_input.append({
        'image_id': test_id_map[int(item['image_id'])],
        'caption': caption,
    })
assert len({item['image_id'] for item in chair_input}) == 5000

OUTPUT.mkdir(parents=True, exist_ok=True)
CHAIR_INPUT = OUTPUT / 'e2_cascade_chair_input.json'
CHAIR_RESULT = OUTPUT / 'e2_cascade_chair_output.json'
CHAIR_INPUT.write_text(
    json.dumps(chair_input, ensure_ascii=False, indent=2), encoding='utf-8')
print('CHAIR input:', CHAIR_INPUT)


# 3. Clone evaluator CHAIR standalone cố định phiên bản.
if not CHAIR_REPO.exists():
    run([
        'git', 'clone',
        'https://github.com/Maxlinn/CHAIR-metric-standalone.git',
        CHAIR_REPO,
    ])
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


# 4. Chạy CHAIR trên đúng 5.000 caption E2.
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
print('\nE2 CASCADE CHAIR TEST')
print(json.dumps({
    'CHAIRs': float(metrics['CHAIRs']),
    'CHAIRi': float(metrics['CHAIRi']),
    'Recall': float(metrics['Recall']),
}, indent=2))
print('Result:', CHAIR_RESULT)
```

So sánh kết quả với hai mốc test đã có:

| Mô hình | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | CIDEr ↑ |
|---|---:|---:|---:|---:|
| ViT-direct + Gate | 0.0490 | 0.034464 | 0.436428 | 1.1820 |
| Q-Former | 0.0432 | 0.030299 | 0.437508 | 1.188245 |
| Q-Former + E2 | kết quả mới | kết quả mới | kết quả mới | 1.194745 |
