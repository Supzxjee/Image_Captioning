# Phân tích định tính Q-Former và E2

Notebook này không train và không inference. Nó dùng hai output test đã lưu để tìm
các ảnh E2 sửa hallucination, làm xấu đi hoặc vẫn còn hallucination; đồng thời tạo
một báo cáo HTML kèm ảnh và năm reference captions.

## Input cần gắn

- Output test Q-Former có CIDEr `1.1882447461`, hoặc output
  `qformer_test_candidates_rebuilt` chứa `test_5000_beam5_candidates.json`;
- output test E2 có CIDEr `1.1947453010`;
- MS COCO 2014 gồm `dataset_coco.json` và thư mục `images`.

Chọn accelerator `None`, bật Internet để tải evaluator CHAIR.

```python
import html
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

INPUT_ROOT = Path('/kaggle/input')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
CHAIR_REPO = Path('/kaggle/working/CHAIR-metric-standalone')
OUTPUT = Path('/kaggle/working/e2_qualitative_analysis')
IMAGE_OUTPUT = OUTPUT / 'images'
CHAIR_COMMIT = '4087a26211aa2339b9a76307cb8f0321ef691d0a'

QFORMER_CIDER = 1.1882447460531307
QFORMER_BLEU4 = 0.37066552705750105
E2_CIDER = 1.194745300990926
E2_BLEU4 = 0.37114510544349455


def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def coco_id(filename):
    match = re.search(r'_(\d{12})\.jpg$', filename)
    if not match:
        raise ValueError(filename)
    return int(match.group(1))


def find_prediction(target_cider, target_bleu4, label, required=True):
    prediction_name = 'test_5000_captions_h1_2_gated.json'
    found = []
    for root, directories, files in os.walk(INPUT_ROOT):
        directories[:] = [name for name in directories
                          if name not in {'images', 'checkpoints', '.git'}]
        if prediction_name not in files:
            continue
        prediction = Path(root) / prediction_name
        metrics_path = prediction.parent / 'test_5000_metrics_h1_2_gated.json'
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        print(label, 'candidate:', prediction, metrics)
        if (abs(float(metrics.get('CIDEr', -1)) - target_cider) < 1e-9 and
                abs(float(metrics.get('Bleu_4', -1)) - target_bleu4) < 1e-9):
            found.append(prediction)
    if not required and not found:
        return None
    assert len(found) == 1, f'{label}: cần đúng một prediction file, tìm thấy {found}'
    return found[0]


def recover_qformer_rank0_from_candidates():
    name = 'test_5000_beam5_candidates.json'
    matches = []
    for root, directories, files in os.walk(INPUT_ROOT):
        directories[:] = [entry for entry in directories
                          if entry not in {'images', 'checkpoints', '.git'}]
        if name not in files:
            continue
        path = Path(root) / name
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            metadata = payload.get('metadata', {})
            records = payload.get('data', [])
        except Exception as error:
            print('Bỏ qua candidate bundle không đọc được:', path, repr(error))
            continue
        signature = {
            'split': metadata.get('split'),
            'count': metadata.get('count'),
            'adapter': metadata.get('visual_adapter'),
            'prompt_conditioned': bool(
                metadata.get('prompt_conditioned_qformer', False)),
            'cascade': bool(metadata.get('cascade_semantic_alignment', False)),
        }
        print('Q-Former candidate bundle:', path, signature)
        if (signature == {
                'split': 'test',
                'count': 5000,
                'adapter': 'qformer',
                'prompt_conditioned': False,
                'cascade': False,
        } and len(records) == 5000 and
                all(row.get('candidates') for row in records)):
            matches.append((path, records))
    assert len(matches) == 1, (
        'Không có caption test Q-Former chuẩn và cũng không tìm thấy đúng một '
        f'candidate bundle baseline: {[path for path, _ in matches]}')
    source, records = matches[0]
    predictions = [
        {
            'image_id': int(row['image_id']),
            'caption': row['candidates'][0]['caption'],
        }
        for row in records
    ]
    destination = OUTPUT / 'qformer_rank0_recovered_predictions.json'
    destination.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Khôi phục Q-Former rank-0 từ:', source)
    print('Predictions tạm:', destination)
    return destination


def prepare_chair_input(prediction_path, eval_to_coco, output_path):
    predictions = json.loads(prediction_path.read_text(encoding='utf-8'))
    assert len(predictions) == 5000
    by_eval = {}
    for item in predictions:
        eval_id = int(item['image_id'])
        caption = item.get('caption')
        assert eval_id in eval_to_coco
        assert isinstance(caption, str) and caption.strip()
        assert eval_id not in by_eval
        by_eval[eval_id] = caption
    assert set(by_eval) == set(eval_to_coco)
    payload = [
        {'image_id': eval_to_coco[eval_id], 'caption': by_eval[eval_id]}
        for eval_id in sorted(by_eval)
    ]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding='utf-8')


assert COCO_JSON.is_file(), COCO_JSON
assert COCO_IMAGES.is_dir(), COCO_IMAGES
OUTPUT.mkdir(parents=True, exist_ok=True)
IMAGE_OUTPUT.mkdir(parents=True, exist_ok=True)

QFORMER_PREDICTIONS = find_prediction(
    QFORMER_CIDER, QFORMER_BLEU4, 'Q-Former', required=False)
if QFORMER_PREDICTIONS is None:
    QFORMER_PREDICTIONS = recover_qformer_rank0_from_candidates()
E2_PREDICTIONS = find_prediction(E2_CIDER, E2_BLEU4, 'E2')
print('Q-Former predictions:', QFORMER_PREDICTIONS)
print('E2 predictions:', E2_PREDICTIONS)


# 1. Ánh xạ global eval_id sang COCO id và metadata ảnh test.
coco_data = json.loads(COCO_JSON.read_text(encoding='utf-8'))
test_records = {}
eval_to_coco = {}
for eval_id, image in enumerate(coco_data['images']):
    if image['split'] != 'test':
        continue
    image_coco_id = coco_id(image['filename'])
    eval_to_coco[eval_id] = image_coco_id
    test_records[image_coco_id] = {
        'eval_id': eval_id,
        'coco_id': image_coco_id,
        'filename': image['filename'],
        'filepath': image['filepath'],
        'references': [sentence['raw'] for sentence in image['sentences'][:5]],
    }
assert len(test_records) == 5000


# 2. Chuyển cả hai prediction file sang COCO ids.
QFORMER_INPUT = OUTPUT / 'qformer_chair_input.json'
E2_INPUT = OUTPUT / 'e2_chair_input.json'
prepare_chair_input(QFORMER_PREDICTIONS, eval_to_coco, QFORMER_INPUT)
prepare_chair_input(E2_PREDICTIONS, eval_to_coco, E2_INPUT)


# 3. Chạy cùng một evaluator CHAIR cho hai mô hình.
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
run([sys.executable, '-c',
     "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"])

CHAIR_CACHE = CHAIR_REPO / 'chair.pkl'
QFORMER_CHAIR = OUTPUT / 'qformer_chair_output.json'
E2_CHAIR = OUTPUT / 'e2_chair_output.json'
for source, destination in ((QFORMER_INPUT, QFORMER_CHAIR),
                            (E2_INPUT, E2_CHAIR)):
    run([
        sys.executable, '-u', CHAIR_REPO / 'chair.py',
        '--cap_file', source,
        '--image_id_key', 'image_id',
        '--caption_key', 'caption',
        '--cache', CHAIR_CACHE,
        '--save_path', destination,
    ], cwd=CHAIR_REPO)


# 4. Ghép kết quả theo ảnh và phân loại transition.
def load_sentences(path):
    payload = json.loads(path.read_text(encoding='utf-8'))
    return payload['overall_metrics'], {
        int(item['image_id']): item for item in payload['sentences']
    }


q_metrics, q_sentences = load_sentences(QFORMER_CHAIR)
e2_metrics, e2_sentences = load_sentences(E2_CHAIR)
assert set(q_sentences) == set(e2_sentences) == set(test_records)


def hallucinated_words(sentence):
    return list(sentence.get('mscoco_hallucinated_words', []))


def transition(before, after):
    before_h = bool(before['metrics']['CHAIRs'])
    after_h = bool(after['metrics']['CHAIRs'])
    if before_h and not after_h:
        return 'improved_to_clean'
    if not before_h and after_h:
        return 'regressed_to_hallucinated'
    if before_h and after_h:
        return 'both_hallucinated'
    return 'both_clean'


records = []
for image_id in sorted(test_records):
    before, after = q_sentences[image_id], e2_sentences[image_id]
    base = dict(test_records[image_id])
    base.update({
        'transition': transition(before, after),
        'qformer_caption': before.get('caption'),
        'e2_caption': after.get('caption'),
        'qformer_hallucinated_words': hallucinated_words(before),
        'e2_hallucinated_words': hallucinated_words(after),
    })
    records.append(base)

counts = Counter(item['transition'] for item in records)
print('Transitions:', dict(counts))


# 5. Xuất nhiều candidate để chọn thủ công, không chỉ giữ ví dụ có lợi cho E2.
groups = {
    name: [item for item in records if item['transition'] == name]
    for name in ('improved_to_clean', 'regressed_to_hallucinated',
                 'both_hallucinated', 'both_clean')
}
groups['improved_to_clean'].sort(
    key=lambda item: (-len(item['qformer_hallucinated_words']), item['coco_id']))
groups['regressed_to_hallucinated'].sort(
    key=lambda item: (-len(item['e2_hallucinated_words']), item['coco_id']))
groups['both_hallucinated'].sort(
    key=lambda item: (len(item['e2_hallucinated_words']) -
                      len(item['qformer_hallucinated_words']), item['coco_id']))
groups['both_clean'] = [item for item in groups['both_clean']
                        if item['qformer_caption'] != item['e2_caption']]

selected = (
    groups['improved_to_clean'][:8] +
    groups['regressed_to_hallucinated'][:5] +
    groups['both_hallucinated'][:5] +
    groups['both_clean'][:5]
)

for item in selected:
    source = COCO_IMAGES / item['filepath'] / item['filename']
    assert source.is_file(), source
    shutil.copy2(source, IMAGE_OUTPUT / item['filename'])

summary = {
    'count': len(records),
    'qformer_metrics': q_metrics,
    'e2_metrics': e2_metrics,
    'transitions': dict(counts),
    'selection': {
        'improved_to_clean': min(8, len(groups['improved_to_clean'])),
        'regressed_to_hallucinated': min(5, len(groups['regressed_to_hallucinated'])),
        'both_hallucinated': min(5, len(groups['both_hallucinated'])),
        'both_clean_changed': min(5, len(groups['both_clean'])),
    },
}
(OUTPUT / 'transition_summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
(OUTPUT / 'qualitative_candidates.json').write_text(
    json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')


# 6. Tạo báo cáo HTML xem trực tiếp trong Output.
cards = []
for item in selected:
    references = ''.join(f'<li>{html.escape(text)}</li>'
                         for text in item['references'])
    cards.append(f'''
    <article class="card {html.escape(item['transition'])}">
      <h2>{html.escape(item['transition'])} — COCO {item['coco_id']}</h2>
      <img src="images/{html.escape(item['filename'])}" alt="COCO image">
      <p><b>Q-Former:</b> {html.escape(item['qformer_caption'])}</p>
      <p><b>Q hallucinated:</b> {html.escape(str(item['qformer_hallucinated_words']))}</p>
      <p><b>E2:</b> {html.escape(item['e2_caption'])}</p>
      <p><b>E2 hallucinated:</b> {html.escape(str(item['e2_hallucinated_words']))}</p>
      <details><summary>5 reference captions</summary><ol>{references}</ol></details>
    </article>''')

report = f'''<!doctype html>
<html><head><meta charset="utf-8"><title>E2 qualitative analysis</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1100px;margin:24px auto;background:#f5f7fb}}
.summary,.card{{background:white;padding:18px;margin:16px 0;border-radius:12px}}
.card img{{max-width:520px;max-height:380px;display:block;margin:12px 0}}
.improved_to_clean{{border-left:8px solid #2e8b57}}
.regressed_to_hallucinated{{border-left:8px solid #c0392b}}
.both_hallucinated{{border-left:8px solid #d68910}}
.both_clean{{border-left:8px solid #2874a6}}
</style></head><body>
<section class="summary"><h1>E2 qualitative analysis</h1>
<pre>{html.escape(json.dumps(summary, indent=2))}</pre></section>
{''.join(cards)}
</body></html>'''
(OUTPUT / 'qualitative_report.html').write_text(report, encoding='utf-8')

print('\nHoàn tất')
print('Summary:', OUTPUT / 'transition_summary.json')
print('Candidates:', OUTPUT / 'qualitative_candidates.json')
print('HTML report:', OUTPUT / 'qualitative_report.html')

# Gợi ý viết báo cáo:
# - chọn khoảng 4 ví dụ improved_to_clean;
# - chọn 1 đến 2 ví dụ regressed_to_hallucinated;
# - chọn 1 ví dụ both_hallucinated;
# - đọc 5 reference captions và quan sát ảnh trước khi nhận xét;
# - không kết luận E2 tốt hơn chỉ dựa vào độ dài caption.
```
