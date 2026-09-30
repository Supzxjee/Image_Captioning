# CHAIR test: Prompt-Conditioned Q-Former trước và sau re-ranking

Đây là đánh giá cuối cùng cho object hallucination. Nó dùng cùng 5.000 ảnh test và
so sánh rank 0 với caption sau re-ranking cố định `decoder=0.3, CLIP=0.4, OCC=0.3`.
Không train, không sinh caption và không tune trọng số.

Gắn Output của notebook test cố định, bật Internet, chọn Accelerator `None`, rồi
chạy cell sau:

```python
import json
import subprocess
import sys
from pathlib import Path

PROJECT_REPO = Path('/kaggle/working/Image_Captioning')
CHAIR_REPO = Path('/kaggle/working/CHAIR-metric-standalone')
OUTPUT = Path('/kaggle/working/prompt_conditioned_qformer_chair_test')
INPUT_ROOT = Path('/kaggle/input')

PROJECT_COMMIT = 'badda5a'
CHAIR_COMMIT = '4087a26211aa2339b9a76307cb8f0321ef691d0a'


def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def exactly_one(name):
    matches = sorted(INPUT_ROOT.rglob(name), key=str)
    print(name, matches)
    assert len(matches) == 1, (name, matches)
    return matches[0]


# 1. Dùng lại output test đã sinh.
CANDIDATES = exactly_one('test_5000_beam5_candidates.json')
FINAL_PREDICTIONS = exactly_one(
    'test_5000_multi_d0p300_c0p400_o0p300_predictions.json'
)


# 2. Clone code cố định.
if not PROJECT_REPO.exists():
    run([
        'git', 'clone',
        'https://github.com/Supzxjee/Image_Captioning.git',
        PROJECT_REPO,
    ])
run(['git', 'fetch', 'origin'], cwd=PROJECT_REPO)
run(['git', 'checkout', '--detach', PROJECT_COMMIT], cwd=PROJECT_REPO)

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


# 3. Chuyển eval_id sang COCO image id thật cho cả hai hệ thống.
bundle = json.loads(CANDIDATES.read_text(encoding='utf-8'))
records = bundle['data']
assert bundle['metadata']['count'] == 5000
eval_to_coco = {int(row['image_id']): int(row['coco_id']) for row in records}
assert len(eval_to_coco) == 5000

baseline = [
    {
        'image_id': int(row['coco_id']),
        'caption': row['candidates'][0]['caption'],
    }
    for row in records
]
final_source = json.loads(FINAL_PREDICTIONS.read_text(encoding='utf-8'))
assert len(final_source) == 5000
final = [
    {
        'image_id': eval_to_coco[int(row['image_id'])],
        'caption': row['caption'],
    }
    for row in final_source
]
assert {row['image_id'] for row in baseline} == {row['image_id'] for row in final}

OUTPUT.mkdir(parents=True, exist_ok=True)
baseline_input = OUTPUT / 'prompt_qformer_rank0_chair_input.json'
final_input = OUTPUT / 'prompt_qformer_reranked_chair_input.json'
baseline_input.write_text(json.dumps(baseline, ensure_ascii=False, indent=2),
                          encoding='utf-8')
final_input.write_text(json.dumps(final, ensure_ascii=False, indent=2),
                       encoding='utf-8')
changed = sum(a['caption'] != b['caption'] for a, b in zip(baseline, final))
assert changed == 2885, changed
print('Changed captions:', changed)


# 4. Dùng cùng evaluator và ground truth CHAIR cho hai prediction files.
baseline_chair = OUTPUT / 'prompt_qformer_rank0_chair_output.json'
final_chair = OUTPUT / 'prompt_qformer_reranked_chair_output.json'
chair_cache = CHAIR_REPO / 'chair.pkl'
assert chair_cache.is_file(), chair_cache

for predictions, result in (
    (baseline_input, baseline_chair),
    (final_input, final_chair),
):
    run([
        sys.executable, '-u', CHAIR_REPO / 'chair.py',
        '--cap_file', predictions,
        '--image_id_key', 'image_id',
        '--caption_key', 'caption',
        '--cache', chair_cache,
        '--save_path', result,
    ], cwd=CHAIR_REPO)


# 5. So sánh theo từng ảnh.
summary_path = OUTPUT / 'prompt_qformer_chair_comparison.json'
run([
    sys.executable, '-u', PROJECT_REPO / 'summarize_chair_comparison.py',
    '--baseline-chair', baseline_chair,
    '--occ-chair', final_chair,
    '--output', summary_path,
])

summary = json.loads(summary_path.read_text(encoding='utf-8'))
assert summary['count'] == 5000
assert summary['changed_captions'] == 2885
print('\nCHAIR RANK 0')
print(json.dumps(summary['baseline'], indent=2))
print('\nCHAIR FIXED RE-RANKING')
print(json.dumps(summary['occ'], indent=2))
print('\nDELTA FINAL - RANK 0')
print(json.dumps(summary['delta_occ_minus_baseline'], indent=2))
print('\nSENTENCE TRANSITIONS')
print(json.dumps(summary['sentence_transitions'], indent=2))
print('Summary:', summary_path)
```

`CHAIRs` và `CHAIRi` càng thấp càng tốt; `Recall` càng cao càng tốt. Báo cáo cả
`sentence_transitions`, đặc biệt so sánh `improved_to_clean` với
`regressed_to_hallucinated`.
