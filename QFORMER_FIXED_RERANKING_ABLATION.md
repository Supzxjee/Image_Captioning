# Ablation: Q-Former cũ + cùng CLIPScore/OCC re-ranking

## Câu hỏi thực nghiệm

Thí nghiệm này tách đóng góp của prompt conditioning khỏi đóng góp của re-ranking.
Nó dùng candidates test đã sinh của Q-Former cũ và áp dụng nguyên trọng số đã khóa
từ validation của Prompt-Conditioned Q-Former:

```text
decoder = 0.3
CLIPScore = 0.4
OCC = 0.3
```

Không train, không sinh caption và không tune lại trọng số. Nếu Q-Former cũ + cùng
re-ranking tốt bằng hoặc tốt hơn pipeline mới, cải thiện đến chủ yếu từ re-ranking.

## Input Kaggle cần gắn

- Output của notebook Q-Former cũ có `test_5000_beam5_candidates.json`;
- ba visual-cache shards;
- `objectdetectioncache.json`.

Có thể gắn đồng thời output của Prompt-Conditioned Q-Former. Cell sẽ đọc metadata
và chỉ chọn candidate bundle có `visual_adapter=qformer` cùng
`prompt_conditioned_qformer=False`. Chọn một GPU và bật Internet.

## Cell Kaggle đầy đủ

```python
import json
import subprocess
import sys
from pathlib import Path

INPUT_ROOT = Path('/kaggle/input')
PROJECT_REPO = Path('/kaggle/working/Image_Captioning')
CHAIR_REPO = Path('/kaggle/working/CHAIR-metric-standalone')
OUTPUT = Path('/kaggle/working/qformer_old_fixed_reranking_ablation')

PROJECT_COMMIT = 'badda5a'
CHAIR_COMMIT = '4087a26211aa2339b9a76307cb8f0321ef691d0a'

VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
DETECTIONS = Path(
    '/kaggle/input/datasets/ducanh2403/'
    'objectdetectionecache/objectdetectioncache.json'
)
CLIP_WEIGHT = 0.4
OBJECT_WEIGHT = 0.3


def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


# 1. Kiểm tra cache dùng cho CLIPScore và OCC.
assert DETECTIONS.is_file(), DETECTIONS
for part in range(1, 4):
    shard = VISUAL_CACHE / f'visual_part_{part:02d}_of_03.h5'
    assert shard.is_file(), shard


# 2. Tìm đúng candidate bundle Q-Former cũ bằng metadata.
candidate_paths = sorted(
    INPUT_ROOT.rglob('test_5000_beam5_candidates.json'), key=str
)
print('\nCandidate bundles:')
matches = []
for path in candidate_paths:
    try:
        bundle = json.loads(path.read_text(encoding='utf-8'))
        metadata = bundle.get('metadata', {})
    except Exception as error:
        print('Bỏ qua file không đọc được:', path, repr(error))
        continue
    profile = {
        'split': metadata.get('split'),
        'count': metadata.get('count'),
        'adapter': metadata.get('visual_adapter'),
        'prompt_conditioned': bool(
            metadata.get('prompt_conditioned_qformer', False)
        ),
    }
    print('-', path, profile)
    if (
        metadata.get('split') == 'test'
        and int(metadata.get('count', -1)) == 5000
        and metadata.get('visual_adapter') == 'qformer'
        and not bool(metadata.get('prompt_conditioned_qformer', False))
    ):
        matches.append(path)

assert len(matches) == 1, (
    f'Cần đúng một candidate bundle Q-Former cũ, tìm thấy {len(matches)}: '
    f'{matches}'
)
RAW_CANDIDATES = matches[0]
GROUND_TRUTH = RAW_CANDIDATES.parent / 'test_5000_gt_candidates.json'
assert GROUND_TRUTH.is_file(), GROUND_TRUTH
print('\nSelected old Q-Former candidates:', RAW_CANDIDATES)
print('Matched ground truth:', GROUND_TRUTH)


# 3. Clone project code cố định.
if not PROJECT_REPO.exists():
    run([
        'git', 'clone',
        'https://github.com/Supzxjee/Image_Captioning.git',
        PROJECT_REPO,
    ])
run(['git', 'fetch', 'origin'], cwd=PROJECT_REPO)
run(['git', 'checkout', '--detach', PROJECT_COMMIT], cwd=PROJECT_REPO)
run(['git', 'rev-parse', '--short', 'HEAD'], cwd=PROJECT_REPO)
run([
    sys.executable, '-m', 'pip', 'install', '-q', '-r',
    PROJECT_REPO / 'requirements.txt',
])


# 4. Tính CLIPScore cho candidates cũ.
OUTPUT.mkdir(parents=True, exist_ok=True)
SCORED_CANDIDATES = OUTPUT / 'test_5000_qformer_old_clipscore_candidates.json'
run([
    sys.executable, '-u', PROJECT_REPO / 'score_clip_candidates.py',
    '--candidates', RAW_CANDIDATES,
    '--visual-cache', VISUAL_CACHE,
    '--output', SCORED_CANDIDATES,
    '--model', 'openai/clip-vit-base-patch16',
    '--batch-size', '64',
    '--device', 'cuda',
])


# 5. Đánh giá rank 0.
BASELINE_DIR = OUTPUT / 'baseline_rank0'
run([
    sys.executable, '-u', PROJECT_REPO / 'rerank_multiscore.py',
    '--candidates', SCORED_CANDIDATES,
    '--ground-truth', GROUND_TRUTH,
    '--detections', DETECTIONS,
    '--output-dir', BASELINE_DIR,
    '--clip-weight', '0.0',
    '--object-weight', '0.0',
    '--min-confidence', '0.5',
])


# 6. Áp dụng cùng fixed weights, không tune trên test.
FINAL_DIR = OUTPUT / 'fixed_d030_c040_o030'
run([
    sys.executable, '-u', PROJECT_REPO / 'rerank_multiscore.py',
    '--candidates', SCORED_CANDIDATES,
    '--ground-truth', GROUND_TRUTH,
    '--detections', DETECTIONS,
    '--output-dir', FINAL_DIR,
    '--clip-weight', CLIP_WEIGHT,
    '--object-weight', OBJECT_WEIGHT,
    '--min-confidence', '0.5',
    '--hallucination-penalty', '1.0',
    '--objects-field', 'objects',
    '--name-key', 'label',
    '--confidence-key', 'conf',
])

baseline_summary = json.loads(
    (BASELINE_DIR / 'test_multiscore_summary.json').read_text(encoding='utf-8')
)
final_summary = json.loads(
    (FINAL_DIR / 'test_multiscore_summary.json').read_text(encoding='utf-8')
)
baseline_result = baseline_summary['results'][0]
final_result = final_summary['results'][0]
assert baseline_result['changed_images'] == 0

# Rank 0 phải tái tạo đúng Q-Former test đã báo cáo.
expected_baseline = {
    'Bleu_1': 0.7673929147791878,
    'Bleu_2': 0.6087332892786202,
    'Bleu_3': 0.47455273221154215,
    'Bleu_4': 0.37066552705750105,
    'METEOR': 0.2850389946676435,
    'ROUGE_L': 0.5731179908462452,
    'CIDEr': 1.1882447460531307,
}
for name, expected in expected_baseline.items():
    actual = baseline_result['metrics'][name]
    assert abs(actual - expected) < 1e-6, (name, actual, expected)
print('\nOld Q-Former rank-0 reproduction PASS')


# 7. Cài evaluator CHAIR cố định.
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


# 8. Chuyển eval_id sang COCO id cho CHAIR.
bundle = json.loads(RAW_CANDIDATES.read_text(encoding='utf-8'))
records = bundle['data']
eval_to_coco = {int(row['image_id']): int(row['coco_id']) for row in records}
baseline_chair_input = [
    {'image_id': int(row['coco_id']), 'caption': row['candidates'][0]['caption']}
    for row in records
]
final_predictions = json.loads(
    Path(final_result['predictions']).read_text(encoding='utf-8')
)
final_chair_input = [
    {
        'image_id': eval_to_coco[int(row['image_id'])],
        'caption': row['caption'],
    }
    for row in final_predictions
]
assert len(baseline_chair_input) == len(final_chair_input) == 5000

baseline_input_path = OUTPUT / 'qformer_old_rank0_chair_input.json'
final_input_path = OUTPUT / 'qformer_old_reranked_chair_input.json'
baseline_input_path.write_text(
    json.dumps(baseline_chair_input, ensure_ascii=False, indent=2), encoding='utf-8'
)
final_input_path.write_text(
    json.dumps(final_chair_input, ensure_ascii=False, indent=2), encoding='utf-8'
)


# 9. CHAIR cho rank 0 và cùng re-ranking.
baseline_chair = OUTPUT / 'qformer_old_rank0_chair_output.json'
final_chair = OUTPUT / 'qformer_old_reranked_chair_output.json'
chair_cache = CHAIR_REPO / 'chair.pkl'
for predictions, result in (
    (baseline_input_path, baseline_chair),
    (final_input_path, final_chair),
):
    run([
        sys.executable, '-u', CHAIR_REPO / 'chair.py',
        '--cap_file', predictions,
        '--image_id_key', 'image_id',
        '--caption_key', 'caption',
        '--cache', chair_cache,
        '--save_path', result,
    ], cwd=CHAIR_REPO)

CHAIR_SUMMARY = OUTPUT / 'qformer_old_chair_comparison.json'
run([
    sys.executable, '-u', PROJECT_REPO / 'summarize_chair_comparison.py',
    '--baseline-chair', baseline_chair,
    '--occ-chair', final_chair,
    '--output', CHAIR_SUMMARY,
])
chair = json.loads(CHAIR_SUMMARY.read_text(encoding='utf-8'))


# 10. Kết quả cần gửi lại.
print('\nQ-FORMER OLD BASELINE METRICS')
print(json.dumps(baseline_result['metrics'], indent=2))
print('\nQ-FORMER OLD + FIXED RE-RANKING METRICS')
print(json.dumps(final_result['metrics'], indent=2))
print('\nChanged images:', final_result['changed_images'])
print('Recognized mentions:', final_result['recognized_mentions'])
print('Supported mentions:', final_result['supported_mentions'])
print('Hallucinated mentions:', final_result['hallucinated_mentions'])
print('\nQ-FORMER OLD CHAIR RANK 0')
print(json.dumps(chair['baseline'], indent=2))
print('\nQ-FORMER OLD CHAIR RE-RANKED')
print(json.dumps(chair['occ'], indent=2))
print('\nCHAIR DELTA')
print(json.dumps(chair['delta_occ_minus_baseline'], indent=2))
print('\nCHAIR TRANSITIONS')
print(json.dumps(chair['sentence_transitions'], indent=2))
print('\nOutput:', OUTPUT)
```

## Quy tắc kết luận

So sánh bốn hàng: Q-Former cũ rank 0, Q-Former cũ re-ranked,
Prompt-Conditioned rank 0 và Prompt-Conditioned re-ranked. Vì cùng fixed weights
được áp dụng trên test, khác biệt giữa hai hàng re-ranked phản ánh ảnh hưởng của
prompt conditioning trong cùng downstream pipeline. Không đổi weights sau kết quả
này.
