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

- Output của notebook Q-Former cũ có một trong hai file
  `test_5000_beam5_candidates.json` hoặc
  `test_5000_beam5_clipscore_candidates.json`;
- ba visual-cache shards;
- `objectdetectioncache.json`.

Có thể gắn đồng thời output của Prompt-Conditioned Q-Former. Cell sẽ đọc metadata
và chỉ chọn candidate bundle có `visual_adapter=qformer` cùng
`prompt_conditioned_qformer=False`. Chọn một GPU và bật Internet. Chỉ checkpoint
hoặc file caption rank 0 là chưa đủ, vì re-ranking cần toàn bộ năm beam candidates
của từng ảnh.

## Cell Kaggle đầy đủ

```python
import json
import os
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

# Nếu có nhiều output Q-Former trong Input, điền đường dẫn file cần dùng tại đây.
# Chấp nhận file raw hoặc file đã chấm CLIPScore. Bình thường để None.
CANDIDATES_OVERRIDE = None


def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


# 1. Kiểm tra cache dùng cho CLIPScore và OCC.
assert DETECTIONS.is_file(), DETECTIONS
for part in range(1, 4):
    shard = VISUAL_CACHE / f'visual_part_{part:02d}_of_03.h5'
    assert shard.is_file(), shard


# 2. Tìm candidate bundle mà không quét 123 nghìn ảnh COCO.
# Chấp nhận cả raw và file đã có CLIPScore từ một Save Version cũ.
skip_directories = {
    'images', 'checkpoints', 'visual-cache', 'visual_cache',
    'prompt-cache', 'prompt_cache', 'Image_Captioning', '__pycache__',
}
candidate_paths = []
if CANDIDATES_OVERRIDE:
    candidate_paths = [Path(CANDIDATES_OVERRIDE)]
else:
    for root, directories, files in os.walk(INPUT_ROOT):
        directories[:] = [
            name for name in directories if name not in skip_directories
        ]
        for filename in files:
            lower = filename.lower()
            if (
                lower.endswith('.json')
                and 'test' in lower
                and 'candidate' in lower
                and 'gt_candidate' not in lower
            ):
                candidate_paths.append(Path(root) / filename)
candidate_paths = sorted(set(candidate_paths), key=str)

print('\nCác file test candidate tìm thấy trong Input:')
if not candidate_paths:
    print('  KHÔNG CÓ FILE NÀO')

matches = []
for path in candidate_paths:
    try:
        bundle = json.loads(path.read_text(encoding='utf-8'))
        metadata = bundle.get('metadata', {})
        records = bundle.get('data', [])
    except Exception as error:
        print('Bỏ qua file không đọc được:', path, repr(error))
        continue
    first = records[0] if records else {}
    is_bundle = (
        len(records) == 5000
        and isinstance(first.get('candidates'), list)
        and len(first.get('candidates', [])) > 0
    )
    adapter = metadata.get('visual_adapter')
    path_hint = str(path).lower()
    prompt_conditioned = bool(
        metadata.get('prompt_conditioned_qformer', False)
    )
    profile = {
        'split': metadata.get('split'),
        'metadata_count': metadata.get('count'),
        'actual_count': len(records),
        'adapter': adapter,
        'prompt_conditioned': prompt_conditioned,
        'has_clipscore': bool(
            first.get('candidates')
            and 'clipscore' in first['candidates'][0]
        ),
    }
    print('-', path, profile)
    if (
        is_bundle
        and metadata.get('split', 'test') == 'test'
        and (adapter == 'qformer' or 'qformer' in path_hint)
        and not prompt_conditioned
        and 'prompt_conditioned' not in path_hint
    ):
        matches.append(path)

if not candidate_paths:
    raise FileNotFoundError(
        'Input hiện tại không chứa beam candidates test. Hãy Add Input output '
        'của notebook Q-Former cũ có test_5000_beam5_candidates.json hoặc '
        'test_5000_beam5_clipscore_candidates.json. Checkpoint/predictions rank-0 '
        'không đủ để re-rank.'
    )
# Một Save Version phục hồi có thể chứa cả raw và scored bundle. Khi đó ưu tiên
# scored bundle duy nhất để không chạy CLIP lại.
if len(matches) > 1:
    scored_matches = []
    for path in matches:
        payload = json.loads(path.read_text(encoding='utf-8'))
        candidate = payload['data'][0]['candidates'][0]
        if 'clipscore' in candidate:
            scored_matches.append(path)
    if len(scored_matches) == 1:
        print('Có cả raw và scored bundle; tự chọn scored bundle.')
        matches = scored_matches

if len(matches) != 1:
    raise RuntimeError(
        f'Cần đúng một candidate bundle Q-Former cũ, nhận diện được {len(matches)}: '
        f'{matches}. Nếu có nhiều file, gán CANDIDATES_OVERRIDE bằng đường dẫn '
        'file Q-Former cũ cần dùng.'
    )
SOURCE_CANDIDATES = matches[0]

# Ground truth thường nằm cạnh raw candidates; nếu file source ở thư mục
# reranking thì tìm trong cùng cây output notebook.
ground_truth_matches = sorted(
    SOURCE_CANDIDATES.parent.rglob('test_5000_gt_candidates.json'), key=str
)
if not ground_truth_matches:
    for parent in SOURCE_CANDIDATES.parents:
        if parent == INPUT_ROOT.parent:
            break
        ground_truth_matches.extend(
            sorted(parent.glob('**/test_5000_gt_candidates.json'), key=str)
        )
        if ground_truth_matches:
            break
ground_truth_matches = sorted(set(ground_truth_matches), key=str)
if len(ground_truth_matches) != 1:
    raise RuntimeError(
        f'Không xác định duy nhất ground truth đi kèm {SOURCE_CANDIDATES}: '
        f'{ground_truth_matches}'
    )
GROUND_TRUTH = ground_truth_matches[0]
assert GROUND_TRUTH.is_file(), GROUND_TRUTH
print('\nSelected old Q-Former candidates:', SOURCE_CANDIDATES)
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


# 4. Tính CLIPScore nếu source chưa có; nếu đã có thì dùng lại trực tiếp.
OUTPUT.mkdir(parents=True, exist_ok=True)
source_bundle = json.loads(SOURCE_CANDIDATES.read_text(encoding='utf-8'))
source_first_candidate = source_bundle['data'][0]['candidates'][0]
if 'clipscore' in source_first_candidate:
    SCORED_CANDIDATES = SOURCE_CANDIDATES
    print('Dùng lại CLIPScore có sẵn:', SCORED_CANDIDATES)
else:
    SCORED_CANDIDATES = (
        OUTPUT / 'test_5000_qformer_old_clipscore_candidates.json'
    )
    run([
        sys.executable, '-u', PROJECT_REPO / 'score_clip_candidates.py',
        '--candidates', SOURCE_CANDIDATES,
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
bundle = json.loads(SOURCE_CANDIDATES.read_text(encoding='utf-8'))
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

## Kết quả test 5.000 ảnh

Bộ trọng số `decoder=0.3`, `CLIPScore=0.4`, `OCC=0.3` được giữ nguyên từ
validation của Prompt-Conditioned Q-Former và áp dụng trực tiếp, không tune lại
trên test của Q-Former cũ.

| Mô hình | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q-Former cũ, rank 0 | 0.767393 | 0.608733 | 0.474553 | **0.370666** | 0.285039 | 0.573118 | 1.188245 |
| Q-Former cũ + fixed re-ranking | **0.772729** | **0.615227** | **0.478446** | 0.369356 | **0.290791** | **0.577798** | **1.221815** |
| Prompt-Conditioned Q-Former, rank 0 | 0.759712 | 0.598945 | 0.463337 | 0.359790 | 0.281008 | 0.566814 | 1.151742 |
| Prompt-Conditioned + fixed re-ranking | 0.762326 | 0.603126 | 0.464612 | 0.356083 | 0.287345 | 0.571062 | 1.188570 |

So với rank 0 của chính Q-Former cũ, re-ranking thay 2.941/5.000 caption
(`58,82%`) và thay đổi metric như sau:

| Metric | Chênh lệch re-ranked − rank 0 |
|---|---:|
| BLEU-1 | +0.005336 |
| BLEU-2 | +0.006493 |
| BLEU-3 | +0.003893 |
| BLEU-4 | -0.001310 |
| METEOR | +0.005752 |
| ROUGE-L | +0.004680 |
| CIDEr | +0.033570 |

Re-ranking làm giảm nhẹ BLEU-4 nhưng cải thiện rõ CIDEr, METEOR, ROUGE-L và
BLEU-1/2/3. Caption sau re-ranking có 7.086 object mentions được nhận diện, gồm
6.552 mentions được YOLO hỗ trợ và 534 mentions bị nghi ngờ (`7,54%`).

## Kết quả CHAIR

| Mô hình | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ |
|---|---:|---:|---:|
| Q-Former cũ, rank 0 | 0.0432 | 0.030299 | 0.437508 |
| Q-Former cũ + fixed re-ranking | **0.0348** | **0.023712** | **0.453131** |
| Prompt-Conditioned, rank 0 | 0.0490 | 0.034464 | 0.436428 |
| Prompt-Conditioned + fixed re-ranking | 0.0364 | 0.024651 | 0.451924 |

Trên Q-Former cũ, fixed re-ranking giảm CHAIRs tuyệt đối `0.0084`, tương đối
`19,44%`; giảm CHAIRi tuyệt đối `0.006587`, tương đối `21,74%`; đồng thời tăng
Recall `0.015623`. Có 82 caption chuyển từ hallucinated sang clean và 40 caption
chuyển theo chiều ngược lại, tức giảm ròng 42 caption hallucinated. Các trường hợp
còn lại gồm 4.744 caption đều clean và 134 caption vẫn hallucinated ở cả hai đầu.

## Kết luận ablation

Với cùng re-ranking cố định, Q-Former cũ cao hơn Prompt-Conditioned Q-Former ở
cả bảy metric caption: `+0.010403` BLEU-1, `+0.012101` BLEU-2, `+0.013834`
BLEU-3, `+0.013273` BLEU-4, `+0.003446` METEOR, `+0.006735` ROUGE-L và
`+0.033246` CIDEr. Nó cũng có CHAIRs/CHAIRi thấp hơn và Recall cao hơn.

Do đó, cải thiện của pipeline cuối đến từ cơ chế chọn lại candidates bằng
CLIPScore và OCC. Cách đưa prompt trực tiếp vào Q-Former hiện tại làm giảm chất
lượng visual queries và không được giữ trong mô hình cuối. Cấu hình được chọn sau
ablation là **Q-Former cũ + fixed multi-score re-ranking 0.3/0.4/0.3**. Kết luận
này áp dụng cho checkpoint và một seed hiện tại; chưa thay thế thí nghiệm nhiều
seed.
