# Đánh giá CHAIR: Q-Former gốc và Q-Former + OCC

## Mục tiêu

Thống kê `hallucinated=582/6905` của OCC chỉ cho biết YOLO không hỗ trợ bao nhiêu
object mention trong caption được chọn. Nó không phải CHAIR vì YOLO prediction
không phải ground truth.

CHAIR đối chiếu object trong caption với hai nguồn ground truth của MS COCO:

- object segmentation/instance annotations;
- object xuất hiện trong năm reference captions.

Hai chỉ số chính đều **càng thấp càng tốt**:

- `CHAIRs`: tỷ lệ caption có ít nhất một object bị ảo giác;
- `CHAIRi`: tỷ lệ object mention bị ảo giác trên tổng object mention được sinh.

Workflow dưới đây đánh giá đúng cùng 5.000 ảnh test cho Q-Former rank 0 và
Q-Former sau OCC weight `0.1`. `Recall` do bản standalone bổ sung, càng cao càng
tốt; đây không phải một trong hai chỉ số CHAIR gốc.

## Input Kaggle cần gắn

Gắn output của phiên `qformer_occ_test_w010` đã chạy xong. Input đó phải chứa:

- `test_5000_beam5_candidates.json`;
- `test_5000_occ_w0p100_predictions.json`.

Cell tự tìm hai file trong `/kaggle/input`, nên không cần sửa đường dẫn nếu mỗi
tên chỉ xuất hiện đúng một lần. Bật Internet để clone hai repository và cài các
dependency nhỏ. Không cần GPU; chọn accelerator `None` là đủ.

## Cell Kaggle hoàn chỉnh

```python
import json
import subprocess
import sys
from pathlib import Path

IMAGE_CAPTIONING = Path('/kaggle/working/Image_Captioning')
CHAIR_REPO = Path('/kaggle/working/CHAIR-metric-standalone')
OUTPUT = Path('/kaggle/working/chair_qformer_occ_test')

IMAGE_CAPTIONING_COMMIT = 'd1877e1'
CHAIR_COMMIT = '4087a26211aa2339b9a76307cb8f0321ef691d0a'


def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def exactly_one(name):
    matches = sorted(Path('/kaggle/input').rglob(name), key=str)
    print(f'\n{name} candidates:')
    for match in matches:
        print('-', match)
    assert len(matches) == 1, (
        f'Cần đúng một file {name}, hiện tìm thấy {len(matches)}. '
        'Gỡ Input trùng hoặc gán đường dẫn thủ công.'
    )
    return matches[0]


# 1. Tìm output test đã có; không sinh caption và không train lại.
CANDIDATES = exactly_one('test_5000_beam5_candidates.json')
OCC_PREDICTIONS = exactly_one('test_5000_occ_w0p100_predictions.json')


# 2. Clone code chuyển eval_id sang COCO id thật.
if not IMAGE_CAPTIONING.exists():
    run([
        'git', 'clone',
        'https://github.com/Supzxjee/Image_Captioning.git',
        IMAGE_CAPTIONING,
    ])
run(['git', 'fetch', 'origin'], cwd=IMAGE_CAPTIONING)
run(['git', 'checkout', '--detach', IMAGE_CAPTIONING_COMMIT], cwd=IMAGE_CAPTIONING)
run(['git', 'rev-parse', '--short', 'HEAD'], cwd=IMAGE_CAPTIONING)


# 3. Clone cố định bản CHAIR standalone tương thích Python 3.
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


# 4. Tạo hai input CHAIR khớp từng ảnh.
# Candidate bundle lưu cả eval_id và coco_id. CHAIR cần coco_id; tuyệt đối không
# đưa trực tiếp image_id trong file OCC cũ vào evaluator.
OUTPUT.mkdir(parents=True, exist_ok=True)
run([
    sys.executable, '-u',
    IMAGE_CAPTIONING / 'prepare_chair_comparison.py',
    '--candidates', CANDIDATES,
    '--occ-predictions', OCC_PREDICTIONS,
    '--output-dir', OUTPUT,
])

manifest = json.loads(
    (OUTPUT / 'chair_input_manifest.json').read_text(encoding='utf-8')
)
assert manifest['count'] == 5000, manifest
assert manifest['changed_captions'] == 34, manifest
print('\nInput manifest:', json.dumps(manifest, indent=2))


# 5. Chạy cùng một evaluator/cùng ground truth cho hai prediction files.
BASELINE_CHAIR = OUTPUT / 'qformer_baseline_chair_output.json'
OCC_CHAIR = OUTPUT / 'qformer_occ_w010_chair_output.json'
CHAIR_CACHE = CHAIR_REPO / 'chair.pkl'
assert CHAIR_CACHE.is_file(), CHAIR_CACHE

for predictions, result in (
    (OUTPUT / 'qformer_baseline_chair_input.json', BASELINE_CHAIR),
    (OUTPUT / 'qformer_occ_w010_chair_input.json', OCC_CHAIR),
):
    run([
        sys.executable, '-u', CHAIR_REPO / 'chair.py',
        '--cap_file', predictions,
        '--image_id_key', 'image_id',
        '--caption_key', 'caption',
        '--cache', CHAIR_CACHE,
        '--save_path', result,
    ], cwd=CHAIR_REPO)


# 6. So sánh theo cặp và lưu các trường hợp OCC cải thiện/làm xấu đi.
SUMMARY = OUTPUT / 'qformer_vs_occ_chair_summary.json'
run([
    sys.executable, '-u',
    IMAGE_CAPTIONING / 'summarize_chair_comparison.py',
    '--baseline-chair', BASELINE_CHAIR,
    '--occ-chair', OCC_CHAIR,
    '--output', SUMMARY,
])

summary = json.loads(SUMMARY.read_text(encoding='utf-8'))
print('\nKẾT QUẢ CUỐI')
print('Q-Former:', summary['baseline'])
print('Q-Former + OCC:', summary['occ'])
print('Delta OCC - baseline:', summary['delta_occ_minus_baseline'])
print('Sentence transitions:', summary['sentence_transitions'])
print('Changed captions:', summary['changed_captions'])
print('Summary:', SUMMARY)
```

## Cách kết luận

Chỉ dùng kết quả test để báo cáo, không chỉnh lại OCC weight theo test. Báo cáo ít
nhất bốn dòng sau:

| Mô hình | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | CIDEr ↑ |
|---|---:|---:|---:|---:|
| Q-Former | kết quả mới | kết quả mới | kết quả mới | 1.188245 |
| Q-Former + OCC 0.1 | kết quả mới | kết quả mới | kết quả mới | 1.188175 |

Đọc thêm `sentence_transitions`:

- `improved_to_clean`: OCC loại được object hallucination khỏi caption;
- `regressed_to_hallucinated`: OCC đổi một caption sạch thành caption có ảo giác;
- `both_hallucinated`: cả hai caption vẫn có object hallucination;
- `both_clean`: cả hai caption đều sạch theo CHAIR.

OCC chỉ nên được xem là có đóng góp nếu `CHAIRs` hoặc `CHAIRi` giảm đủ rõ và số
`improved_to_clean` lớn hơn `regressed_to_hallucinated`. Do OCC chỉ đổi 34/5.000
caption, mức thay đổi toàn tập được dự đoán là nhỏ; đây vẫn là kết quả ablation có
giá trị vì nó cho biết hậu xử lý dựa trên YOLO có tác động thực tế đến ảo giác hay
không.

## Nguồn phương pháp

- Bài báo gốc: [Object Hallucination in Image Captioning](https://aclanthology.org/D18-1437/).
- Mã nguồn gốc: [LisaAnne/Hallucination](https://github.com/LisaAnne/Hallucination).
- Bản Python 3 standalone dùng trong cell: [Maxlinn/CHAIR-metric-standalone](https://github.com/Maxlinn/CHAIR-metric-standalone).

Bản standalone giữ cách tính CHAIR nhưng không phải repository chính thức của tác
giả bài báo. Vì vậy trong báo cáo cần ghi rõ repository và commit evaluator đã
dùng; không trộn kết quả này với một implementation CHAIR khác trong cùng bảng.

