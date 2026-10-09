# Audit prompt quan hệ trước khi phát triển E3

Workflow này không train và không cần GPU. Nó đọc JSON prompt
`promptcache_H1_1_semantic_dedup.json`, thống kê toàn bộ cache và xuất 120 ảnh
mẫu thành CSV + HTML để đánh dấu quan hệ đúng, dư thừa, sai hoặc bị thiếu.

Add Input gồm MS COCO 2014 và dataset chứa semantic prompt JSON. Chọn
Accelerator `None`.

```python
import csv
import html
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
OUTPUT = Path('/kaggle/working/relation_prompt_audit')
SAMPLE_SIZE = 120
SEED = 42


def select_one(paths, label):
    paths = sorted(set(paths), key=str)
    assert paths, f'Không tìm thấy {label}'
    if len(paths) > 1:
        print(f'Tìm thấy nhiều {label}:')
        for path in paths:
            print(' -', path)
        dataset_paths = [p for p in paths if '/kaggle/input/datasets/' in p.as_posix()]
        selected = dataset_paths[0] if dataset_paths else paths[0]
    else:
        selected = paths[0]
    print(f'Dùng {label}:', selected)
    return selected


def triplet_text(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return ' '.join(str(item) for item in value).strip()
    if isinstance(value, dict):
        subject = value.get('subject', value.get('subj', value.get('source', '')))
        relation = value.get('relation', value.get('predicate', value.get('rel', '')))
        obj = value.get('object', value.get('obj', value.get('target', '')))
        assembled = ' '.join(str(item) for item in (subject, relation, obj) if item)
        return assembled.strip() or json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def relation_name(value):
    if isinstance(value, dict):
        return str(value.get('relation', value.get('predicate', value.get('rel', 'unknown'))))
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return str(value[1])
    text = triplet_text(value).lower()
    known = ('inside', 'contains', 'above', 'below', 'left of', 'right of',
             'near', 'overlapping', 'on', 'under')
    return next((name for name in known if f' {name} ' in f' {text} '), 'other')


def object_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get('label', value.get('name', value.get('class', value))))
    return str(value)


# 1. Chọn prompt JSON và nạp COCO metadata.
prompt_json = select_one(
    Path('/kaggle/input').rglob('promptcache_H1_1_semantic_dedup.json'),
    'semantic prompt JSON')
raw = json.loads(prompt_json.read_text(encoding='utf-8'))
records = raw.get('data', raw) if isinstance(raw, dict) else raw
assert isinstance(records, dict) and records, type(records)

coco = json.loads(COCO_JSON.read_text(encoding='utf-8'))
coco_by_name = {item['filename']: item for item in coco['images']}
assert coco_by_name


# 2. Chuẩn hóa record và tính thống kê toàn cache.
rows = []
relation_counts = Counter()
for key, entry in records.items():
    if not isinstance(entry, dict):
        continue
    filename = Path(key).name
    triplets = entry.get('triplets') or []
    objects = entry.get('objects') or []
    prompt = str(entry.get('prompt', entry.get('prompt_string', '')))
    triplet_strings = [triplet_text(item) for item in triplets]
    triplet_strings = [item for item in triplet_strings if item]
    relation_counts.update(relation_name(item) for item in triplets)
    coco_item = coco_by_name.get(filename, {})
    rows.append({
        'filename': filename,
        'filepath': str(coco_item.get('filepath', '')),
        'split': str(coco_item.get('split', 'unknown')),
        'objects': '; '.join(object_text(item) for item in objects),
        'triplets': ' | '.join(triplet_strings),
        'triplet_count': len(triplet_strings),
        'prompt': prompt,
        'prompt_chars': len(prompt),
    })

assert rows, 'Không chuẩn hóa được prompt records'
with_relations = [row for row in rows if row['triplet_count'] > 0]
summary = {
    'prompt_json': str(prompt_json),
    'record_count': len(rows),
    'images_with_relations': len(with_relations),
    'images_without_relations': len(rows) - len(with_relations),
    'relation_coverage': len(with_relations) / len(rows),
    'total_triplets': sum(row['triplet_count'] for row in rows),
    'mean_triplets_all_images': sum(row['triplet_count'] for row in rows) / len(rows),
    'mean_triplets_when_present': (
        sum(row['triplet_count'] for row in with_relations) / len(with_relations)
        if with_relations else 0.0),
    'mean_prompt_chars': sum(row['prompt_chars'] for row in rows) / len(rows),
    'relation_distribution': dict(relation_counts.most_common()),
}


# 3. Lấy mẫu ưu tiên test split và ảnh có nhiều quan hệ.
rng = random.Random(SEED)
pool = [row for row in with_relations if row['split'] == 'test'] or with_relations
ranked = sorted(pool, key=lambda row: (-row['triplet_count'], row['filename']))
high_count = ranked[:min(40, len(ranked))]
remaining = [row for row in pool if row not in high_count]
rng.shuffle(remaining)
sample = high_count + remaining[:max(0, SAMPLE_SIZE - len(high_count))]
sample = sample[:SAMPLE_SIZE]
assert sample


# 4. Xuất thumbnail, CSV có cột đánh giá thủ công và HTML.
OUTPUT.mkdir(parents=True, exist_ok=True)
thumb_dir = OUTPUT / 'images'
thumb_dir.mkdir(exist_ok=True)

fieldnames = [
    'index', 'filename', 'split', 'objects', 'triplets', 'triplet_count',
    'prompt', 'manual_label', 'manual_note',
]
csv_path = OUTPUT / 'relation_prompt_audit.csv'
html_cards = []
with csv_path.open('w', encoding='utf-8-sig', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for index, row in enumerate(sample, 1):
        source = COCO_IMAGES / row['filepath'] / row['filename']
        if not source.is_file():
            candidates = list(COCO_IMAGES.rglob(row['filename']))
            source = candidates[0] if len(candidates) == 1 else source
        assert source.is_file(), source
        thumb_path = thumb_dir / f'{index:03d}_{row["filename"]}'
        with Image.open(source) as image:
            image = image.convert('RGB')
            image.thumbnail((480, 360))
            image.save(thumb_path, quality=88)
        writer.writerow({
            'index': index,
            'filename': row['filename'],
            'split': row['split'],
            'objects': row['objects'],
            'triplets': row['triplets'],
            'triplet_count': row['triplet_count'],
            'prompt': row['prompt'],
            'manual_label': '',
            'manual_note': '',
        })
        html_cards.append(f'''<article>
          <h2>#{index} — {html.escape(row['filename'])}</h2>
          <img src="images/{html.escape(thumb_path.name)}" loading="lazy">
          <p><b>Objects:</b> {html.escape(row['objects'])}</p>
          <p><b>Triplets:</b> {html.escape(row['triplets'])}</p>
          <p><b>Prompt:</b> {html.escape(row['prompt'])}</p>
          <p class="rating">Đánh giá: □ đúng-hữu ích &nbsp; □ đúng-dư-thừa
          &nbsp; □ sai &nbsp; □ thiếu-quan-hệ</p>
        </article>''')

html_path = OUTPUT / 'relation_prompt_audit.html'
html_path.write_text('''<!doctype html><meta charset="utf-8">
<title>Relation Prompt Audit</title>
<style>
body{font:15px Arial;margin:24px;background:#f6f7f9;color:#17202a}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}
article{background:white;border:1px solid #d7dce2;border-radius:12px;padding:16px}
img{display:block;max-width:100%;max-height:360px;margin:auto;border-radius:8px}
h1{color:#123b59}.rating{background:#fff3cd;padding:10px;border-radius:6px}
</style><h1>Audit prompt quan hệ — 120 ảnh</h1><main>''' +
    '\n'.join(html_cards) + '</main>', encoding='utf-8')

summary_path = OUTPUT / 'relation_prompt_summary.json'
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

print('\nRELATION PROMPT AUDIT SUMMARY')
print(json.dumps(summary, ensure_ascii=False, indent=2))
print('\nCSV:', csv_path)
print('HTML:', html_path)
print('Summary:', summary_path)
```

Sau khi chạy, mở `relation_prompt_audit.html` để xem nhanh. Trong CSV, điền
`manual_label` bằng một trong bốn nhãn:

- `correct_useful`
- `correct_redundant`
- `wrong_relation`
- `missing_important_relation`

Chỉ cần gán nhãn 50–100 ảnh trước báo cáo tuần. Tỷ lệ lỗi này là bằng chứng trực
tiếp để giải thích vì sao E3 cần thêm bước kiểm chứng quan hệ bằng hình ảnh.

