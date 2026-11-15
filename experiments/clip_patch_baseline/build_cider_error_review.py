import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import math
from pathlib import Path
import random
import urllib.request
from zipfile import ZipFile

parser = argparse.ArgumentParser()
parser.add_argument('archive', type=Path)
parser.add_argument('review', type=Path)
parser.add_argument('manifest', type=Path)
args = parser.parse_args()

with ZipFile(args.archive) as archive:
    def read(name):
        return json.loads(archive.read(name))
    def keyed(rows):
        result = {row['coco_id']: row for row in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate COCO IDs')
        return result
    status = read('run_status.json')
    assert status['complete'] and not status['test_used']
    source = read('recovery_source.json')
    assert source['kaggle_source'] == 'ducanh3/notebook7bc83c4c76/94'
    assert source['receipt']['sha256'] == '80e1280b9eef216c5db421a738c7a5e0550bb34b1a62a2a13b2c8e76d40b1a24'
    inputs = keyed(read('rerank/inference_inputs.json'))
    final = keyed(read('rerank/clip_object_fixed_captions.json'))
    scores = keyed(read('rerank/clip_object_fixed_cider_per_image.json'))
    beams = keyed(read('beams_rank0.json') + read('beams_rank1.json'))
    ranks = dict(zip(inputs, read('rerank/selected_ranks.json'), strict=True))
    assert len(inputs) == 5000 and inputs.keys() == final.keys() == scores.keys() == beams.keys()
    for image_id, rank in ranks.items():
        candidates = beams[image_id]['candidates']
        assert len(candidates) == 5 and 0 <= rank < 5
        assert candidates[rank]['caption'] == final[image_id]['caption']
        assert math.isfinite(scores[image_id]['CIDEr'])

rng = random.Random(1500)
ordered = sorted(scores, key=lambda image_id: (scores[image_id]['CIDEr'], image_id))
groups = {image_id: 'low' for image_id in rng.sample(ordered[:500], 30)}
groups.update({image_id: 'comparison' for image_id in rng.sample(ordered[500:], 10)})
review_order = list(groups)
rng.shuffle(review_order)

def image_data(image_id):
    path = Path(inputs[image_id]['image_path'])
    assert path.parent.name in {'train2014', 'val2014'}
    url = f'https://s3.amazonaws.com/images.cocodataset.org/{path.parent.name}/{path.name}'
    with urllib.request.urlopen(url, timeout=20) as response:
        data = response.read()
    if not data.startswith(b'\xff\xd8') or hashlib.sha256(data).hexdigest() != inputs[image_id]['image_sha256']:
        raise ValueError(f'Image hash mismatch: {image_id}')
    return base64.b64encode(data).decode('ascii')

with ThreadPoolExecutor(max_workers=8) as pool:
    images = list(pool.map(image_data, review_order))

template = Path(__file__).with_name('cider_error_review.html').read_text(encoding='utf-8')
rows, records = [], []
for number, (image_id, image) in enumerate(zip(review_order, images, strict=True), 1):
    caption = final[image_id]['caption']
    options = [candidate['caption'] for candidate in beams[image_id]['candidates']]
    options = list(dict.fromkeys(text for text in options if text != caption))
    rng.shuffle(options)
    alternatives = ''.join(f'<li>{html.escape(text)}</li>' for text in options)
    rows.append(f"""<section data-id="{image_id}">
<h2>Ảnh {number} · COCO {image_id}</h2>
<img loading="lazy" src="data:image/jpeg;base64,{image}" alt="Ảnh COCO {image_id}">
<div><p><b>Caption cuối:</b> {html.escape(caption)}</p>
<label>Lỗi chính <select data-field="error">
<option value="">Chưa chấm</option><option value="correct">Đúng, đủ ý</option>
<option value="object">Sai hoặc thêm đối tượng</option><option value="action">Sai hành động hoặc quan hệ</option>
<option value="missing">Thiếu ý chính</option><option value="attribute">Sai thuộc tính hoặc số lượng</option>
<option value="language">Lặp hoặc diễn đạt khó hiểu</option><option value="unclear">Không rõ</option>
</select></label>
<details><summary>Xem các caption còn lại</summary><ol>{alternatives}</ol>
<label>Có câu tốt hơn rõ ràng? <select data-field="alternative"><option value="">Chưa chấm</option>
<option value="yes">Có</option><option value="no">Không</option><option value="unclear">Không rõ</option></select></label>
</details>
<label>Ghi chú <textarea data-field="note" rows="2"></textarea></label>
</div></section>""")
    records.append(dict(coco_id=image_id, group=groups[image_id], cider=scores[image_id]['CIDEr'],
                        selected_rank=ranks[image_id], image_sha256=inputs[image_id]['image_sha256']))
args.review.write_text(template.replace('<!-- CASES -->', '\n'.join(rows)), encoding='utf-8')
args.manifest.write_text(json.dumps(dict(source=source, seed=1500, cases=records), indent=2) + '\n', encoding='utf-8')
print('Verified 5000 selected captions and 40 image hashes. Review generated.')
