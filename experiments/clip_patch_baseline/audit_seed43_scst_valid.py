"""Inspect saved VALID text/CHAIR changes, without models or metric scorers."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
from pathlib import Path
import random
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def categories(row):
    return [item[1] if isinstance(item, (list, tuple)) else item
            for item in row['mscoco_hallucinated_words']]


def compare(xe, scst, xe_cider, scst_cider):
    a = {r['image_id']: r for r in xe['sentences']}
    b = {r['image_id']: r for r in scst['sentences']}
    if len(a) != 5000 or set(a) != set(b) or len(b) != 5000 or len(xe['sentences']) != 5000 or len(scst['sentences']) != 5000:
        raise ValueError('Need matching VALID5000')
    if set(xe_cider) != set(a) or set(scst_cider) != set(a):
        raise ValueError('Wrong per-image CIDEr IDs')
    if any(set(a[i]['mscoco_gt_words']) != set(b[i]['mscoco_gt_words']) for i in a):
        raise ValueError('Changed reference object sets')
    ids = sorted(a)
    groups = dict(new_flags=[i for i in ids if not categories(a[i]) and categories(b[i])],
                  repairs=[i for i in ids if categories(a[i]) and not categories(b[i])],
                  shared_flags=[i for i in ids if categories(a[i]) and categories(b[i])],
                  clean_both=[i for i in ids if not categories(a[i]) and not categories(b[i])])
    rng = random.Random(20261119)
    sample = [(g, i) for g, count in [('new_flags', 20), ('repairs', 5), ('clean_both', 5)]
              for i in rng.sample(groups[g], min(count, len(groups[g])))]
    cases = [dict(coco_id=i, group=g, xe_caption=a[i]['caption'], scst_caption=b[i]['caption'],
                  xe_hallucinated_words=a[i]['mscoco_hallucinated_words'],
                  scst_hallucinated_words=b[i]['mscoco_hallucinated_words'],
                  annotated_objects=sorted(set(a[i]['mscoco_gt_words'])),
                  xe_CIDEr=xe_cider[i], scst_CIDEr=scst_cider[i]) for g, i in sample]
    return dict(images=5000, split='val', test_used=False,
                XE_flagged=sum(bool(categories(a[i])) for i in ids),
                SCST_flagged=sum(bool(categories(b[i])) for i in ids),
                transition_counts={g: len(v) for g, v in groups.items()},
                changed_captions=sum(a[i]['caption'] != b[i]['caption'] for i in ids),
                new_flag_categories=dict(Counter(c for i in groups['new_flags'] for c in categories(b[i])).most_common()),
                new_flag_ids=groups['new_flags'], sample_seed=20261119, cases=cases,
                limits=['Automatic CHAIR flags, not independently verified visual hallucinations.',
                        'Top1 XE versus SCST1500 diagnostic, not SCST500 or rerank evaluation.',
                        'Selected seed43; no representative population or causal claim.'])


def fetch_image(task):
    case, directory = task
    path = directory / f"{case['coco_id']:012d}.jpg"
    name = 'COCO_val2014_' + path.name
    urls = [f'https://images.cocodataset.org/val2014/{name}',
            f'http://images.cocodataset.org/val2014/{name}']
    error = None
    if path.is_file():
        raw = path.read_bytes()
    else:
        raw = None
        for url in urls:
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(request, timeout=12) as response:
                    candidate = response.read(6*1024**2)
                if not candidate.startswith(b'\xff\xd8') or len(candidate) >= 6*1024**2:
                    raise ValueError('Response is not a bounded JPEG')
                raw = candidate
                path.write_bytes(raw)
                break
            except Exception as exc:
                error = str(exc)
    return dict(coco_id=case['coco_id'], filename=path.name,
                sha256=hashlib.sha256(raw).hexdigest() if raw else None,
                error=None if raw else error)


def page(result, image_dir):
    import base64
    cards = []
    annotations = HERE/'CSA_SEED43_SCST_VALID_ASSISTED_REVIEW.json'
    labels = read(annotations)['cases'] if annotations.is_file() else {}
    for j, case in enumerate(result['cases'], 1):
        path = image_dir / f"{case['coco_id']:012d}.jpg"
        photo = '<p>Ảnh chưa tải được; xem ghi chú tải ảnh trong JSON.</p>'
        if path.is_file():
            photo = '<img src="data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode() + '">'
        esc = lambda value: html.escape(str(value))
        review = labels.get(str(case['coco_id']))
        note = 'Chưa xem trực quan.'
        if review:
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != review['image_sha256']:
                raise ValueError('Changed visually reviewed image')
            note = review['judgment'] + ': ' + review['note']
        cards.append(f'''<article><h2>#{j} · COCO {case['coco_id']} · {esc(case['group'])}</h2>
        <div class="pair">{photo}<div><p><b>XE:</b> {esc(case['xe_caption'])}</p>
        <p><b>SCST1500:</b> {esc(case['scst_caption'])}</p>
        <p><b>CHAIR đánh dấu ở XE:</b> {esc(case['xe_hallucinated_words'])}</p>
        <p><b>CHAIR đánh dấu ở SCST:</b> {esc(case['scst_hallucinated_words'])}</p>
        <p><b>Nhãn đối tượng trong bộ chấm:</b> {esc(', '.join(case['annotated_objects']))}</p>
        <p><b>Nhận xét trợ lý, cần bạn kiểm tra:</b> {esc(note)}</p></div></div></article>''')
    return '''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Đối chiếu VALID: XE và SCST</title>
    <style>body{font:17px Arial;max-width:1150px;margin:30px auto;padding:0 20px;color:#203047}article{border-top:1px solid #ccd5df;padding:20px 0}.pair{display:grid;grid-template-columns:420px 1fr;gap:24px}img{max-width:420px;max-height:420px;object-fit:contain}h2{font-size:20px}@media(max-width:800px){.pair{display:block}img{max-width:100%}}</style>
    <h1>Đối chiếu trên VALID5000: XE và SCST1500, seed43</h1>
    <p>Mẫu cố định gồm 20 ca CHAIR đánh dấu mới, 5 ca được sửa và 5 ca không bị đánh dấu ở cả hai bản.
    Cần xem ảnh để xác nhận; nhãn tự động chưa chứng minh lỗi thị giác. Đây không phải đánh giá mù.</p>''' + ''.join(cards) + '</html>'


def run(downloads):
    xe_zip = downloads/'csa_per_caption_h5_seed43_resume178_review.zip'
    scst_zip = downloads/'csa_per_caption_seed43_scst1500_test_rerank_review.zip'
    for path, cache in [(xe_zip, 'CSA_PER_CAPTION_H5_SEED43_FULL_XE10_RESULT.json'),
                        (scst_zip, 'CSA_PER_CAPTION_SEED43_SCST1500_TEST_RESULT.json')]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != read(HERE/cache)['zip_sha256']:
            raise ValueError('Changed reviewed archive: '+path.name)
    with zipfile.ZipFile(xe_zip) as z:
        xe = json.loads(z.read('quality/epoch_08/chair_output.json'))
        per_xe = json.loads(z.read('evaluations/epoch_08/cider_per_image.json'))
        prediction_hashes = {f'val_rank{rank}.json': hashlib.sha256(z.read(
            f'evaluations/epoch_08/val_rank{rank}.json')).hexdigest() for rank in (0, 1)}
    with zipfile.ZipFile(scst_zip) as z:
        scst = json.loads(z.read('chair_output.json'))
        per_scst = json.loads(z.read('cider_per_image.json'))
    result = compare(xe, scst, {r['coco_id']: r['CIDEr'] for r in per_xe},
                     {r['coco_id']: r['CIDEr'] for r in per_scst})
    result['sources'] = dict(XE='ducanh3/notebook7bc83c4c76/179', SCST='ducanh3/notebook7bc83c4c76/185')
    result['XE_prediction_payload_sha256'] = prediction_hashes
    if (HERE/'CSA_SEED43_SCST_VALID_ASSISTED_REVIEW.json').is_file():
        result['assistant_review_file'] = 'CSA_SEED43_SCST_VALID_ASSISTED_REVIEW.json'
    image_dir = downloads/'seed43_scst_valid_audit_images'
    image_dir.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        result['image_files'] = list(pool.map(fetch_image, [(c, image_dir) for c in result['cases']]))
    destination = HERE/'CSA_SEED43_SCST_VALID_REGRESSIONS_RESULT.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    review = downloads.parent/'seed43_scst_valid_audit.html'
    review.write_text(page(result, image_dir), encoding='utf-8')
    print('CHAIR transition counts:', result['transition_counts'], flush=True)
    print('New-flag categories:', result['new_flag_categories'], flush=True)
    print('Images available:', sum(r['sha256'] is not None for r in result['image_files']), '/', len(result['cases']), flush=True)
    print('REVIEW:', review, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--downloads', type=Path, default=Path('C:/Users/leduc/Downloads/18-11'))
    run(parser.parse_args().downloads)
