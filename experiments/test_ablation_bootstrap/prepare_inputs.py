"""Validate saved JSON; export paired scores/counts. No caption metric scorer."""
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    'full': (HERE/'full_review_cache', ROOT/'experiments/clip_patch_baseline/CSA_PER_CAPTION_BEST9_TEST5000_RESULT.json'),
    'zero': (ROOT/'experiments/prompt_text_ablation/test_v193_review_cache', ROOT/'experiments/prompt_text_ablation/TEST_RESULT.json'),
    'm2': (ROOT/'experiments/bank_only_ablation/test_v245_review_cache', ROOT/'experiments/bank_only_ablation/TEST_RESULT.json'),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate image ID')
    return result


def same_ids(*mappings):
    if not mappings or any(set(m) != set(mappings[0]) for m in mappings[1:]):
        raise ValueError('Unpaired image IDs')


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) < 1e-10, (a, b)


def prepare():
    arms, provenance = {}, {}
    expected_ids, expected_references = None, None
    for arm, (cache, aggregate_path) in SOURCES.items():
        manifest = read(cache/'ARCHIVE_MANIFEST.json')
        assert manifest['CRC_pass']
        for name, digest in manifest['files'].items():
            assert sha(cache/name) == digest, name
        aggregate = read(aggregate_path)
        result = read(cache/'result.json')
        locked = read(cache/'test_lock.json')
        scoring = read(cache/'scoring_lock.json')
        assert result['checkpoint_sha256'] == aggregate['checkpoint_sha256'] == locked['checkpoint_sha256'] == scoring['checkpoint_sha256']
        assert not result['training'] and not result['reranking'] and not result['test_tuning']
        assert locked['beam_width'] == 5 and locked['max_new_tokens'] == 20 and not locked['reranking']
        for name, digest in scoring['files_sha256'].items():
            assert sha(cache/name) == digest
        scores = indexed(read(cache/'cider_per_image.json'), 'coco_id')
        chair = indexed(read(cache/'chair_output.json')['sentences'], 'image_id')
        captions = indexed(read(cache/'captions.json'), 'coco_id')
        refs = indexed(read(cache/'test_references.json'), 'coco_id')
        same_ids(scores, chair, captions, refs)
        ids = sorted(scores)
        assert len(ids) == 5000 and ids == locked['expected_test_ids']
        if expected_ids is None:
            expected_ids, expected_references = ids, refs
        assert ids == expected_ids and refs == expected_references
        rows = []
        for image_id in ids:
            ch = chair[image_id]
            assert ch['caption'] == captions[image_id]['caption']
            h, o = len(ch['mscoco_hallucinated_words']), len(ch['mscoco_generated_words'])
            assert 0 <= h <= o
            score = scores[image_id]['CIDEr']
            assert math.isfinite(score)
            rows.append((score, h, o))
        cider = sum(r[0] for r in rows)/len(rows)
        hcap = sum(r[1] > 0 for r in rows)
        hmentions, mentions = sum(r[1] for r in rows), sum(r[2] for r in rows)
        reconstructed = dict(CIDEr=cider, CHAIRs=hcap/5000, CHAIRi=hmentions/mentions)
        for metric, value in reconstructed.items():
            close(value, result['metrics'][metric])
            close(value, aggregate['metrics'][metric])
        for metric, value in dict(images=5000, hallucinated_captions=hcap, hallucinated_mentions=hmentions, object_mentions=mentions).items():
            assert aggregate['chair_counts'][metric] == value
        arms[arm] = rows
        provenance[arm] = dict(
            archive_sha256=manifest['archive_sha256'], checkpoint_sha256=result['checkpoint_sha256'],
            checkpoint_source=result['source_version'], test_output_source=aggregate.get('test_output_source', 'historical full TEST archive; Save Version not restated'),
            aggregate_path=str(aggregate_path.relative_to(ROOT)).replace('\\', '/'),
            aggregate_sha256=sha(aggregate_path), metrics=reconstructed,
            counts=dict(images=5000, hallucinated_captions=hcap, hallucinated_mentions=hmentions, object_mentions=mentions),
            files_sha256={name: sha(cache/name) for name in ('captions.json', 'cider_per_image.json', 'chair_output.json', 'test_references.json', 'test_lock.json', 'scoring_lock.json')})
    output = HERE/'PAIRED_INPUT.csv'
    with output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['coco_id'] + [f'{arm}_{field}' for arm in SOURCES for field in ('CIDEr', 'hallucinated_mentions', 'object_mentions')])
        for i, image_id in enumerate(expected_ids):
            writer.writerow([image_id] + [value for arm in SOURCES for value in arms[arm][i]])
    info = dict(images=5000, columns='CIDEr raw score; integer CHAIR numerator/denominator; CHAIRs is numerator>0',
                input_sha256=sha(output), sources=provenance, training=False, model_inference=False, metric_scorers_run=False)
    (HERE/'INPUT_MANIFEST.json').write_text(json.dumps(info, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(images=5000, input_sha256=sha(output), points={a:p['metrics'] for a,p in provenance.items()}), indent=2))


if __name__ == '__main__':
    prepare()
