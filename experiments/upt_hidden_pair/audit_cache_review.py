"""Audit returned JSON/source/log artifacts only; no HDF5/model/inference execution."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile
from hidden_data import COHORT_SHA, DATASET_SHA, UPT_SHA, SCHEMA, human_sample
from pilot_data import identity

HERE = Path(__file__).resolve().parent


def audit(path):
    with zipfile.ZipFile(path) as z:
        assert len(z.namelist()) == len(set(z.namelist()))
        read = lambda name: json.loads(z.read(name))
        receipt, groups, protocol = read('cache_receipt.json'), read('cohort.json'), read('extraction_protocol.json')
        assert receipt['status'] == 'CACHE_COMPLETE' and receipt['schema'] == SCHEMA
        assert receipt['cohort_sha256'] == identity(groups) == COHORT_SHA
        assert receipt['dataset_sha256'] == DATASET_SHA and receipt['upt_checkpoint_sha256'] == UPT_SHA
        assert (receipt['slots'], receipt['width'], receipt['selection_box_threshold']) == (8, 1024, .2)
        assert receipt['test_used'] is False and receipt['consumed_caption_or_relation_references'] is False
        assert protocol['implementation_sha256'] == receipt['implementation_sha256']
        for name, sha in receipt['implementation_sha256'].items():
            assert hashlib.sha256(z.read('source/' + name)).hexdigest() == sha, name
        for name, sha in receipt['files'].items():
            if name.endswith('.h5'):
                assert name not in z.namelist(), 'Review archive should not silently include feature weights'
            else:
                assert hashlib.sha256(z.read(name)).hexdigest() == sha, name
        expected = groups['train'] + groups['val']
        assert len(groups['train']) == 10000 and len(groups['val']) == 1000
        assert len({r['coco_id'] for r in expected}) == 11000
        actual = []
        for rank in (0, 1):
            complete, runtime = read(f'complete_rank{rank}.json'), read(f'runtime_rank{rank}.json')
            assert complete['status'] == 'COMPLETE' and complete['images'] == 5500
            assert complete['model_sha256'] == runtime['model_sha256'] == receipt['model_sha256']
            records = [json.loads(line) for line in z.read(f'pairs_rank{rank}.jsonl').splitlines()]
            assert len(records) == 5500
            for record, planned in zip(records, expected[rank::2]):
                assert all(record[key] == value for key, value in planned.items())
                assert record['classifier_hook_parity'] is True and record['native_order_checked'] is True
                n = record['selected_pair_count']
                assert 0 <= n <= 8 and n == min(8, record['eligible_pair_count'])
                assert n <= record['eligible_pair_count'] <= record['human_nonhuman_pair_count'] <= record['native_pair_count']
                for key in ('selected_native_rows', 'selected_pairs', 'selected_labels', 'selected_confidence', 'boxes_original_normalized'):
                    assert len(record[key]) == n
                native, pairs = record['selected_native_rows'], record['selected_pairs']
                assert len(set(native)) == n and len(set(map(tuple, pairs))) == n
                for i in range(n):
                    s, o = pairs[i]
                    assert s >= 0 and o >= 0 and s != o and 0 <= native[i] < record['native_pair_count']
                    assert record['selected_labels'][i][0] == 0 and record['selected_labels'][i][1] != 0
                    assert all(math.isfinite(c) and .2 < c <= 1 for c in record['selected_confidence'][i])
                    for b in record['boxes_original_normalized'][i]:
                        assert len(b) == 4 and all(math.isfinite(x) for x in b) and b[2] > b[0] and b[3] > b[1]
                order = sorted(range(n), key=lambda i: (-math.prod(record['selected_confidence'][i]), *pairs[i], native[i]))
                assert order == list(range(n)), 'Selected rows not sorted by locked policy'
                for key in ('features_fp16_sha256', 'image_sha256'):
                    assert re.fullmatch('[0-9a-f]{64}', record[key])
                actual.append(record)
        counts = {r['coco_id']: r['selected_pair_count'] for r in actual}
        for split, rows in groups.items():
            coverage = dict(images=len(rows), with_pairs=sum(counts[r['coco_id']] > 0 for r in rows),
                            pair_count_histogram=dict(Counter(str(counts[r['coco_id']]) for r in rows)))
            assert coverage == receipt['coverage'][split]
        excluded = json.loads((HERE / 'excluded_phrase_ids.json').read_text())
        selected = human_sample(groups['val'], counts, excluded)
        log = z.read('notebook.log').decode(errors='replace')
        assert 'Ran 5 tests' in log and 'Ran 3 tests' in log and 'Ran 1 test' in log
        assert 'CACHE COMPLETE.' in log
        return dict(status='ARTIFACT_AUDIT_PASS_HDF5_BYTES_NOT_SUPPLIED',
            archive_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            cache_receipt_sha256=hashlib.sha256(z.read('cache_receipt.json')).hexdigest(),
            images=11000, workers=[5500, 5500], coverage=receipt['coverage'],
            selected_pairs=sum(counts.values()), model_sha256=receipt['model_sha256'],
            human32_preoutput=dict(rows=[dict(r, pair_count=counts[r['coco_id']]) for r in selected],
                with_pairs=sum(counts[r['coco_id']] > 0 for r in selected)),
            limits=['Actual HDF5 arrays are absent; byte hashes, per-row feature hashes and all-native topK eligibility cannot be independently replayed from this review.',
                    'Training notebooks verify mounted HDF5 file hashes, row IDs, shapes, padding, source policy and finiteness.',
                    'Detection-confidence coverage is not semantic relation correctness. No caption training result yet.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('archive', type=Path)
    args = p.parse_args()
    result = audit(args.archive)
    (HERE / 'CACHE_RECEIVED_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(result['status'], result['images'], result['coverage'])
