"""Build a locked diagnostic from existing human/V-COCO labels, never missing-edge negatives."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments/semantic_evidence_pilot'))
from captioning.upt_prompts import relation_text
from pilot_data import cohorts
from probe_data import digest, identity, human_pairs, vcoco_pairs, finalize


def prepare(root, output):
    root = Path(root)
    files = dict(dataset=root/'vcoco_alignment_week1/sources/dataset_coco.json',
                 grounded=root/'vcoco_alignment_week1/grounded_links_v1/relation_dev_images.json',
                 cases=root/'target_swap_annotation_v1/cases.json',
                 review=root/'target_swap_annotation_v1/human_batch1_review.json')
    source = {k: json.loads(p.read_text(encoding='utf-8')) for k,p in files.items()}
    assert digest(files['dataset']) == '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
    assert digest(files['cases']) == source['review']['manifest_sha256']
    captions = {r['filename']: r for r in source['dataset']['images']}
    excluded = {r['filename'] for r in cohorts(source['dataset'])['train']}
    rows = human_pairs(source['cases'], source['review'], excluded, captions, relation_text)
    rows += vcoco_pairs(source['grounded'], excluded, captions, relation_text)
    rows = finalize(rows, captions)
    assert rows and not any(r['filename'] in excluded for r in rows)
    assert all(r['caption_split'] in ('train', 'restval') for r in rows)
    manifest = dict(schema_version=1, purpose='Frozen E matcher diagnostic; no caption training or checkpoint selection',
        checkpoint_sha256='651f411f589c5dd7beb0fae98fafaa7d9ea06a7fd7f2b49400ac878f60227b18',
        checkpoint_protocol_sha256='3e34a4561edc269152c0a61f3c76a807b56da85192a18d1e8bac99c8642a085f',
        checkpoint_cohort_sha256='6bbdf3b75869927334ec6c2f868edf9f52346c0e79401aa0e118ba5a02fb711b',
        sources_sha256={k:digest(p) for k,p in files.items()}, excluded_pilot_train_filenames_sha256=identity(sorted(excluded)),
        excluded_pilot_train_images=len(excluded),
        test_used=False, validation_images_used=False, alignment_train_overlap=0,
        caption_pretraining_overlap='All diagnostic images are Karpathy train/restval, historically used for caption pretraining; not unseen-test evidence.',
        scope='Existing reused human diagnostics + V-COCO val-role labels; GT boxes, not UPT detections; no annotation absence interpreted as false.',
        counts=dict(pairs=len(rows), images=len({r['filename'] for r in rows}), tasks=dict(Counter(r['task'] for r in rows)),
                    positive_predicates=dict(Counter((r['task']+':'+r['predicate']) for r in rows))), rows=rows)
    manifest['rows_sha256'] = identity(rows)
    Path(output).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(manifest['counts'], indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--local-output-root', required=True);p.add_argument('--output', required=True)
    a=p.parse_args();prepare(a.local_output_root, a.output)
