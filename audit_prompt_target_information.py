"""Measure target information lost in text serialization; no caption-quality claims."""
import argparse
import hashlib
import json
from pathlib import Path

from captioning.selected_relation_prompts import build_selected_prompts
from captioning.supervised_relations import RELATIONS


def serialize_case(case, side):
    target = case['targets'][side]
    scores = [-1.0] * len(RELATIONS)
    scores[RELATIONS.index(case['relation'])] = 1.0
    row = dict(filename=case['filename'], subject='person', object=target['label'],
               subject_box=case['subject_box'], object_box=target['bbox'], scores=scores)
    return build_selected_prompts([row], [0.0] * len(RELATIONS))[case['filename']]


def grounded_payload(entry):
    """Keep instance identity and boxes alongside language for a future cache bridge.

    This is an interchange contract, NOT a CLIP tensor cache or decoder input.
    IDs are metadata, not invented words to send to CLIP.
    """
    objects = {obj['instance_id']: obj for obj in entry['objects']}
    return [dict(text=claim['prompt'], predicate=claim['relation'],
                 subject_instance=claim['subject_instance'],
                 object_instance=claim['object_instance'],
                 subject_box=objects[claim['subject_instance']]['bbox'],
                 object_box=objects[claim['object_instance']]['bbox'])
            for claim in entry['relations']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--filtered', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    package, source = Path(args.package), Path(args.filtered)
    cases = json.loads((package / 'review/cases.json').read_text(encoding='utf-8'))['cases']
    mapping = {case['case_id']: case for case in cases}
    filtered = json.loads(source.read_text(encoding='utf-8'))
    rows = []
    for contrast in filtered['eligible_contrasts']:
        case = mapping[contrast['case_id']]
        a, b = serialize_case(case, 0), serialize_case(case, 1)
        pa, pb = grounded_payload(a), grounded_payload(b)
        rows.append(dict(case_id=case['case_id'], predicate=case['relation'],
                         same_class=case['same_class'], text_A=a['prompt'], text_B=b['prompt'],
                         text_identical=a['prompt'] == b['prompt'],
                         grounded_payload_different=pa != pb,
                         payload_A=pa, payload_B=pb))
    summary = dict(variant='prompt_target_information_audit_v1', eligible_cases=len(rows),
                   same_class=sum(r['same_class'] for r in rows),
                   identical_text=sum(r['text_identical'] for r in rows),
                   distinct_grounded_payload=sum(r['grounded_payload_different'] for r in rows),
                   filtered_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   notes='Train annotations only. Exact same text implies same deterministic text '
                         'encoding, not identical grounded visual memory. No caption checkpoint '
                         'executed; no CIDEr/BLEU/CHAIR measurement. Payload is not a tensor cache.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'cases.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
