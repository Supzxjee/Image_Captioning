"""Audit cached review text only; never reads HDF5/checkpoints or runs a model."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import kv_data as kd

HERE = Path(__file__).resolve().parent


def audit():
    cached = json.loads((HERE/'review/kv_preparation_loaded.json').read_bytes())
    members = cached['members']
    receipt = json.loads(members['cpu_preparation.json'])
    stats = json.loads(members['statistics.json'])
    auxiliary = json.loads(members['auxiliary_facts.json'])
    meta = json.loads(members['metadata.json'])
    banks = json.loads(members['detector_boxes.json'])
    sources = json.loads(members['source_receipts.json'])
    protocol = json.loads(members['protocol.json'])
    checks = 0

    def require(condition, label):
        nonlocal checks
        if not condition:
            raise AssertionError(label)
        checks += 1

    require(receipt['status'] == 'READY_FOR_MODEL_PREFLIGHT', 'CPU completion')
    require(not any(receipt[n] for n in ('neural_forward','feature_extraction','checkpoint_loaded')), 'no model claim')
    require(receipt['policy'] == protocol['policy'] == meta['policy'] == kd.POLICY, 'locked policy')
    require(protocol['config'] == json.loads(members['requested_config.json']), 'config identity')
    require(protocol['config']['source_commit'] == receipt['source_commit'], 'source commit')
    verified = []
    for name, expected in receipt['payload_sha256'].items():
        if name in members:
            require(hashlib.sha256(members[name].encode()).hexdigest() == expected, 'payload '+name)
            verified.append(name)
    for name in ('kv_data.py', 'prepare_kv_cpu.py'):
        raw = subprocess.check_output(['git','show',receipt['source_commit']+
              ':experiments/semantic_grounding_method/'+name], cwd=kd.ROOT)
        require(raw == members['source/'+name].encode(), 'executed source '+name)

    source_dir = kd.ROOT/'experiments/relation_frames_inventory/sources'
    original = {}
    for split, spec in sources['vsr']['files'].items():
        raw = (source_dir/(split+'.jsonl')).read_bytes()
        require(len(raw) == spec['bytes'] and hashlib.sha256(raw).hexdigest() == spec['sha256'], 'VSR '+split)
        original[split] = [json.loads(line) for line in raw.splitlines() if line.strip()]
    old = json.loads(gzip.decompress((source_dir.parent/'V2_MANIFEST.json.gz').read_bytes()))
    excluded = {kd.image_id(row) for split in ('dev','test') for row in original[split]}
    excluded |= {int(p[role]['image_id']) for p in old['heldout'] for role in ('positive','negative')}
    presence = json.loads(gzip.decompress((kd.ROOT/'experiments/concept_mass_grounding/TRAIN_PRESENCE.json.gz').read_bytes()))
    parent = json.loads(gzip.decompress((HERE/'data/gain_metadata.json.gz').read_bytes()))
    expected_train = [presence['train_rows'][int(i)] for i in np.random.default_rng(42).permutation(10000)[:4000]]
    rows = meta['rows']
    require(meta['train_indices'] == list(range(4000)) and meta['dev_indices'] == list(range(4000,5000)), 'native cohort indices')
    for actual, expected in zip(rows[:5000], expected_train + parent['groups']['val']):
        require(all(actual[k] == expected[k] for k in ('coco_id','filename','image_id')), 'native row identity')
        require(len(actual['references']) == len(actual['reference_tokens']) == 5, 'native references')
    ids = [r['coco_id'] for r in rows]
    require(len(ids) == len(set(ids)) == receipt['rows'] == 5472, 'unique staged rows')
    require(meta['coco_to_row'] == {str(iid):i for i,iid in enumerate(ids)}, 'ID index')
    require(meta['word2idx'] == parent['word2idx'], 'frozen vocabulary')
    require(set(banks) == set(map(str,ids)), 'bank coverage')
    row_by_id = {r['coco_id']:r for r in rows}
    for iid, bank in banks.items():
        image = row_by_id[int(iid)]
        require(len(bank) <= 6, 'top6 cap')
        require(bank == sorted(bank,key=lambda b:(-b['confidence'],b['original_index'])), 'bank order')
        for box in bank:
            require(set(box) == {'bbox_xyxy','confidence','original_index'}, 'class-free proposals')
            require(kd.clipped(box['bbox_xyxy'], image['width'], image['height']) == box['bbox_xyxy'], 'box bounds')

    parts = auxiliary['partitions']
    facts = parts['fit'] + parts['held']
    require(len({r['fact_id'] for r in facts}) == len(facts), 'deduplicated fact IDs')
    require(kd.partition(facts,set(ids[:4000])) == parts, 'image-held partition')
    require(not ({r['image_id'] for r in facts} & excluded), 'official/old held exclusion')
    for fact in facts:
        iid = fact['image_id']
        require(iid in row_by_id and fact['subject_slot'] != fact['object_slot'], 'distinct staged instances')
        require(max(fact['subject_slot'],fact['object_slot']) < len(banks[str(iid)]), 'predicted slots exist')
        key = kd.fact_key(fact)
        require(kd.identity(key) == fact['fact_id'], 'fact identity')
        require(fact['query_views'] == [kd.template(fact['subject'],fact['predicate'],fact['object']),
                kd.template(fact['object'],kd.INVERSE[fact['predicate']],fact['subject'])], 'inverse queries')
        require(not fact['transported_inverse_is_independent_observation'], 'no double-count claim')
        for position in fact['source_rows']:
            row = original['train'][position]
            require(kd.image_id(row) == iid and row['label'] == fact['label'] and kd.unanimous(row), 'observed truth')
            parsed = kd.parse_assertion(row, {fact['subject'],fact['object']})
            require(parsed is not None, 'source parse')
            _, subject, obj = parsed
            lexical = min((subject,row['relation'],obj),(obj,kd.INVERSE[row['relation']],subject))
            require(list(lexical) == fact['lexical_group'], 'source equivalent fact')
    for split, facts_in_split in parts.items():
        require(kd.multipositive_groups(facts_in_split) == auxiliary['multipositive'][split], 'observed MP groups')
    readiness = kd.readiness(parts)
    require(readiness == stats['readiness'] and readiness['counts'] == receipt['auxiliary'], 'readiness recomputed')
    require(readiness['passes_operational_gates'], 'all operational gates')
    represented = sum(len(r['source_rows']) for r in facts)
    quarantined_rows = sum(len(r['source_rows']) for r in auxiliary['quarantined'])
    require(represented+quarantined_rows == stats['matched'], 'source observation accounting')
    require(stats['matched']+stats['rejected']['no_distinct_predicted_pair'] == stats['candidates'], 'candidate accounting')
    require(sum(stats['rejected'].values())+stats['matched'] == len(original['train']), 'all original rows accounted')
    aux_ids = {r['image_id'] for r in facts}
    require(set(ids[5000:]) == aux_ids-set(ids[:5000]), 'aux-only rows')
    result = dict(status='CPU_METADATA_AUDIT_PASS', zip_sha256=cached['zip_sha256'],
                  receipt_sha256=hashlib.sha256(members['cpu_preparation.json'].encode()).hexdigest(),
                  assertions=checks, source_commit=receipt['source_commit'], seconds=receipt['seconds'],
                  caption_train=4000, caption_dev=1000, staged_rows=len(ids),
                  auxiliary=readiness['counts'], candidates=stats['candidates'],
                  matched_observations=stats['matched'], accepted_observations=represented,
                  quarantined_orbits=len(auxiliary['quarantined']),quarantined_observations=quarantined_rows,
                  fit_caption_train_overlap=len({r['image_id'] for r in parts['fit']}&set(ids[:4000])),
                  held_caption_dev_overlap=len({r['image_id'] for r in parts['held']}&set(ids[4000:5000])),
                  verified_review_payloads=verified, inputs_h5_sha256_reported=receipt['payload_sha256']['inputs.h5'],
                  hdf5_independently_checked=False, gt_iou_rematched_locally=False,
                  model_or_cuda_run=False, not_semantic_or_caption_success=True)
    kd.save(HERE/'KV_CPU_PREPARATION_RESULT.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    audit()
