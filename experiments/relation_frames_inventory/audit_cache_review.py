"""Validate cached small ZIP metadata. No tensors or model runtime are loaded."""
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    cached=json.loads((HERE/'review/cache_loaded.json').read_text(encoding='utf-8'));texts=cached['texts']
    receipt=json.loads(texts['receipt.json']);execution=json.loads(texts['execution_status.json'])
    package=json.loads((HERE/'PREPARATION_NOTEBOOK_PACKAGE.json').read_text(encoding='utf-8'))
    source=Path('C:/Users/leduc/Downloads/relation_frames_v2_cache_review.zip')
    # The binary manifest was not loaded in the initial text-only read; read that member once.
    with zipfile.ZipFile(source) as z:raw_manifest=z.read('relation_frames_manifest.bin')
    if raw_manifest!=(HERE/'V2_MANIFEST.json.gz').read_bytes():raise ValueError('Manifest bytes differ')
    if execution['returncode']!=0 or execution['source_commit']!=package['source_commit']:raise ValueError('Wrong execution')
    if receipt['status']!='CACHE_COPY_COMPLETE_PENDING_LABEL_AUDIT' or receipt['images']!=612 or receipt['legacy_prompt_coverage']!=612:raise ValueError('Incomplete cache')
    if any(receipt[k] for k in ('feature_extraction','caption_training','neural_forward')):raise ValueError('Unexpected model execution')
    if receipt['preparation_source_sha256']!=package['assets']['prepare_frames_cpu.py']['sha256']:raise ValueError('Wrong preparation code')
    for name,expected in receipt['payload_sha256'].items():
        if name=='inputs.h5':continue
        raw=raw_manifest if name.endswith('.bin') else texts[name].encode('utf-8')
        if sha(raw)!=expected:raise ValueError('Payload hash differs: '+name)
    manifest=json.loads(gzip.decompress(raw_manifest));images=json.loads(texts['images.json']);bank=json.loads(texts['detector_boxes.json'])
    want={r['image_id']:r['image'] for split in ('train','heldout') for p in manifest[split] for r in (p['positive'],p['negative'])}
    ids=[r['coco_id'] for r in images]
    if ids!=sorted(want) or len(ids)!=612 or set(bank)!=set(map(str,ids)):raise ValueError('Wrong image bindings')
    for row in images:
        original=want[row['coco_id']]
        if (row['filename'],row['width'],row['height'])!=(original['file_name'],original['width'],original['height']):raise ValueError('Image identity differs')
    target_checks=0
    for split in ('train','heldout'):
        for p in manifest[split]:
            for r in (p['positive'],p['negative']):
                objects=bank[str(r['image_id'])];lookup={o['original_index']:o for o in objects}
                if len(objects)>12 or len(lookup)!=len(objects):raise ValueError('Wrong predicted bank')
                for role in ('subject','object'):
                    e=r['predicted'][role];o=lookup[e['index']]
                    if o['label']!=e['label'] or o['bbox_xyxy']!=e['bbox_xyxy']:raise ValueError('Pair points to another predicted instance')
                    target_checks+=1
    human=json.loads((HERE/'HUMAN_AUDIT32_RESULT.json').read_text(encoding='utf-8'))
    if human['status']!='HUMAN_REVIEW_COMPLETE' or human['effective_confirmations']!=32:raise ValueError('Human review incomplete')
    result=dict(status='DATA_AND_CACHE_METADATA_ACCEPTED_MODEL_PREFLIGHT_PENDING',zip_sha256=cached['zip_sha256'],
        receipt_sha256=sha(texts['receipt.json'].encode()),source_commit=execution['source_commit'],
        preparation_source_sha256=receipt['preparation_source_sha256'],images=612,original_prompt_coverage=612,
        predicted_target_bindings_checked=target_checks,seconds=receipt['seconds'],
        checked_payload_sha256={n:h for n,h in receipt['payload_sha256'].items() if n!='inputs.h5'},
        declared_payload_sha256=receipt['payload_sha256'],human_review_status=human['status'],
        human_review_receipt_sha256=sha((HERE/'HUMAN_AUDIT32_RESULT.json').read_bytes()),
        original_prep_human_flag='false is a historical preparation snapshot; separate completed human audit applies',
        limits=['inputs.h5 absent from small ZIP: SHA/readback reported by pinned worker, not independently rehashed here',
                'Before GPU, verify full HDF5 hash, schema and data against this receipt',
                'Preparation contains no model training or evidence of caption/grounding gain'],
        local_neural_execution=False)
    (HERE/'CACHE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
