"""CPU-only packaging of a locked VG schedule; never runs a model."""
import json
from pathlib import Path
import sys
import zlib
from region_data import select, schedule, vocabulary, identity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/upt_hidden_pair'))
sys.path.insert(0, str(ROOT / 'experiments/upt_alignment_pilot'))
from hidden_data import locked_groups, DATASET_SHA
from pilot_data import digest


def main():
    dataset_path = Path('C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/sources/dataset_coco.json')
    assert digest(dataset_path) == DATASET_SHA
    dataset = json.loads(dataset_path.read_text(encoding='utf-8'))
    groups = locked_groups(dataset)
    audit = ROOT / 'experiments/vg_phrase_audit'
    inventory = audit / 'review/region24/inventory.jsonl'
    excluded = set(json.loads((ROOT / 'experiments/upt_hidden_pair/excluded_phrase_ids.json').read_text()))
    viewed = set()
    for name in ('SAMPLE30_MANIFEST.json', 'REGION24_MANIFEST.json'):
        viewed.update(r['coco_id'] for r in json.loads((audit / name).read_text(encoding='utf-8')))
    excluded.update(viewed)
    vocab = vocabulary(dataset)
    with inventory.open(encoding='utf-8') as stream:
        bank, report = select(map(json.loads, stream), vocab, {r['coco_id'] for r in groups['train']}, excluded)
    indices = schedule(bank)
    used = sorted(set(indices))
    remap = {old: new for new, old in enumerate(used)}
    compact = [{k:v for k,v in bank[i].items() if k != 'patch_mask'} for i in used]
    payload = dict(schema='vg_region_caption_aux_v1', bank=compact,
        schedule=[remap[i] for i in indices], word2idx_sha256=identity(vocab),
        train_image_ids=[r['coco_id'] for r in groups['train']],
        exclusions=sorted(excluded), viewed_audit_image_ids=sorted(viewed))
    raw = (json.dumps(payload, ensure_ascii=False, separators=(',', ':'))+'\n').encode('utf-8')
    (HERE / 'auxiliary_payload.bin').write_bytes(zlib.compress(raw, 9))
    report.update(status='LOCKED_DATA_NOT_TRAINED', dataset_json_sha256=DATASET_SHA,
        source_inventory_sha256=digest(inventory), source_inventory_regions=168522,
        source_annotation_zip_sha256='a34a7623a1c03d89d4f7df10b7ed7945eb2d7e67084eee249ab8e3433e6a6dcf',
        source_metadata_zip_sha256='b87a94918cb2ff4d952cf1dfeca0b9cf6cd6fd204c2f8704645653be1163681a',
        word2idx_sha256=identity(vocab), auxiliary_payload_sha256=__import__('hashlib').sha256(raw).hexdigest(),
        compressed_payload_sha256=digest(HERE/'auxiliary_payload.bin'),
        scheduled_draws=len(indices), scheduled_unique_phrases=len(compact),
        scheduled_unique_images=len({r['coco_id'] for r in compact}),
        schedule_sha256=identity(payload['schedule']), excluded_viewed_audit_images=len(viewed),
        sampling='Image-first balanced hash cycles; max5 phrases/image in bank. One aux draw per caption image index, same caption DistributedSampler seed42; global8 per step,1250updates.',
        supervision='Weak original VG annotations, NOT visually verified GT. All54 audit images excluded uniformly without consulting their labels.',
        max_phrase_words=28, max_encoded_length=30, oov_fraction_allowed=0., min_patch_centers=4,
        global_aux_draws=10000, phrase_weight=.1, test_used=False, model_executed=False)
    (HERE/'AUXILIARY_MANIFEST.json').write_bytes((json.dumps(report, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
