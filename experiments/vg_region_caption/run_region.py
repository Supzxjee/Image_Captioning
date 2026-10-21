"""G/T auxiliary pilot; reuse A's staging, train loop and evaluator unchanged."""
import argparse
import json
from pathlib import Path
import os
import run_hidden as hidden
import run_pilot as shared
from hidden_data import INPUT_HASHES
from pilot_data import digest, save
from region_data import identity, patch_mask

LIVE_MODEL = None


def load_payload(cfg, dataset, tokenizer):
    raw = (Path(__file__).parent/'auxiliary_payload.json').read_bytes()
    manifest = json.loads((Path(__file__).parent/'AUXILIARY_MANIFEST.json').read_text())
    import hashlib
    if hashlib.sha256(raw).hexdigest() != manifest['auxiliary_payload_sha256']:
        raise ValueError('Auxiliary data differ from locked payload')
    data = json.loads(raw)
    if data['schema'] != 'vg_region_caption_aux_v1' or identity(tokenizer.word2idx) != data['word2idx_sha256']:
        raise ValueError('Vocabulary or payload schema differs')
    if data['train_image_ids'] != [r['coco_id'] for r in cfg['groups']['train']]:
        raise ValueError('Auxiliary schedule/caption cohort order differs')
    allowed = set(data['train_image_ids']) - set(data['exclusions'])
    if len(data['schedule']) != 10000 or identity(data['schedule']) != manifest['schedule_sha256']:
        raise ValueError('Schedule differs')
    by_coco = {r['cocoid']: r for r in dataset['images']}
    for row in data['bank']:
        if row['coco_id'] not in allowed or by_coco[row['coco_id']]['split'] not in ('train','restval'):
            raise ValueError('Heldout/exposed auxiliary image')
        encoded, _ = tokenizer.encode(row['phrase'])
        n = len(row['encoded'])
        if n > 30 or n < 3 or encoded[:n].tolist() != row['encoded'] or row['encoded'][-1] != tokenizer.eos_idx or tokenizer.word2idx['<unk>'] in row['encoded']:
            raise ValueError('Phrase encoding/OOV/EOS differs')
        if sum(patch_mask(row['region_normalized'])) < 4:
            raise ValueError('Region lacks four patch centers')
    if not all(isinstance(i, int) and 0 <= i < len(data['bank']) for i in data['schedule']):
        raise ValueError('Invalid schedule index')
    return data, manifest


def prepare(config):
    import h5py
    cfg = json.loads(Path(config).read_text())
    arm = cfg['arm']
    if arm not in ('G','T'):
        raise ValueError('Expected G or T')
    # Staging is exactly A's (including identity/reward/legacy/cache checks).
    stage_cfg = dict(cfg, arm='A')
    temporary = Path(config).with_name('caption_stage_'+arm+'.json')
    save(temporary, stage_cfg)
    hidden.prepare(temporary)
    out = Path(cfg['output'])
    protocol = out/'protocol.json'
    cfg = json.loads(protocol.read_text())
    dataset, tokenizer = shared.corpus(cfg)
    data, manifest = load_payload(cfg, dataset, tokenizer)
    # Fail before training if patch-coordinate preprocessing differs.
    from captioning.visual_cache import VisualCache
    cache = VisualCache([cfg['visual_cache']], id_key='coco_id')
    metadata = []
    try:
        for path in cache.paths:
            with h5py.File(path) as f:
                m = json.loads(f.attrs['metadata_json'])
                if not (m['model_id']=='openai/clip-vit-base-patch16' and m['output']=='vision_model.last_hidden_state'
                        and m['preprocessing']=='bilinear' and m['image_size']==[224,224] and m['antialias'] is True
                        and m['dataset_json_sha256']==INPUT_HASHES['dataset_json']):
                    raise ValueError('Cache geometry/preprocessing differs from square-resize patch coordinates')
                metadata.append(dict(path=path, metadata=m))
    finally:
        cache.close()
    with h5py.File(cfg['stage']) as f:
        if f['image_ids'][:].tolist() != [r['coco_id'] for r in cfg['groups']['train']+cfg['groups']['val']]:
            raise ValueError('Stage order differs from aux image mapping')
    save(out/'visual_metadata.json', metadata)
    save(out/'auxiliary_manifest.json', manifest)
    cfg.update(arm=arm, conditions=['on'], checkpoint_format='vg_region_caption_v1',
        phrase_weight=.1, auxiliary_schedule_sha256=manifest['schedule_sha256'],
        auxiliary_payload_sha256=manifest['auxiliary_payload_sha256'],
        aux_policy='Both omit CLS/legacy for auxiliary CE only; G masks original-region keys at every layer, T reads196patches; same10kphrases, no new parameters',
        loss='Caption CE + .05SCST after200updates (CIDErD+.5object_count_v2), plus .1 token-mean phrase CE includingEOS at every update',
        comparison_limits='G/T isolate region key selection given same text, not pure local evidence: CLIP patches already contextualized; weak VG labels; region CE is inherited, novelty unproven.',
        locked_numeric_gate=dict(G_minus_A_CIDEr_min=.01, G_minus_A_paired95_lower_gt=0.,
            G_CIDEr_gt_original=True, CHAIRs_increase_max=0., CHAIRi_increase_max=0., Recall_drop_max=.01,
            attribution_requires_G_minus_T_paired95_lower_gt=0., blind_review_required=True),
        evaluation_policy='Raw greedy original+end on reused dev1000; no test, no rerank/checkpoint selection; match A artifacts/runtime before attributing gains.')
    save(protocol, cfg)
    print('G/T DATA READY; auxiliary images/phrases/draws:', manifest['scheduled_unique_images'],
          manifest['scheduled_unique_phrases'], manifest['scheduled_draws'], flush=True)


def build_model(cfg, tokenizer):
    from captioning.models import UniversalVisionEncoder, CaptionDecoder, ImageCaptioningModel
    from captioning.checkpoints import load_checkpoint
    from captioning.scst_runtime import model_digest
    from region_model import RegionCaptionModel
    global LIVE_MODEL
    base = ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False, visual_adapter='qformer',
        num_visual_queries=32, qformer_layers=2), CaptionDecoder(len(tokenizer.word2idx)))
    load_checkpoint(base, cfg['checkpoint'], tokenizer)
    LIVE_MODEL = RegionCaptionModel(base, cfg['arm']).cuda().eval()
    LIVE_MODEL.initial_full_sha256 = model_digest(LIVE_MODEL)
    return LIVE_MODEL


def make_loader(cfg, dataset, tokenizer, split, rank, world):
    import torch
    from torch.utils.data import Dataset, DataLoader
    loader = hidden.make_loader(cfg, dataset, tokenizer, split, rank, world)
    if split != 'train':
        return loader
    data, _ = load_payload(cfg, dataset, tokenizer)
    positions = {r['coco_id']: i for i, r in enumerate(cfg['groups']['train'])}
    original = loader.dataset
    class Auxiliary(Dataset):
        def __len__(self): return len(original)
        def __getitem__(self, index):
            batch = original[index]
            row = data['bank'][data['schedule'][index]]
            visual_index = positions[row['coco_id']]
            tokens, _ = tokenizer.encode(row['phrase'])
            batch.update(aux_visual=torch.from_numpy(original.f['visual'][visual_index]),
                aux_patch_mask=torch.tensor(patch_mask(row['region_normalized']), dtype=torch.bool),
                aux_caption=tokens)
            return batch
    return DataLoader(Auxiliary(), batch_size=cfg['global_batch_size']//world, sampler=loader.sampler,
        num_workers=0, pin_memory=True, drop_last=False)


def worker(protocol):
    from captioning.scst_runtime import model_digest
    cfg = json.loads(Path(protocol).read_text())
    for name, expected in cfg['implementation_sha256'].items():
        if digest(Path(__file__).parent/name) != expected:
            raise ValueError('Runtime source changed: '+name)
    if digest(cfg['checkpoint']) != INPUT_HASHES['checkpoint'] or digest(cfg['stage']) != cfg['stage_sha256']:
        raise ValueError('Checkpoint or staged tensors changed')
    shared.build_model, shared.make_loader, shared.predict = build_model, make_loader, hidden.predict
    shared.worker(protocol)
    path = Path(cfg['output'])/f"worker_complete_rank{os.environ['RANK']}.json"
    done = json.loads(path.read_text())
    done.update(initial_full_sha256=LIVE_MODEL.initial_full_sha256,
        final_full_sha256=model_digest(LIVE_MODEL), final_base_sha256=model_digest(LIVE_MODEL.base))
    save(path, done)


def evaluate(protocol):
    hidden.evaluate(protocol)
    cfg = json.loads(Path(protocol).read_text())
    path = Path(cfg['output'])/'run_summary.json'
    result = json.loads(path.read_text())
    result['limits'] = [cfg['validation_scope'], cfg['comparison_limits'],
        'CHAIR evaluates object hallucination, not relation truth; no new grounding claim from phrase loss alone.']
    result['auxiliary_schedule_sha256'] = cfg['auxiliary_schedule_sha256']
    result['auxiliary_payload_sha256'] = cfg['auxiliary_payload_sha256']
    result['status'] = 'REGION_CAPTION_PILOT_COMPLETE_WAIT_MATCHED_G_T_A'
    save(path, result)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['prepare','worker','evaluate','archive'])
    p.add_argument('--config',required=True); args=p.parse_args()
    (hidden.archive if args.mode=='archive' else globals()[args.mode])(args.config)
