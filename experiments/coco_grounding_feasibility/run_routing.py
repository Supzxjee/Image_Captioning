"""User-run B/R pilot. No model loading/training on import."""
import argparse
import json
import os
from pathlib import Path
from pilot_data import digest, save
from routing_data import read_manifest, tensors_for
import routing_stage as stage


def prepare(config):
    import h5py
    from captioning.visual_cache import VisualCache
    from run_pilot import corpus
    cfg = json.loads(Path(config).read_text())
    stage.verify_sources(cfg)
    if digest(Path(__file__).parent/'captioning/models.py') != cfg['models_source_sha256']:
        raise ValueError('Baseline model source differs.')
    if cfg['arm'] not in ('B','R') or cfg['aux_weight'] != .05:
        raise ValueError('Locked arms/auxiliary weight differ.')
    if digest(Path(cfg['chair'])/'chair.pkl') != '73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':
        raise ValueError('Different CHAIR cache.')
    cache = VisualCache([cfg['visual_cache']],id_key='coco_id')
    metadata = []
    try:
        for path in cache.paths:
            with h5py.File(path) as f:
                m = json.loads(f.attrs['metadata_json'])
                if not (m['model_id']=='openai/clip-vit-base-patch16' and m['output']=='vision_model.last_hidden_state'
                        and m['preprocessing']=='bilinear' and m['image_size']==[224,224] and m['antialias'] is True
                        and m['dataset_json_sha256']==stage.INPUT_HASHES['dataset_json']):
                    raise ValueError('Visual cache preprocessing incompatible with square patch masks.')
                metadata.append(dict(path=str(path),metadata=m))
    finally:
        cache.close()
    # Validate labels before the expensive 11k staging pass.
    dataset, tokenizer = corpus(cfg)
    cfg['groups'] = stage.locked_groups(dataset)
    read_manifest(cfg,dataset,tokenizer)
    stage.prepare(config)
    out = Path(cfg['output'])
    cfg = json.loads((out/'protocol.json').read_text())
    cfg.update(source_checkpoint_sha256=stage.INPUT_HASHES['checkpoint'],conditions=['on'],
        checkpoint_name='checkpoint.bin',checkpoint_format='coco_routing_pilot_v11',
        hidden_receipt_sha256=None,human_review_cohort_sha256=None,
        validation_scope='Historically reused development1000; not an independent test.',
        comparison_limits='B/R differ only in spatial-vs-bbox auxiliary loss. Historical A lacks byte-logged order.',
        test_used=False,auxiliary_scope='TRAIN noun spans only; no boxes/labels needed at inference.')
    save(out/'visual_metadata.json',metadata)
    save(out/'protocol.json',cfg)


def loader(cfg,dataset,tokenizer,split,rank,world):
    import torch
    from torch.utils.data import DataLoader, Dataset
    if world != 2:
        raise ValueError('One arm needs both T4 GPUs.')
    old = stage.loader(cfg,dataset,tokenizer,split,rank)
    if split != 'train':
        return old
    by_image, masks, slots, width = read_manifest(cfg,dataset,tokenizer)
    class Labeled(Dataset):
        def __len__(self):return len(old.dataset)
        def __getitem__(self,index):
            result = old.dataset[index]
            coco_id = cfg['groups']['train'][index]['coco_id']
            targets = tensors_for(by_image[coco_id],masks,slots,width)
            result.update({k:torch.from_numpy(v) for k,v in targets.items()})
            return result
    return DataLoader(Labeled(),batch_sampler=old.batch_sampler,num_workers=0,pin_memory=True)


def build(cfg,tokenizer):
    from routing_model import RoutingModel
    from run_route_smoke import build as build_base
    return RoutingModel(build_base(cfg,tokenizer),cfg['arm']).cuda().eval()


def predict(model,batches,tokenizer,condition):
    return stage.predict(model,batches,tokenizer,condition), []


def worker(protocol):
    from captioning.scst_runtime import model_digest
    import run_pilot as shared
    cfg = json.loads(Path(protocol).read_text())
    stage.verify_sources(cfg)
    out, rank = Path(cfg['output']), int(os.environ['RANK'])
    if json.loads((out/f'preflight_rank{rank}.json').read_text())['status'] != 'PREFLIGHT_PASS':
        raise ValueError('Do not train after failed/missing preflight.')
    captured = {}
    def build_recorded(config,tokenizer):
        model = build(config,tokenizer)
        captured.update(model=model,initial=model_digest(model))
        return model
    shared.build_model, shared.make_loader, shared.predict = build_recorded, loader, predict
    shared.worker(protocol)
    completion = json.loads((out/f'worker_complete_rank{rank}.json').read_text())
    completion.update(initial_full_sha256=captured['initial'],final_full_sha256=model_digest(captured['model']),
        final_base_sha256=model_digest(captured['model'].base),noun_receipt_sha256=cfg['noun_receipt_sha256'])
    save(out/f'worker_complete_rank{rank}.json',completion)


def evaluate(protocol):
    import run_hidden
    cfg = json.loads(Path(protocol).read_text());out = Path(cfg['output'])
    histories = [json.loads((out/f'history_rank{rank}.json').read_text()) for rank in (0,1)]
    for h in histories:
        if len(h)!=1250 or sum(r['global_spans'] for r in h)!=17904 or sum(r['global_person_spans'] for r in h)!=4476:
            raise ValueError('Incorrect number of updates/auxiliary exposures.')
    ids = [iid for h in histories for r in h for iid in r['local_image_ids']]
    if len(ids)!=10000 or len(set(ids))!=10000 or set(ids)!={r['image_id'] for r in cfg['groups']['train']}:
        raise ValueError('Missing/duplicate actual TRAIN image exposures.')
    for x,y in zip(*histories):
        if x['global_spans']!=y['global_spans'] or x['global_person_spans']!=y['global_person_spans']:
            raise ValueError('Rank count collective differs.')
    save(out/'exposure_check.json',dict(status='PASS',images=10000,unique_spans=17904,person_spans=4476,
        actual_order_logged=True,person_fraction_not_gradient_fraction=True))
    run_hidden.evaluate(protocol)
    summary = json.loads((out/'run_summary.json').read_text())
    for key in ('hidden_receipt_sha256','human_review_cohort_sha256'):
        summary.pop(key,None)
    summary.update(status='ROUTING_PILOT_COMPLETE_WAIT_B_R_COMPARISON',
        noun_receipt_sha256=cfg['noun_receipt_sha256'],aux_weight=.05,
        limits=[cfg['validation_scope'],cfg['comparison_limits'],
            'Weak noun-to-class annotations; attention path mass is not a causal explanation or verified grounding.',
            'No new inference parameters, no relation grounding claim; one seed. Gates need matched comparison.'])
    save(out/'run_summary.json',summary)


def archive(protocol):
    from run_hidden import archive as save_review
    save_review(protocol)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','preflight','worker','evaluate','archive'])
    p.add_argument('--config',required=True)
    a=p.parse_args()
    if a.mode=='preflight':
        from routing_preflight import preflight
        preflight(a.config)
    else:globals()[a.mode](a.config)
