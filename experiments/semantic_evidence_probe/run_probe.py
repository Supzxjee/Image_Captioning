"""Read-only E matcher probe. Two independent inference workers; no optimizer or training."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from probe_data import digest, identity, summarize

CLIP = 'openai/clip-vit-base-patch16'
REVISION = '57c216476eefef5ab752ec549e440a49ae4ae5f3'


def load_manifest(cfg):
    for name, expected in cfg['source_sha256'].items():
        if digest(Path(__file__).parent/name) != expected:
            raise ValueError(f'Probe source changed: {name}')
    manifest = json.loads(Path(cfg['manifest']).read_text(encoding='utf-8'))
    if digest(cfg['manifest']) != cfg['manifest_sha256'] or identity(manifest['rows']) != manifest['rows_sha256']:
        raise ValueError('Probe manifest changed')
    assert manifest['alignment_train_overlap'] == 0 and not manifest['test_used']
    assert manifest['excluded_pilot_train_images'] == 10000
    assert len({r['case_id'] for r in manifest['rows']}) == len(manifest['rows'])
    return manifest


def worker(config, rank):
    import numpy as np
    import torch
    from transformers import CLIPTextModel, AutoTokenizer
    from captioning.visual_cache import VisualCache
    from captioning.scst_runtime import configure_strict_runtime, runtime_manifest, model_digest
    from pilot_model import EvidenceMemory
    cfg = json.loads(Path(config).read_text())
    manifest = load_manifest(cfg)
    if digest(cfg['checkpoint']) != manifest['checkpoint_sha256']:
        raise ValueError('Expected the exact completed E checkpoint, not A or the original SCST checkpoint')
    configure_strict_runtime(True)
    torch.cuda.set_device(rank)
    torch.manual_seed(42)
    device = f'cuda:{rank}'
    checkpoint = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True)
    assert checkpoint['format'] == 'semantic_evidence_pilot_v1' and checkpoint['arm'] == 'E'
    assert checkpoint['completed_updates'] == 1250
    assert checkpoint['protocol_sha256'] == manifest['checkpoint_protocol_sha256']
    assert checkpoint['cohort_sha256'] == manifest['checkpoint_cohort_sha256']
    module = EvidenceMemory().to(device).eval().requires_grad_(False)
    state = {k.removeprefix('extra.'):v for k,v in checkpoint['model_state_dict'].items() if k.startswith('extra.')}
    module.load_state_dict(state, strict=True)
    del checkpoint, state
    initial_digest = model_digest(module)
    encoder = CLIPTextModel.from_pretrained(CLIP, revision=REVISION).to(device).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(CLIP, revision=REVISION)
    image_ids = sorted({r['coco_id'] for r in manifest['rows']})[rank::2]
    selected = set(image_ids)
    rows = [r for r in manifest['rows'] if r['coco_id'] in selected]
    texts = sorted({c['text'] for row in rows for c in row['candidates']})
    embeddings = {}
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            batch = texts[start:start+64]
            token = tokenizer(batch, padding=True, truncation=True, max_length=77, return_tensors='pt').to(device)
            for text, vector in zip(batch, encoder(**token).pooler_output.float()):
                embeddings[text] = vector.detach().clone()
    del encoder
    cache = VisualCache([cfg['visual_cache']], id_key='coco_id')
    cache.require_ids(image_ids, 'locked evidence diagnostic')
    groups = defaultdict(list)
    for row in rows: groups[row['coco_id']].append(row)
    results = []
    try:
        # Read in source-row order without copying the multi-GB visual cache to /tmp.
        for count, image_id in enumerate(sorted(image_ids, key=lambda i:cache.index[i]), 1):
            image = torch.from_numpy(cache.read(image_id)[1:]).to(device)
            assert image.shape == (196,768) and torch.isfinite(image).all()
            for row in groups[image_id]:
                text = torch.stack([embeddings[c['text']] for c in row['candidates']])
                boxes = torch.tensor([c['bounds'] for c in row['candidates']], device=device)
                patches = image[None].expand(2,-1,-1)
                roles = torch.full((2,), 2, dtype=torch.long, device=device)
                scores = {}
                with torch.inference_mode():
                    for method in ('boxed', 'global_view', 'zero_visual'):
                        b, p = boxes, patches
                        if method == 'global_view':
                            b = torch.zeros_like(boxes); b[:,2,2:] = 1
                        if method == 'zero_visual': p = torch.zeros_like(patches)
                        _, values, _ = module.attend(p, text, roles, b)
                        assert torch.isfinite(values).all()
                        scores[method] = values.cpu().tolist()
                results.append(dict(row, scores=scores))
            if count == 1 or count % 10 == 0 or count == len(image_ids):
                print(f'GPU{rank}: {count}/{len(image_ids)} images; {len(results)} pairs', flush=True)
    finally:
        cache.close()
    assert model_digest(module) == initial_digest
    runtime = runtime_manifest()
    runtime['worker_gpu'] = dict(rank=rank, name=torch.cuda.get_device_name(rank))
    payload = dict(rank=rank, records=results, no_training=True, parameters_unchanged=True,
                   module_state_sha256=initial_digest, runtime=runtime)
    out = Path(cfg['output'])
    temporary = out/f'rank{rank}.json.partial'
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(out/f'rank{rank}.json')


def run(config):
    cfg = json.loads(Path(config).read_text())
    manifest = load_manifest(cfg)
    out = Path(cfg['output'])
    if out.exists(): raise FileExistsError(f'Use a fresh output: {out}')
    out.mkdir(parents=True)
    logs, workers = [], []
    try:
        for rank in (0,1):
            log = (out/f'worker{rank}.log').open('w',encoding='utf-8'); logs.append(log)
            workers.append(subprocess.Popen([sys.executable,'-u',__file__,'--config',str(Path(config).resolve()),'--rank',str(rank)], stdout=log, stderr=subprocess.STDOUT))
        last = 0
        while any(p.poll() is None for p in workers):
            if any(p.poll() not in (None, 0) for p in workers):
                raise RuntimeError('Inference worker failed; inspect worker logs')
            if time.monotonic()-last >= 10:
                for rank in (0,1):
                    lines = (out/f'worker{rank}.log').read_text(encoding='utf-8', errors='replace').splitlines()
                    print(f'Worker{rank}: '+(lines[-1] if lines else 'loading pinned checkpoint/text encoder'), flush=True)
                last = time.monotonic()
            time.sleep(1)
        if any(p.returncode for p in workers): raise RuntimeError('Worker failed')
    finally:
        for process in workers:
            if process.poll() is None: process.terminate(); process.wait()
        for log in logs: log.close()
        for rank in (0,1):
            path=out/f'worker{rank}.log'
            if path.exists(): print(path.read_text(encoding='utf-8',errors='replace')[-2000:], flush=True)
    shards = [json.loads((out/f'rank{rank}.json').read_text()) for rank in (0,1)]
    assert shards[0]['module_state_sha256'] == shards[1]['module_state_sha256']
    rows = sorted([r for s in shards for r in s['records']], key=lambda r:(r['task'],r['case_id']))
    assert [{k:v for k,v in r.items() if k!='scores'} for r in rows] == manifest['rows']
    summary = dict(status='FROZEN_MATCHER_DIAGNOSTIC_COMPLETE', metrics=summarize(rows),
        checkpoint_sha256=manifest['checkpoint_sha256'], manifest_sha256=cfg['manifest_sha256'],
        no_training=True, parameters_unchanged=True, test_used=False,
        limitations=[manifest['caption_pretraining_overlap'], manifest['scope'],
            'Holding/riding labels follow V-COCO ontology; not an exhaustive universal relation truth.',
            'Cosine pair ranking is not calibrated confidence or caption-quality evidence.',
            'GT/human boxes differ from UPT detector boxes used during E training.',
            'Zero visual features are an out-of-distribution lexical/bias control, not a fair standalone model.',
            'Global-view removes box membership/masks; unchanged scores may reflect patch/box resolution.',
            'Reused selected diagnostic, limited actions; bootstrap samples images, not training seeds.'])
    for name, value in [('scores.json',rows),('summary.json',summary)]:
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False),encoding='utf-8')
    shutil.copyfile(cfg['manifest'], out/'probe_manifest.json')
    shutil.copyfile(config, out/'config.json')
    source_dir = out/'source'; source_dir.mkdir()
    for name in cfg['source_sha256']:
        shutil.copyfile(Path(__file__).parent/name, source_dir/name)
    print(json.dumps(summary,indent=2),flush=True)
    archive=shutil.make_archive(str(out)+'_review','zip',out)
    print('Download:',archive,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--rank',type=int,choices=(0,1))
    a=p.parse_args()
    if a.rank is None: run(a.config)
    else: worker(a.config,a.rank)
