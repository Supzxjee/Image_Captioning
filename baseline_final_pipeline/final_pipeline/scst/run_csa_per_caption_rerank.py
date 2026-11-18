import argparse, hashlib, io, json, os, time
from pathlib import Path
import numpy as np
from real_bank_data import read,save,sha
from csa_per_caption_rerank_data import checked_lock, read_beams, POLICIES
from captioning.multiscore_reranking import rerank_multiscore
from score_prompt_test_extra import java_path

def fixed_rank(row, clipscore, detected):
    if len(row['candidates']) != 5 or len(clipscore) != 5 or any((not np.isfinite(x) or not 0 <= x <= 2.50001 for x in clipscore)):
        raise ValueError('Invalid TEST CLIP scores/width')
    scores = [c['logprob'] for c in row['candidates']]
    if not all(np.isfinite(scores)) or scores != sorted(scores, reverse=True):
        raise ValueError('Invalid beam logprobs')
    enriched = [dict(c, clipscore=float(s)) for c, s in zip(row['candidates'], clipscore)]
    chosen, _ = rerank_multiscore(dict(candidates=enriched), detected, 0.4, 0.3)
    return int(chosen['original_rank'])

def prefetch(cfg):
    from huggingface_hub import snapshot_download
    out = Path(cfg['output'])
    locked = checked_lock(out)['clip']
    root = Path(snapshot_download(locked['id'], revision=locked['revision'], allow_patterns=list(locked['files_sha256']), cache_dir=cfg['model_cache']))
    for n, h in locked['files_sha256'].items():
        if sha(root / n) != h:
            raise ValueError('CLIP assets differ from VALID164: ' + n)
    save(out / 'clip_assets.json', dict(**locked, path=str(root)))
    print('CLIP model/processor match frozen VALID164 bytes; no BLIP download.', flush=True)

def text_preflight(cfg):
    from transformers import CLIPProcessor
    out = Path(cfg['output'])
    lock = checked_lock(out)
    rows = read_beams(out, lock)
    model = read(out / 'clip_assets.json')
    processor = CLIPProcessor.from_pretrained(model['path'], local_files_only=True, use_fast=False)
    lengths = [len(v) for v in processor.tokenizer([c['caption'] for r in rows for c in r['candidates']], truncation=False)['input_ids']]
    if len(lengths) != 25000 or max(lengths) > 77:
        raise ValueError('Would truncate TEST candidates')
    save(out / 'token_coverage.json', dict(images=5000, captions=25000, max_tokens=max(lengths), limit=77))

def worker(cfg):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    rank = int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 2 or torch.cuda.device_count() != 2:
        raise ValueError('Select T4x2')
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(cfg['output'])
    lock = checked_lock(out)
    asset = read(out / 'clip_assets.json')
    for n, h in lock['clip']['files_sha256'].items():
        if sha(Path(asset['path']) / n) != h:
            raise ValueError('Changed CLIP model/processor')
    base = CLIPModel.from_pretrained(asset['path'], local_files_only=True, use_safetensors=False, attn_implementation='eager').float().to(device).eval().requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(asset['path'], local_files_only=True, use_fast=False)
    if sha(out / 'inference_inputs.json') != lock['inference_sha256']:
        raise ValueError('Changed TEST inference inputs')
    inputs = read(out / 'inference_inputs.json')
    beams = read_beams(out, lock)
    started = time.monotonic()
    count = 0
    path = out / f'scores_rank{rank}.jsonl'
    with path.open('x', encoding='utf-8') as log, torch.inference_mode():
        for row, beam in list(zip(inputs, beams))[rank::2]:
            if time.monotonic() - started > cfg['phase_seconds']:
                raise TimeoutError('TEST CLIP scoring cap; preserve partial scores')
            raw = Path(row['image_path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row['image_sha256']:
                raise ValueError('Changed raw TEST image')
            with Image.open(io.BytesIO(raw)) as image:
                image = image.convert('RGB')
            pixels = processor(images=image, return_tensors='pt')['pixel_values'].to(device)
            text = processor.tokenizer([c['caption'] for c in beam['candidates']], padding=True, truncation=False, return_tensors='pt').to(device)
            image_features = F.normalize(base.get_image_features(pixel_values=pixels).float(), dim=-1)
            text_features = F.normalize(base.get_text_features(**text).float(), dim=-1)
            clipscore = (2.5 * (image_features * text_features).sum(-1).clamp_min(0)).cpu().tolist()
            chosen = fixed_rank(beam, clipscore, row['detected'])
            log.write(json.dumps(dict(coco_id=row['coco_id'], image_sha256=row['image_sha256'], clipscore=clipscore, selected_rank=chosen), allow_nan=False) + '\n')
            log.flush()
            count += 1
            if count % 100 == 0:
                print('TEST CLIP GPU', rank, count, '/2500', flush=True)
            if count == 25 and (time.monotonic() - started) / count * (2500 - count) > cfg['phase_seconds'] - (time.monotonic() - started):
                raise TimeoutError('Measured TEST CLIP projection exceeds cap')
    save(out / f'worker{rank}.json', dict(complete=True, images=count, scored_pairs=count * 5, scores_sha256=sha(path), seconds=time.monotonic() - started, torch_version=torch.__version__, training=False, clip_revision=lock['clip']['revision']))

def score(cfg):
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.spice.spice import Spice
    java_path(Path(cfg['test_output']))

    import subprocess, sys
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from saved_quality_data import chair_counts
    out = Path(cfg['output'])
    lock = checked_lock(out)
    inputs = read(out / 'inference_inputs.json')
    beams = read_beams(out, lock)
    if sha(out / 'inference_inputs.json') != lock['inference_sha256'] or sha(out / 'test_references.json') != lock['references_sha256']:
        raise ValueError('Changed locked inputs')
    scored = {}
    for rank in (0, 1):
        w = read(out / f'worker{rank}.json')
        path = out / f'scores_rank{rank}.jsonl'
        if not w['complete'] or w['images'] != 2500 or w['scored_pairs'] != 12500 or (sha(path) != w['scores_sha256']):
            raise ValueError('Incomplete TEST scoring worker')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if [r['coco_id'] for r in rows] != [r['coco_id'] for r in inputs[rank::2]]:
            raise ValueError('Wrong TEST score IDs')
        scored.update({r['coco_id']: r for r in rows})
    if len(scored) != 5000:
        raise ValueError('Missing TEST scores')
    chosen = []
    for row, beam in zip(inputs, beams):
        s = scored[row['coco_id']]
        if s['image_sha256'] != row['image_sha256']:
            raise ValueError('Wrong image-score identity')
        k = fixed_rank(beam, s['clipscore'], row['detected'])
        if k != s['selected_rank']:
            raise ValueError('Fixed selection replay differs')
        chosen.append(k)
    save(out / 'selected_ranks.json', chosen)
    refs = read(out / 'test_references.json')
    ptb = PTBTokenizer()
    ref = ptb.tokenize({r['coco_id']: [dict(caption=c) for c in r['references']] for r in refs})
    chair = Path(cfg['chair'])
    metrics = {}
    flags = {}
    if sha(chair / 'chair.pkl') != '73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':
        raise ValueError('Changed CHAIR cache')
    for policy in POLICIES:
        selected = [dict(coco_id=r['coco_id'], caption=r['candidates'][0 if policy == 'decoder_rank0' else k]['caption']) for r, k in zip(beams, chosen)]
        hyp = ptb.tokenize({r['coco_id']: [dict(caption=r['caption'])] for r in selected})
        c, per = Cider().compute_score(ref, hyp)
        b, _ = Bleu(4).compute_score(ref, hyp)
        r, _ = Rouge().compute_score(ref, hyp)
        ci = out / (policy + '_chair_input.json')
        co = out / (policy + '_chair_output.json')
        save(ci, [dict(image_id=r['coco_id'], caption=r['caption']) for r in selected])
        subprocess.run([sys.executable, '-u', chair / 'chair.py', '--cap_file', ci, '--cache', chair / 'chair.pkl', '--save_path', co], cwd=chair, check=True, timeout=300)
        ch = read(co)
        m, counts = chair_counts(ch, selected)
        meteor = Meteor()
        try:
            meteor_score, meteor_per = meteor.compute_score(ref, hyp)
        finally:
            del meteor
        spice_score, spice_per = Spice().compute_score(ref, hyp)
        if len(meteor_per) != 5000 or len(spice_per) != 5000 or not np.isfinite([meteor_score, spice_score]).all():
            raise ValueError('Incomplete METEOR/SPICE')
        metrics[policy] = dict(CIDEr=float(c), BLEU_1=float(b[0]), BLEU_4=float(b[3]), METEOR=float(meteor_score), SPICE=float(spice_score), ROUGE_L=float(r), **m, counts=counts, mean_words=sum((len(r['caption'].split()) for r in selected)) / 5000, short_le2=sum((len(r['caption'].split()) <= 2 for r in selected)))
        save(out / (policy + '_captions.json'), selected)
        save(out / (policy + '_cider_per_image.json'), [dict(coco_id=int(i), CIDEr=float(v)) for i, v in zip(ref, per)])
        flags[policy] = {r['image_id']: bool(r['mscoco_hallucinated_words']) for r in ch['sentences']}
    a, b = [flags[p] for p in POLICIES]
    changes = dict(changed_captions=sum((k != 0 for k in chosen)), rank_histogram={str(k): chosen.count(k) for k in range(5)}, CHAIR_repairs=sum((a[i] and (not b[i]) for i in a)), CHAIR_new_flags=sum((not a[i] and b[i] for i in a)))
    result = dict(status='CSA_PER_CAPTION_FIXED_RERANK_VALID5000_COMPLETE', images=5000, test_used=False, training=False, checkpoint_sha256=lock['checkpoint_sha256'], metrics=metrics, changes=changes, selector=lock['selector'], tuned_on_test=False, oracle_used_for_selection=False, prior_project_test_exposure=True, selection='both prelocked policies reported; fixed VALID164 policy', NLL_note='same frozen captioner; reranking changes output only, no target-conditioned pass')
    save(out / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prefetch', 'text_preflight', 'worker', 'score'))
    parser.add_argument('config')
    args = parser.parse_args()
    globals()[args.mode](read(args.config))
