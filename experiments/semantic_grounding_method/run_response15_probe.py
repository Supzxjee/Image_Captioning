"""User-run frozen C0 diagnostic. No neural import or execution at module import."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
from response15_data import assets, contrast, save, sha, summary, tensor_map_digest

HERE = Path(__file__).resolve().parent
C0_SHA = '5d92cd5cb48077c7332bb3ff9b295cad1df5d76ede818adabef28035f055fa9d'
PROMPT_SHA = 'd7910e9bf87e3aad86d806c77ec336b34793e3bef68a3f662fdcb2e9456ca38d'
CLIP_REV = '57c216476eefef5ab752ec549e440a49ae4ae5f3'


def run(cfg, out):
    started = time.monotonic()
    receipt = json.loads((HERE / 'RESPONSE15_ASSET_RECEIPT.json').read_bytes())
    root, manifest = assets(cfg['assets_input'], Path(cfg['scratch']) / 'unpacked_assets', receipt)
    for key, expected in [('checkpoint', C0_SHA), ('legacy_prompt', PROMPT_SHA)]:
        print('Hashing pinned input:', key, flush=True)
        if sha(cfg[key]) != expected:
            raise ValueError('Wrong pinned input: ' + key)
    save(out / 'asset_manifest.json', manifest)
    # Neural code is reached only in the user-run worker, after cheap asset checks.
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel
    from captioning.models import UniversalVisionEncoder, CaptionDecoder, ImageCaptioningModel
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.scst_runtime import configure_strict_runtime, model_digest, runtime_manifest, tensor_digest
    configure_strict_runtime(True)
    if not torch.cuda.is_available():
        raise RuntimeError('Select a Kaggle GPU; no CPU model fallback')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device('cuda:0')
    state = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
    if state.get('arm') != 'C0' or state.get('updates') != 500:
        raise ValueError('Expected exact C0 endpoint500')
    vocab = state['caption_word2idx']
    vh = hashlib.sha256(json.dumps(vocab, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    if vh != manifest['vocab_sha256']:
        raise ValueError('Native caption vocabulary differs')
    weights = state['model_state_dict']
    if not weights or any(not k.startswith('base.') for k in weights):
        raise ValueError('Wrong native checkpoint wrapper')
    model = ImageCaptioningModel(
        UniversalVisionEncoder(load_backbone=False, visual_adapter='qformer', num_visual_queries=32, qformer_layers=2),
        CaptionDecoder(10000))
    bare_weights = {k[5:]: v for k, v in weights.items()}
    model.load_state_dict(bare_weights, strict=True)
    model.requires_grad_(False).eval()
    # Historical9693... is a base+teacher wrapper digest, not this bare captioner.
    # Bind the loaded bare model to every tensor in the SHA-pinned C0 file instead.
    expected_tensors = {k: tensor_digest(v) for k, v in bare_weights.items()}
    actual_tensors = {k: tensor_digest(v) for k, v in model.state_dict().items()}
    initial_digest = tensor_map_digest(actual_tensors)
    identity = dict(scope='bare ImageCaptioningModel; no teacher wrapper',
        checkpoint_sha256=C0_SHA, expected_tensor_count=len(expected_tensors),
        actual_tensor_count=len(actual_tensors),
        expected_from_pinned_checkpoint=tensor_map_digest(expected_tensors),
        actual_loaded=initial_digest, exact_tensor_map_match=expected_tensors == actual_tensors)
    save(out / 'identity.json', identity)
    print('Bare C0 identity:', identity, flush=True)
    if not identity['exact_tensor_map_match']:
        raise ValueError('Loaded bare C0 tensors differ from pinned checkpoint; see identity.json')
    del state, weights, bare_weights, expected_tensors, actual_tensors
    rows = manifest['pairs']
    encoded = {r['sample']: [contrast(r['changed_before'], r['changed_after'], vocab),
                              contrast(r['stable_true'], r['stable_false'], vocab)] for r in rows}
    print('Loading native prompt cache once; no text from labels used as prompt.', flush=True)
    cache = torch.load(cfg['legacy_prompt'], map_location='cpu', weights_only=False)
    aligned = align_prompt_cache(cache['data'], [r['filename'] for r in rows])
    missing = [r['filename'] for r in rows if r['filename'] not in aligned]
    if missing:
        raise ValueError('Missing original native prompts: ' + repr(missing))
    prompts = {}
    for r in rows:
        p = aligned[r['filename']]
        tokens, mask = p['tokens'].float().clone(), p['mask'].clone()
        if tokens.shape != (20, 512) or mask.shape != (20,) or not torch.isfinite(tokens).all():
            raise ValueError('Wrong native prompt shape/values')
        if not ((mask == 0) | (mask == 1)).all() or not mask.any():
            raise ValueError('Wrong native prompt mask')
        prompts[r['sample']] = (tokens, mask.bool())
    del aligned, cache
    model.to(device)
    print('Prefetching pinned CLIP and encoding30source JPEGs.', flush=True)
    clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch16', revision=CLIP_REV).requires_grad_(False).eval().to(device)
    mean = torch.tensor([.48145466, .4578275, .40821073], device=device).view(3, 1, 1)
    std = torch.tensor([.26862954, .26130258, .27577711], device=device).view(3, 1, 1)
    images = out / 'images'
    images.mkdir()
    features = {}
    with torch.inference_mode():
        for r in rows:
            pixels = []
            for side in ('original', 'edited'):
                src = root / r[side + '_file']
                shutil.copyfile(src, images / (r['sample'] + '_' + side + '.jpg'))
                with Image.open(src) as im:
                    arr = np.asarray(im.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)).copy()
                x = torch.from_numpy(arr).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.
                pixels.append((x-mean)/std)
            f = clip.vision_model(pixel_values=torch.stack(pixels)).last_hidden_state
            if f.shape != (2, 197, 768) or not torch.isfinite(f).all():
                raise ValueError('Invalid CLIP output')
            features[r['sample']] = f.half().float().cpu()
    del clip
    torch.cuda.empty_cache()
    inverse = {i: w for w, i in vocab.items()}

    def captions(memory, padding):
        ids = torch.full((len(memory), 1), vocab['<bos>'], device=device, dtype=torch.long)
        live = torch.ones(len(memory), device=device, dtype=torch.bool)
        for _ in range(29):
            logits = model.decoder(ids, memory, memory_key_padding_mask=padding)[:, -1]
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite generation')
            nxt = torch.where(live, logits.argmax(-1), vocab['<pad>'])
            ids = torch.cat([ids, nxt[:, None]], 1)
            live &= nxt.ne(vocab['<eos>'])
            if not live.any():
                break
        result = []
        for row in ids.cpu().tolist():
            words = []
            for i in row[1:]:
                if i == vocab['<eos>']:
                    break
                if i not in (vocab['<bos>'], vocab['<pad>']):
                    words.append(inverse[i])
            result.append(' '.join(words))
        return result

    def scores(enc, memory, padding):
        ids = torch.tensor(enc['ids'], device=device, dtype=torch.long)
        logits = model.decoder(ids[:-1][None].expand(len(memory), -1), memory, memory_key_padding_mask=padding)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite fact logits')
        selected = logits.float().log_softmax(-1).gather(-1, ids[1:][None, :, None].expand(len(memory), -1, 1)).squeeze(-1)
        return selected[:, enc['start']:].mean(-1).cpu().tolist()

    records = []
    gpu_started = time.monotonic()
    with torch.inference_mode():
        for r in rows:
            prompt, mask = prompts[r['sample']]
            p = prompt[None].expand(2, -1, -1).to(device)
            pm = mask[None].expand(2, -1).to(device)
            f = features[r['sample']].to(device)
            if not records:
                same, pad = model.build_memory(f[:1].expand(2, -1, -1), p, pm)
                ids = torch.tensor(encoded[r['sample']][0][0]['ids'][:-1], device=device)[None].expand(2, -1)
                logits = model.decoder(ids, same, memory_key_padding_mask=pad)
                torch.testing.assert_close(logits[:1], logits[1:], rtol=1e-5, atol=1e-5)
                c = captions(same, pad)
                if c[0] != c[1]:
                    raise ValueError('Same-input greedy sanity failed')
                save(out / 'sanity.json', dict(same_input_logits_close=True, same_input_greedy_equal=True, rtol=1e-5, atol=1e-5))
            memory, padding = model.build_memory(f, p, pm)
            ca, cb = [scores(e, memory, padding) for e in encoded[r['sample']][0]]
            sa, sb = [scores(e, memory, padding) for e in encoded[r['sample']][1]]
            cap = captions(memory, padding)
            item = dict(sample=r['sample'], coco_id=r['coco_id'], vg_id=r['vg_id'],
                expected={k: r[k] for k in ('changed_before', 'changed_after', 'stable_true', 'stable_false')},
                candidate_mean_logprobs=dict(changed_before=ca, changed_after=cb, stable_true=sa, stable_false=sb),
                changed_d_original=ca[0]-cb[0], changed_d_edited=ca[1]-cb[1],
                stable_d_original=sa[0]-sb[0], stable_d_edited=sa[1]-sb[1],
                caption_original=cap[0], caption_edited=cap[1], shared_prompt=True,
                prompt_sha256=tensor_digest(prompt), mask_sha256=tensor_digest(mask),
                visual_sha256=[tensor_digest(z) for z in features[r['sample']]])
            records.append(item)
            save(out / 'records_partial.json', records)
            print('Pair', len(records), '/15', r['sample'], 'captions:', repr(cap), flush=True)
            if time.monotonic()-gpu_started > 1800:
                raise TimeoutError('Frozen diagnostic inference cap exceeded')
    if {r['sample'] for r in records} != {r['sample'] for r in rows} or len(records) != 15:
        raise ValueError('Incomplete diagnostic')
    if model_digest(model) != initial_digest or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Frozen C0 changed')
    save(out / 'records.json', records)
    result = dict(summary(records), status='DIAGNOSTIC_COMPLETE', training=False,
        model_digest=initial_digest, checkpoint_sha256=C0_SHA, runtime=runtime_manifest(),
        seconds=time.monotonic()-started, prompt_condition=manifest['prompt_condition'],
        features='pinned CLIP FP32 -> FP16 cache rounding -> FP32; fresh VG JPEG, not historical cache parity',
        clip_revision=CLIP_REV, source_commit=cfg['source_commit'])
    save(out / 'summary.json', result)
    body = ['<!doctype html><meta charset="utf-8"><title>C0 response15</title><style>body{font:17px system-ui;max-width:1100px;margin:auto}article{border-top:1px solid #aaa;padding:20px}.pair{display:flex}.pair div{width:50%}img{width:95%;height:320px;object-fit:contain}p{line-height:1.5}</style><h1>C0:15 cặp ảnh, prompt gốc giữ cố định</h1><p>Caption semantic correctness chưa được tự chấm. Đây là diagnostic development, không phải kết quả CIDEr/CHAIR hoặc hiệu quả phương pháp.</p>']
    for r in records:
        body.append('<article><h2>'+html.escape(r['sample'])+'</h2><div class="pair">')
        for side in ('original', 'edited'):
            body.append('<div><img src="images/'+r['sample']+'_'+side+'.jpg"><p><b>'+side+':</b> '+html.escape(r['caption_'+side])+'</p></div>')
        body.append('</div><p>'+html.escape(str(r['expected']))+'</p><p>Changed contrast: '+str(round(r['changed_d_original'],4))+' → '+str(round(r['changed_d_edited'],4))+'; Stable contrast: '+str(round(r['stable_d_original'],4))+' → '+str(round(r['stable_d_edited'],4))+'</p></article>')
    (out / 'index.html').write_text(''.join(body), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    cfg = json.loads(Path(p.parse_args().config).read_bytes())
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'config.json', cfg)
    try:
        run(cfg, out)
        save(out / 'execution_status.json', dict(success=True, training=False, pairs=15))
    except BaseException as e:
        save(out / 'execution_status.json', dict(success=False, status='INCOMPLETE', error=repr(e), training=False))
        raise
    finally:
        with zipfile.ZipFile(out.with_name(out.name+'_review.zip'), 'x', zipfile.ZIP_DEFLATED) as z:
            for f in out.rglob('*'):
                if f.is_file():
                    z.write(f, str(f.relative_to(out)))
        print('SEND REVIEW:', out.with_name(out.name+'_review.zip'), flush=True)


if __name__ == '__main__':
    main()
