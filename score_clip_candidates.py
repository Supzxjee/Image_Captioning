"""Attach standard CLIPScore values to saved beam candidates using visual cache."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--visual-cache', action='append', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='openai/clip-vit-base-patch16')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error('batch-size must be positive.')

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, CLIPModel
    from captioning.visual_cache import VisualCache

    source_path = Path(args.candidates)
    bundle = json.loads(source_path.read_text(encoding='utf-8'))
    records = bundle.get('data', [])
    if not records or bundle.get('metadata', {}).get('count') != len(records):
        raise ValueError('Candidate bundle is empty or incomplete.')
    cache = VisualCache(args.visual_cache, id_key='coco_id')
    cache.require_ids([record['coco_id'] for record in records], 'CLIPScore')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Loading {args.model} on {device}.', flush=True)
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for parameter in model.parameters():
        parameter.requires_grad = False

    try:
        with torch.no_grad():
            for start in range(0, len(records), args.batch_size):
                batch = records[start:start + args.batch_size]
                raw = np.stack([cache.read(record['coco_id']) for record in batch])
                cls_tokens = torch.from_numpy(raw[:, 0]).to(device=device, dtype=torch.float32)
                # Cache stores CLIPVisionModel.last_hidden_state. Reapply the
                # official CLS post-layernorm and visual projection used by CLIP.
                pooled = model.vision_model.post_layernorm(cls_tokens)
                image_features = F.normalize(model.visual_projection(pooled), dim=-1)

                texts, owners = [], []
                for image_index, record in enumerate(batch):
                    for candidate in record.get('candidates', []):
                        texts.append(candidate['caption'])
                        owners.append(image_index)
                encoded = tokenizer(texts, padding=True, truncation=True,
                                    max_length=77, return_tensors='pt')
                encoded = {key: value.to(device) for key, value in encoded.items()}
                text_output = model.text_model(**encoded)
                text_features = F.normalize(
                    model.text_projection(text_output.pooler_output), dim=-1)
                owner_tensor = torch.tensor(owners, device=device, dtype=torch.long)
                cosine = (image_features[owner_tensor] * text_features).sum(dim=-1)
                scores = (2.5 * cosine.clamp_min(0)).cpu().tolist()
                cursor = 0
                for record in batch:
                    for candidate in record['candidates']:
                        candidate['clipscore'] = float(scores[cursor])
                        cursor += 1
                completed = min(start + len(batch), len(records))
                print(f'CLIPScore {completed}/{len(records)}', flush=True)
    finally:
        cache.close()

    metadata = bundle.setdefault('metadata', {})
    metadata.update({
        'clipscore_model': args.model,
        'clipscore_formula': '2.5 * max(cosine(CLIP image, CLIP text), 0)',
        'clipscore_visual_source': 'cached last_hidden_state CLS + post_layernorm + visual_projection',
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    main()
