"""Build fixed-size per-object CLIP text embeddings from saved detector output."""
import argparse
import hashlib
import json
from pathlib import Path

from build_vlm_prompts import records_from_json
from captioning.config import Config
from captioning.object_concepts import build_object_concept_index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Saved .json or .pt detector output.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset-json-path', default=Config().dataset_json_path)
    parser.add_argument('--objects-field', default='objects')
    parser.add_argument('--name-key', default='label')
    parser.add_argument('--confidence-key', default='conf')
    parser.add_argument('--min-confidence', type=float, default=0.5)
    parser.add_argument('--max-objects', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--limit', type=int, default=0, help='Smoke only; 0 builds all splits.')
    args = parser.parse_args(argv)
    if not 0 <= args.min_confidence <= 1:
        parser.error('min-confidence must be in [0, 1].')
    if args.max_objects < 1 or args.batch_size < 1 or args.limit < 0:
        parser.error('max-objects/batch-size must be positive; limit nonnegative.')

    import torch
    source_path = Path(args.source)
    if source_path.suffix == '.json':
        source = json.loads(source_path.read_text(encoding='utf-8'))
    elif source_path.suffix == '.pt':
        source = torch.load(source_path, map_location='cpu', weights_only=False)
    else:
        parser.error('source must be .json or .pt')
    if isinstance(source, dict) and isinstance(source.get('data'), dict):
        source = source['data']
    records = records_from_json(args.dataset_json_path)
    total = len(records)
    if args.limit:
        records = records[:args.limit]
    selected = build_object_concept_index(
        source, records, args.objects_field, args.name_key, args.confidence_key,
        args.min_confidence, args.max_objects)
    del source

    labels = sorted({label for items in selected.values() for label, _ in items})
    from transformers import CLIPTextModel, CLIPTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
    model = CLIPTextModel.from_pretrained(
        'openai/clip-vit-base-patch16').float().to(device).eval()
    embeddings = {}
    for start in range(0, len(labels), args.batch_size):
        batch_labels = labels[start:start + args.batch_size]
        prompts = [f'a photo of a {label}' for label in batch_labels]
        inputs = tokenizer(prompts, padding=True, truncation=True,
                           return_tensors='pt').to(device)
        with torch.inference_mode():
            pooled = model(**inputs).pooler_output.cpu().half()
        if pooled.shape != (len(batch_labels), 512) or not torch.isfinite(pooled).all():
            raise ValueError(f'Invalid pooled CLIP embeddings: {pooled.shape}')
        embeddings.update(zip(batch_labels, pooled))
        print(f'Encoded labels: {min(start + args.batch_size, len(labels))}/{len(labels)}',
              flush=True)

    data = {}
    empty = 0
    for filename, items in selected.items():
        tokens = torch.zeros(args.max_objects, 512, dtype=torch.float16)
        mask = torch.zeros(args.max_objects, dtype=torch.uint8)
        confidences = torch.zeros(args.max_objects, dtype=torch.float16)
        names = []
        for index, (label, confidence) in enumerate(items):
            tokens[index] = embeddings[label]
            mask[index] = 1
            confidences[index] = confidence
            names.append(label)
        empty += int(not names)
        data[filename] = {
            'tokens': tokens,
            'mask': mask,
            'labels': names,
            'confidences': confidences,
        }

    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        'variant': 'per_object_clip_prompts',
        'key': 'filename',
        'prompt_template': 'a photo of a {label}',
        'encoder': 'openai/clip-vit-base-patch16',
        'feature': 'text_model.pooler_output',
        'storage': 'fp16',
        'min_confidence': args.min_confidence,
        'max_objects': args.max_objects,
        'unique_labels': len(labels),
        'empty_images': empty,
        'count': len(data),
        'complete': len(data) == total,
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'dataset_sha256': hashlib.sha256(
            Path(args.dataset_json_path).read_bytes()).hexdigest(),
    }
    temporary = destination.with_suffix(destination.suffix + '.partial')
    torch.save({'data': data, 'metadata': metadata}, temporary)
    temporary.replace(destination)
    destination.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2), flush=True)
    print(f'Saved: {destination}', flush=True)


if __name__ == '__main__':
    main()
