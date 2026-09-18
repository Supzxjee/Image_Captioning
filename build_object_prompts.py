"""Build a CLIP objects-only cache from the ORIGINAL saved YOLO detections."""
import argparse
import hashlib
import json
from pathlib import Path

from captioning.config import Config, MAX_PROMPT_LEN
from captioning.object_prompts import build_object_texts
from build_vlm_prompts import records_from_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='JSON image->objects map, or .pt bundle containing raw objects.')
    parser.add_argument('--objects-field', default='objects')
    parser.add_argument('--name-key', default='', help='Explicit class-name key inside detection dictionaries, if needed.')
    parser.add_argument('--dataset-json-path', default=Config().dataset_json_path)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--limit', type=int, default=0, help='Smoke only; 0 = all train/val/test images.')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.limit < 0:
        parser.error('batch-size must be positive; limit nonnegative.')
    import torch
    source_path = Path(args.source)
    if source_path.suffix == '.json':
        source = json.loads(source_path.read_text(encoding='utf-8'))
    elif source_path.suffix == '.pt':
        source = torch.load(source_path, map_location='cpu', weights_only=False)
    else:
        parser.error('source must be .json or .pt')
    if isinstance(source, dict) and 'data' in source and isinstance(source['data'], dict):
        source = source['data']
    records = records_from_json(args.dataset_json_path)
    total = len(records)
    if args.limit:
        records = records[:args.limit]
    texts = build_object_texts(source, records, args.objects_field, args.name_key)
    del source
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    from transformers import CLIPTextModel, CLIPTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
    model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch16').float().to(device).eval()
    data, truncated = {}, 0
    names = list(texts)
    for start in range(0, len(names), args.batch_size):
        batch_names = names[start:start+args.batch_size]
        prompts = [texts[n] for n in batch_names]
        truncated += sum(len(ids) > MAX_PROMPT_LEN for ids in tokenizer(prompts, truncation=False)['input_ids'])
        inputs = tokenizer(prompts, padding='max_length', truncation=True,
                           max_length=MAX_PROMPT_LEN, return_tensors='pt').to(device)
        with torch.inference_mode():
            tokens = model(**inputs).last_hidden_state.cpu().half()
        if tokens.shape[1:] != (MAX_PROMPT_LEN, 512) or not torch.isfinite(tokens).all():
            raise ValueError('Invalid CLIP prompt tokens.')
        masks = inputs['attention_mask'].cpu().to(torch.uint8)
        for i, name in enumerate(batch_names):
            data[name] = {'tokens': tokens[i].clone(), 'mask': masks[i].clone(),
                          'prompt': texts[name]}
        if start == 0 or start % (100*args.batch_size) == 0:
            print(f'Objects cache: {len(data)}/{len(names)}', flush=True)
    metadata = {'variant': 'yolo_objects_only', 'key': 'filename',
                'encoder': 'openai/clip-vit-base-patch16', 'feature': 'text_model.last_hidden_state',
                'extraction': 'fp32', 'storage': 'fp16', 'max_length': MAX_PROMPT_LEN,
                'objects_field': args.objects_field, 'name_key': args.name_key,
                'ordering': 'source order, unique lowercase class names',
                'empty_detections': sum(not text for text in texts.values()),
                'count': len(data), 'complete': len(data) == total, 'truncated_prompts': truncated,
                'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                'dataset_sha256': hashlib.sha256(Path(args.dataset_json_path).read_bytes()).hexdigest()}
    temporary = destination.with_suffix('.pt.partial')
    torch.save({'data': data, 'metadata': metadata}, temporary)
    temporary.replace(destination)
    destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for name in names[:10]:
        print(name, repr(texts[name]), flush=True)
    print(f'Saved {destination}; truncated={truncated}, empty={metadata["empty_detections"]}', flush=True)


if __name__ == '__main__':
    main()
