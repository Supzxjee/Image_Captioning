"""Encode relation-prompt JSON as the CLIP token cache used by caption training."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from build_vlm_prompts import records_from_json
from captioning.config import Config, MAX_PROMPT_LEN


def index_prompts(source):
    if not isinstance(source, dict):
        raise ValueError('Relation prompt source must be a mapping.')
    indexed = {}
    for key, entry in source.items():
        filename = PurePosixPath(str(key).replace('\\', '/')).name
        if filename in indexed:
            raise ValueError(f'Ambiguous relation prompt filename: {filename}')
        if not isinstance(entry, dict) or not isinstance(entry.get('prompt'), str):
            raise ValueError(f'{filename}: missing string prompt.')
        indexed[filename] = entry
    return indexed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset-json-path', default=Config().dataset_json_path)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error('--batch-size must be positive.')

    import torch
    from transformers import CLIPTextModel, CLIPTokenizer

    source_path = Path(args.source)
    source = json.loads(source_path.read_text(encoding='utf-8'))
    indexed = index_prompts(source)
    records = records_from_json(args.dataset_json_path)
    expected = [record['filename'] for record in records]
    extras = set(indexed) - set(expected)
    missing = set(expected) - set(indexed)
    if extras:
        raise ValueError(f'{len(extras)} prompt filenames are outside dataset JSON.')
    if missing and not args.allow_partial:
        raise ValueError(f'Missing {len(missing)} prompts; full cache required for training.')
    ordered = [name for name in expected if name in indexed]

    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
    model = CLIPTextModel.from_pretrained(
        'openai/clip-vit-base-patch16').float().to(device).eval()
    data, truncated = {}, 0
    for start in range(0, len(ordered), args.batch_size):
        names = ordered[start:start + args.batch_size]
        texts = [indexed[name]['prompt'] for name in names]
        lengths = [len(ids) for ids in tokenizer(texts, truncation=False)['input_ids']]
        truncated += sum(length > MAX_PROMPT_LEN for length in lengths)
        inputs = tokenizer(
            texts, padding='max_length', truncation=True,
            max_length=MAX_PROMPT_LEN, return_tensors='pt').to(device)
        with torch.inference_mode():
            tokens = model(**inputs).last_hidden_state.cpu().half()
        if tokens.shape[1:] != (MAX_PROMPT_LEN, 512) or not torch.isfinite(tokens).all():
            raise ValueError(f'Invalid CLIP prompt tokens: {tokens.shape}')
        masks = inputs['attention_mask'].cpu().to(torch.uint8)
        for index, name in enumerate(names):
            data[name] = {
                'tokens': tokens[index].clone(),
                'mask': masks[index].clone(),
                'prompt': texts[index],
            }
        if start == 0 or start % (100 * args.batch_size) == 0:
            print(f'Encoded prompts: {len(data)}/{len(ordered)}', flush=True)

    metadata = {
        'variant': 'semantic_instance_aware_relation_v2_1',
        'heuristic_version': 'H1_3_semantic_instance_aware_v2_1',
        'encoder': 'openai/clip-vit-base-patch16',
        'feature': 'text_model.last_hidden_state',
        'storage': 'fp16',
        'extraction': 'fp32',
        'max_length': MAX_PROMPT_LEN,
        'key': 'filename',
        'count': len(data),
        'complete': not missing and len(data) == len(expected),
        'truncated_prompts': truncated,
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
    print('Saved:', destination, flush=True)


if __name__ == '__main__':
    main()
