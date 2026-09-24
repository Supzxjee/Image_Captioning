"""Cache five frozen CLIP text embeddings per Karpathy image in HDF5."""
import argparse
import hashlib
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-json-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Images per batch; each image contributes five captions.')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--limit', type=int, default=0, help='Smoke only; 0 caches all images.')
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.limit < 0:
        parser.error('batch-size must be positive and limit nonnegative.')

    import h5py
    import numpy as np
    import torch
    from tqdm.auto import tqdm
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    from captioning.visual_cache import coco_image_id

    dataset_path = Path(args.dataset_json_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    payload = json.loads(dataset_path.read_text(encoding='utf-8'))
    records = payload.get('images', [])
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError('No images to cache.')
    for record in records:
        if len(record.get('sentences', [])) < 5:
            raise ValueError(f'Expected five captions for {record.get("filename")}')

    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    if temporary.exists():
        raise FileExistsError(temporary)

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable.')
    model_name = 'openai/clip-vit-base-patch16'
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModelWithProjection.from_pretrained(model_name).float().to(device).eval()

    with h5py.File(temporary, 'w') as handle:
        features = handle.create_dataset(
            'features', shape=(len(records), 5, 512), dtype='float16',
            chunks=(1, 5, 512))
        imgids = handle.create_dataset('imgids', shape=(len(records),), dtype='int64')
        handle.attrs['complete'] = False
        handle.attrs['model'] = model_name
        handle.attrs['feature'] = 'CLIPTextModelWithProjection.text_embeds'
        handle.attrs['captions_per_image'] = 5
        handle.attrs['dataset_sha256'] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

        for start in tqdm(range(0, len(records), args.batch_size), desc='Caching caption embeddings'):
            batch = records[start:start + args.batch_size]
            texts = [sentence['raw'] for record in batch for sentence in record['sentences'][:5]]
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=77,
                                return_tensors='pt').to(device)
            with torch.inference_mode():
                embeddings = model(**encoded).text_embeds
            embeddings = embeddings.reshape(len(batch), 5, 512).float().cpu().numpy()
            if not np.isfinite(embeddings).all():
                raise ValueError(f'Non-finite caption embedding at rows {start}:{start + len(batch)}')
            features[start:start + len(batch)] = embeddings.astype(np.float16)
            imgids[start:start + len(batch)] = [coco_image_id(record['filename']) for record in batch]
            if start == 0 or (start // args.batch_size + 1) % 100 == 0:
                handle.flush()
                print(f'Caption embeddings: {start + len(batch)}/{len(records)}', flush=True)

        handle.attrs['complete'] = True
        handle.flush()

    temporary.replace(destination)
    metadata = {
        'complete': True,
        'count': len(records),
        'shape': [len(records), 5, 512],
        'dtype': 'float16',
        'model': model_name,
        'feature': 'CLIPTextModelWithProjection.text_embeds',
        'dataset_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        'smoke': bool(args.limit),
    }
    destination.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Completed caption cache: {destination}', flush=True)


if __name__ == '__main__':
    main()
