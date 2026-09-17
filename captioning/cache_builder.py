"""Build baseline-compatible CLIP caches, one bounded Kaggle output per run."""
import argparse
import hashlib
import json
from pathlib import Path
from .visual_cache import coco_image_id

MODEL_ID = 'openai/clip-vit-base-patch16'
BYTES_PER_IMAGE = 197 * 768 * 2


def select_part(records, part, parts):
    if parts < 1 or not 1 <= part <= parts:
        raise ValueError('Require 1 <= part <= parts.')
    start = len(records) * (part - 1) // parts
    stop = len(records) * part // parts
    return records[start:stop]


def collect_records(path):
    with open(path, encoding='utf-8') as f:
        bundle = json.load(f)
    records, ids = [], set()
    for eval_id, image in enumerate(bundle['images']):
        if image['split'] not in {'train', 'restval', 'val', 'test'}:
            continue
        image_id = coco_image_id(image['filename'])
        if image_id in ids:
            raise ValueError(f'Duplicate COCO ID: {image_id}')
        ids.add(image_id)
        records.append(dict(coco_id=image_id, eval_id=eval_id,
                            filename=image['filename'], filepath=image['filepath'],
                            split=image['split']))
    return records


def write_feature_batches(path, records, batches, metadata):
    """Write aligned arrays with an explicit completion marker; no huge RAM tensor."""
    import h5py
    import numpy as np
    path = Path(path)
    if not records:
        raise ValueError('Cannot cache an empty part.')
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    with h5py.File(path, 'x') as f:
        f.attrs['complete'] = False
        f.attrs['metadata_json'] = json.dumps(metadata)
        f.attrs['feature_layout'] = 'features[i] belongs to imgids[i]'
        f.attrs['id_key'] = 'coco_id'
        f['imgids'] = np.array([r['coco_id'] for r in records], dtype=np.int64)
        f['eval_ids'] = np.array([r['eval_id'] for r in records], dtype=np.int64)
        text = h5py.string_dtype('utf-8')
        for key in ('filename', 'filepath', 'split'):
            f.create_dataset(key, data=[r[key] for r in records], dtype=text)
        store = f.create_dataset('features', shape=(len(records), 197, 768),
                                 dtype='float16', chunks=(1, 197, 768))
        offset = 0
        for array in batches:
            array = np.asarray(array)
            if array.ndim != 3 or array.shape[1:] != (197, 768) or len(array) == 0:
                raise ValueError(f'Invalid batch shape: {array.shape}')
            if offset + len(array) > len(records):
                raise ValueError('More feature rows than image IDs.')
            if not np.isfinite(array).all():
                raise ValueError('Non-finite CLIP output.')
            with np.errstate(over='ignore'):
                saved = array.astype(np.float16)
            if not np.isfinite(saved).all():
                raise ValueError('Features overflow when stored as FP16.')
            store[offset:offset+len(saved)] = saved
            offset += len(saved)
            f.attrs['rows_written'] = offset
            f.flush()
        if offset != len(records):
            raise ValueError(f'Incomplete cache: {offset}/{len(records)} rows.')
        f.attrs['complete'] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-json-path', default='/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
    parser.add_argument('--base-path', default='/kaggle/input/datasets/vuthetam/mscoco-2014/images')
    parser.add_argument('--output-dir', default='/kaggle/working/visual_cache')
    parser.add_argument('--part', type=int, default=1)
    parser.add_argument('--parts', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--loader-timeout', type=float, default=120,
                        help='Seconds to wait for a worker batch; ignored with num-workers=0.')
    parser.add_argument('--limit', type=int, default=0, help='Smoke test: first N images in the selected part, 0=all.')
    parser.add_argument('--verify-samples', type=int, default=10)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.limit < 0 or args.verify_samples < 1 or args.loader_timeout <= 0:
        parser.error('Invalid batch size, workers, limit or verification sample count.')
    records = select_part(collect_records(args.dataset_json_path), args.part, args.parts)
    if args.limit:
        records = records[:args.limit]
    estimated = len(records) * BYTES_PER_IMAGE
    if not records:
        raise ValueError('Selected part is empty.')
    if estimated > 18_000_000_000:
        raise ValueError('Part exceeds 18 GB tensor budget; increase --parts.')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Check existing output before creating another 9GB artifact in the same version.
    existing = sum(p.stat().st_size for p in output_dir.parent.rglob('*') if p.is_file())
    if existing + estimated + 200_000_000 > 19_000_000_000:
        raise ValueError('Existing outputs plus this cache approach 20 GB; use a fresh Kaggle version/session.')
    suffix = '_smoke' if args.limit else ''
    final = output_dir / f'visual_part_{args.part:02d}_of_{args.parts:02d}{suffix}.h5'
    pending = final.with_suffix('.partial')
    if final.exists() or pending.exists():
        raise FileExistsError(f'Output already exists: {final} or {pending}. Use a fresh output directory.')
    print(f'Part {args.part}/{args.parts}: {len(records)} images, ~{estimated/1e9:.2f} GB tensors', flush=True)
    import torch
    import torchvision
    import transformers
    import h5py
    import numpy as np
    from PIL import Image, ImageFile
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import CLIPModel
    from .data import build_image_transform
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Explicit FP32, no autocast/TF32 for the frozen visual encoder.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backbone = CLIPModel.from_pretrained(MODEL_ID).vision_model.float().to(device).eval()
    backbone.requires_grad_(False)
    transform = build_image_transform('bilinear')
    from .cache_images import CacheImages
    dataset = CacheImages(records, args.base_path, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device == 'cuda',
                        timeout=args.loader_timeout if args.num_workers else 0)
    metadata = dict(model_id=MODEL_ID, model_revision=getattr(backbone.config, '_commit_hash', None),
                    output='vision_model.last_hidden_state', preprocessing='bilinear',
                    image_size=[224, 224], antialias=True,
                    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711],
                    compute_dtype='float32', storage_dtype='float16', autocast=False,
                    dataset_json_sha256=hashlib.sha256(Path(args.dataset_json_path).read_bytes()).hexdigest(),
                    part=args.part, parts=args.parts, smoke=bool(args.limit), image_count=len(records),
                    torch=torch.__version__, torchvision=torchvision.__version__, transformers=transformers.__version__)
    # Preflight extraction and quantization check before the expensive full part.
    sample_count = min(args.verify_samples, len(records))
    sample_pixels = torch.stack([dataset[i] for i in range(sample_count)]).to(device)
    with torch.no_grad():
        sample_features = backbone(pixel_values=sample_pixels).last_hidden_state.float()
    if not torch.isfinite(sample_features).all():
        raise ValueError('Non-finite features in preflight.')
    if not torch.allclose(sample_features, sample_features.half().float(), atol=0.01, rtol=0.005):
        raise ValueError('FP16 storage failed preflight tolerance; do not proceed.')
    print(f'Preflight PASS: {sample_count} images; FP32 extraction/FP16 storage.', flush=True)

    def batches():
        iterator = iter(loader)
        for i in tqdm(range(len(loader)), desc='Caching CLIP FP32'):
            if i % 50 == 0:
                print(f'Waiting for image batch {i+1}/{len(loader)} at row {i*args.batch_size}', flush=True)
            try:
                pixels = next(iterator)
            except Exception:
                near = records[i*args.batch_size:(i+1)*args.batch_size]
                print(f'Image loader failed while requesting rows {i*args.batch_size}:{(i+1)*args.batch_size}. Requested filenames: {[r["filename"] for r in near]}. Workers may also be prefetching later rows.', flush=True)
                raise
            if i % 50 == 0:
                print(f'Running CLIP for batch {i+1}', flush=True)
            with torch.no_grad():
                features = backbone(pixel_values=pixels.to(device, non_blocking=True)).last_hidden_state.float()
            array = features.cpu().numpy()
            if (i+1) % 50 == 0:
                print(f'Computed {min((i+1)*args.batch_size, len(records))}/{len(records)}; writing HDF5 next', flush=True)
            yield array

    write_feature_batches(pending, records, batches(), metadata)
    # Re-read real saved rows and compare to fresh FP32 extraction.
    with h5py.File(pending, 'r') as f:
        saved = torch.from_numpy(np.asarray(f['features'][:sample_count], dtype=np.float32)).to(device)
    if not torch.allclose(sample_features, saved, atol=0.01, rtol=0.005):
        raise RuntimeError('Saved cache verification failed; .partial retained, not published as .h5.')
    pending.rename(final)
    manifest = final.with_suffix('.json')
    manifest.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'PASS: saved-row verification. Completed cache: {final}', flush=True)


if __name__ == '__main__':
    main()
