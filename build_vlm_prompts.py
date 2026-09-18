"""Generate resumable image-only scene shards; embed both ablations separately."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import copy
import inspect

from captioning.config import Config, MAX_PROMPT_LEN
from captioning.semantic_prompts import SCENE_INSTRUCTION, parse_scene, scene_prompt, generate_valid_scene


def records_from_json(path):
    images = json.loads(Path(path).read_text(encoding='utf-8'))['images']
    # Deliberately discard all caption/annotation fields before model input.
    records = [{k: image[k] for k in ('filename', 'filepath', 'split')}
               for image in images if image['split'] in {'train', 'restval', 'val', 'test'}]
    names = [r['filename'] for r in records]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate image filenames in split JSON.')
    return records


def load_scene_rows(paths):
    rows = {}
    for path in paths:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row['scene'] = parse_scene(json.dumps(row['scene']))
            name = row['filename']
            if name in rows:
                raise ValueError(f'Duplicate scene: {name}')
            rows[name] = row
    return rows


def generate(args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a GPU for VLM extraction.')
    records = records_from_json(args.dataset_json_path)
    if not 1 <= args.part <= args.parts:
        raise ValueError('part must be between 1 and parts.')
    records = records[len(records)*(args.part-1)//args.parts:len(records)*args.part//args.parts]
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError('Empty scene shard; reduce parts or check split JSON.')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        'model': args.model, 'revision': args.revision, 'part': args.part, 'parts': args.parts,
        'limit': args.limit, 'max_pixels': args.max_pixels, 'max_new_tokens': args.max_new_tokens,
        'instruction': SCENE_INSTRUCTION,
        'dataset_sha256': hashlib.sha256(Path(args.dataset_json_path).read_bytes()).hexdigest(),
        'filenames_sha256': hashlib.sha256(json.dumps([r['filename'] for r in records]).encode()).hexdigest(),
        'generation': 'greedy', 'source': 'images_only',
    }
    meta_path = output.with_suffix(output.suffix + '.meta.json')
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        if any(existing.get(k) != v for k, v in metadata.items()):
            raise ValueError('Resume configuration differs; use another output file.')
    elif output.exists():
        raise ValueError('Scene file has no metadata; cannot safely resume.')
    done = load_scene_rows([output]) if output.exists() else {}
    expected = {r['filename']: r for r in records}
    if set(done) - set(expected):
        raise ValueError('Resume contains images outside this shard.')
    if set(done) == set(expected):
        existing['complete'] = True
        meta_path.write_text(json.dumps(existing, indent=2), encoding='utf-8')
        print(f'Already complete: {len(done)} scenes.', flush=True)
        return
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision,
                                              min_pixels=256*28*28, max_pixels=args.max_pixels,
                                              use_fast=True)
    # dtype was added to newer Transformers; older supported versions use torch_dtype.
    dtype_key = ('dtype' if 'dtype' in inspect.signature(
        Qwen2_5_VLForConditionalGeneration.from_pretrained).parameters else 'torch_dtype')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, **{dtype_key: torch.float16},
        attn_implementation='sdpa').to('cuda').eval()
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    resolved = getattr(model.config, '_commit_hash', None)
    if meta_path.exists() and existing.get('resolved_revision') != resolved:
        raise ValueError('Model revision changed; resume using the original revision.')
    metadata.update(resolved_revision=resolved, complete=False,
                    repair_policy={'schema_feedback_retries': args.retries}, processor_use_fast=True)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    started, count = time.monotonic(), 0
    for record in records:
        if record['filename'] in done:
            continue
        path = Path(args.base_path) / record['filepath'] / record['filename']
        with Image.open(path) as image:
            # Processor-supported PIL image: no captions, IDs or filenames in the message.
            messages = [{'role': 'user', 'content': [
                {'type': 'image', 'image': image.convert('RGB')},
                {'type': 'text', 'text': SCENE_INSTRUCTION}]}]
            def generate_raw(previous_raw, previous_error):
                conversation = list(messages)
                if previous_raw is not None:
                    conversation += [
                        {'role': 'assistant', 'content': [{'type': 'text', 'text': previous_raw}]},
                        {'role': 'user', 'content': [{'type': 'text', 'text':
                            'Your JSON failed validation: ' + previous_error +
                            '. Reinspect the image and return the complete corrected JSON only. '
                            'Use exact object names for endpoints and only the allowed spatial predicates. '
                            'Do not invent missing objects or relations. If a relation cannot be supported, '
                            'omit it; relations may be an empty list.'}]}]
                inputs = processor.apply_chat_template(conversation, tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    generated = model.generate(**inputs, generation_config=generation_config,
                                               max_new_tokens=args.max_new_tokens)
                return processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:],
                                              skip_special_tokens=True)[0]

            def log_error(attempt, raw, error):
                with output.with_suffix('.errors.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'filename': record['filename'], 'attempt': attempt + 1,
                                        'raw': raw, 'error': error}) + '\n')
                print(f'{record["filename"]}: schema attempt {attempt + 1} failed: {error}', flush=True)

            try:
                scene, raw, retry_count = generate_valid_scene(generate_raw, args.retries, log_error)
            except ValueError as error:
                raise RuntimeError(f'Invalid VLM scene for {record["filename"]}; see errors file. '
                                   'Valid earlier rows are saved; no empty fallback prompt.') from error
        with output.open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(record, scene=scene, raw=raw, retries=retry_count), ensure_ascii=False) + '\n')
            f.flush()
        count += 1
        done[record['filename']] = record
        if count == 1 or count % 25 == 0:
            seconds = (time.monotonic()-started)/count
            print(f'Scenes {len(done)}/{len(records)} | {seconds:.2f}s/image | '
                  f'ETA {(len(records)-len(done))*seconds/60:.1f}min', flush=True)
    metadata['complete'] = True
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Completed: {output}', flush=True)


def embed(args):
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer
    records = records_from_json(args.dataset_json_path)
    scenes = load_scene_rows(args.scenes)
    scene_metadata = []
    for path in args.scenes:
        meta_path = Path(str(path) + '.meta.json')
        if not meta_path.is_file():
            raise ValueError(f'Missing extraction metadata: {meta_path}')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if not meta.get('complete') or meta.get('source') != 'images_only':
            raise ValueError(f'Unfinished or unsupported scene shard: {path}')
        if meta.get('limit') and not args.allow_partial:
            raise ValueError('Smoke shards cannot build full experiment caches.')
        if meta.get('dataset_sha256') != hashlib.sha256(Path(args.dataset_json_path).read_bytes()).hexdigest():
            raise ValueError(f'Scene split JSON hash mismatch: {path}')
        scene_metadata.append(meta)
    profile_keys = ('model', 'resolved_revision', 'max_pixels', 'max_new_tokens',
                    'instruction', 'generation', 'parts', 'repair_policy', 'processor_use_fast')
    if any(any(meta.get(k) != scene_metadata[0].get(k) for k in profile_keys)
           for meta in scene_metadata[1:]):
        raise ValueError('Scene shards use different VLM extraction profiles.')
    expected = {r['filename']: r for r in records}
    if set(scenes) - set(expected):
        raise ValueError('Scene filenames do not belong to the split JSON.')
    for name, row in scenes.items():
        if any(row[k] != expected[name][k] for k in ('filepath', 'split')):
            raise ValueError(f'Scene split/path mismatch: {name}')
    if set(scenes) != set(expected) and not args.allow_partial:
        raise ValueError(f'Missing {len(set(expected)-set(scenes))} scenes; full train/val/test required.')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destinations = {v: output/f'prompt_vlm_{v}.pt' for v in ('objects', 'spatial')}
    if any(p.exists() for p in destinations.values()):
        raise FileExistsError('Prompt cache already exists; use another output directory.')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
    model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch16').float().to(device).eval()
    ordered = [r['filename'] for r in records if r['filename'] in scenes]
    for variant, destination in destinations.items():
        data, truncated, relations_lost = {}, 0, 0
        for start in range(0, len(ordered), args.batch_size):
            names = ordered[start:start+args.batch_size]
            texts = [scene_prompt(scenes[n]['scene'], variant) for n in names]
            lengths = [len(ids) for ids in tokenizer(texts, truncation=False)['input_ids']]
            truncated += sum(n > MAX_PROMPT_LEN for n in lengths)
            if variant == 'spatial':
                for name in names:
                    relations = scenes[name]['scene']['relations']
                    text = '; '.join('{subject} {predicate} {object}'.format(**r) for r in relations)
                    relations_lost += bool(relations) and len(tokenizer(text)['input_ids']) > MAX_PROMPT_LEN
            inputs = tokenizer(texts, padding='max_length', truncation=True,
                               max_length=MAX_PROMPT_LEN, return_tensors='pt').to(device)
            with torch.inference_mode():
                tokens = model(**inputs).last_hidden_state.cpu().half()
            if tokens.shape[1:] != (MAX_PROMPT_LEN, 512) or not torch.isfinite(tokens).all():
                raise ValueError('Invalid CLIP prompt token shape or nonfinite values.')
            masks = inputs['attention_mask'].cpu().to(torch.uint8)
            for index, name in enumerate(names):
                data[name] = {'tokens': tokens[index].clone(), 'mask': masks[index].clone(),
                              'prompt': texts[index]}
            if start == 0 or start % (args.batch_size*100) == 0:
                print(f'Embed {variant}: {len(data)}/{len(ordered)}', flush=True)
        metadata = {'variant': variant, 'encoder': 'openai/clip-vit-base-patch16',
                    'feature': 'text_model.last_hidden_state', 'max_length': MAX_PROMPT_LEN,
                    'storage': 'fp16', 'extraction': 'fp32', 'key': 'filename',
                    'count': len(data), 'complete': len(data) == len(expected),
                    'vlm_profile': {k: scene_metadata[0].get(k) for k in profile_keys},
                    'truncated_prompts': truncated, 'relation_prefix_truncated': relations_lost,
                    'scene_sha256': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.scenes},
                    'dataset_sha256': hashlib.sha256(Path(args.dataset_json_path).read_bytes()).hexdigest()}
        temporary = destination.with_suffix('.pt.partial')
        torch.save({'data': data, 'metadata': metadata}, temporary)
        temporary.replace(destination)
        destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(f'Saved {destination}; truncated={truncated}/{len(data)}, '
              f'relation-prefix-truncated={relations_lost}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='mode', required=True)
    defaults = Config()
    for mode in ('generate', 'embed'):
        sub = subs.add_parser(mode)
        sub.add_argument('--dataset-json-path', default=defaults.dataset_json_path)
        if mode == 'generate':
            sub.add_argument('--base-path', default=defaults.base_path)
            sub.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct')
            sub.add_argument('--revision', default='main')
            sub.add_argument('--part', type=int, default=1)
            sub.add_argument('--parts', type=int, default=1)
            sub.add_argument('--limit', type=int, default=100, help='Pilot default; 0 = entire shard.')
            sub.add_argument('--max-pixels', type=int, default=512*28*28)
            sub.add_argument('--max-new-tokens', type=int, default=384)
            sub.add_argument('--retries', type=int, default=2, help='Additional schema-feedback attempts per image.')
            sub.add_argument('--output', required=True)
        else:
            sub.add_argument('--scenes', nargs='+', required=True)
            sub.add_argument('--output-dir', required=True)
            sub.add_argument('--batch-size', type=int, default=128)
            sub.add_argument('--allow-partial', action='store_true', help='Smoke cache only; cannot train full split.')
    args = parser.parse_args(argv)
    if args.mode == 'generate':
        if args.parts < 1 or args.limit < 0 or args.max_pixels < 256*28*28 or args.max_new_tokens < 1 or args.retries < 0:
            parser.error('Invalid partition, limit or generation budget.')
        generate(args)
    else:
        if args.batch_size < 1:
            parser.error('batch-size must be positive.')
        embed(args)


if __name__ == '__main__':
    main()
