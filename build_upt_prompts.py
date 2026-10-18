"""Offline UPT prompt builder for saved native-score audit or full-cache shards."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

from audit_vcoco_alignment_data import digest
from captioning.upt_cache import identity, read_shard
from captioning.upt_prompts import METHOD, from_audit, from_cache, build_prompt
from captioning.upt_action_text import ACTION_TEXT, uniform_policy
from run_vcoco_upt_full_audit import CHECKPOINT


def review_image(filename, output, image_root=None):
    if Path(filename).name != filename or '/' in filename or '\\' in filename:
        raise ValueError('Expected an image filename, not a path.')
    split = 'train2014' if filename.startswith('COCO_train2014_') else 'val2014'
    if image_root is None:
        return 'https://images.cocodataset.org/' + split + '/' + filename
    source = image_root / filename
    if not source.is_file():
        source = image_root / split / filename
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = output / 'images' / filename
    destination.parent.mkdir(exist_ok=True)
    shutil.copyfile(source, destination)
    return 'images/' + filename


def records(args):
    if args.audit_zip:
        with zipfile.ZipFile(args.audit_zip) as archive:
            if len(archive.namelist()) != len(set(archive.namelist())) or archive.testzip() is not None:
                raise ValueError('Corrupt/duplicate audit ZIP entries.')
            cfg = json.loads(archive.read('protocol.json'))
            summary = json.loads(archive.read('run_summary.json'))
            if cfg.get('score_mode') != 'native' or cfg['checkpoint_sha256'] != CHECKPOINT or summary['status'] != 'FROZEN_RELATION_AUDIT_COMPLETE':
                raise ValueError('Expected completed frozen native-score audit.')
            if hashlib.sha256(archive.read('protocol.json')).hexdigest() != summary['protocol_sha256']:
                raise ValueError('Audit protocol identity differs.')
            for rank in (0, 1):
                rows = [json.loads(line) for line in archive.read(f'upt_{rank}.jsonl').splitlines()]
                if [r['image_id'] for r in rows] != [r['image_id'] for r in cfg['images'][rank::2]]:
                    raise ValueError('Audit shard incomplete/reordered.')
                for row in rows:
                    detections, claims = from_audit(row)
                    yield row['filename'], detections, claims
    else:
        root = args.cache
        meta = json.loads((root / 'cache_identity.json').read_text())
        cfg = json.loads((root / 'session_config.json').read_text())
        data = meta['identity']
        if identity(data) != meta['signature'] or cfg['signature'] != meta['signature']:
            raise ValueError('Cache identity mismatch.')
        if data['schema'] != 'upt_full_coco_supported_logits_v1' or data['checkpoint_sha256'] != CHECKPOINT:
            raise ValueError('Wrong cache/model schema.')
        if identity(cfg['tasks']) != data['image_plan_sha256'] or cfg['vocabulary'] != data['vocabulary']:
            raise ValueError('Cache task plan/vocabulary differs.')
        manifest_path = root / 'shard_manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if manifest is not None and manifest['signature'] != meta['signature']:
            raise ValueError('Cache manifest signature differs.')
        for task in cfg['tasks']:
            path = root / (task['name'] + '.jsonl.gz')
            if not path.exists():
                if args.allow_partial:
                    continue
                raise FileNotFoundError(f'Incomplete cache: {path}. Use --allow-partial only for previews.')
            if manifest is not None and path.name in manifest['shards'] and digest(path) != manifest['shards'][path.name]['sha256']:
                raise ValueError('Cache shard hash differs from manifest.')
            for row in read_shard(path, task, meta['signature']):
                detections, claims = from_cache(row, data['vocabulary'], getattr(args, 'all_actions', False))
                yield row['filename'], detections, claims


def run(args):
    if args.output.exists():
        raise FileExistsError('Use a new output; preserve earlier prompts.')
    if args.all_actions:
        if args.policy is not None or args.audit_zip:
            raise ValueError('--all-actions requires full-cache input, without --policy; audit ZIP has only hold/ride.')
        policy = uniform_policy(args.relation_threshold)
    else:
        if args.policy is None:
            raise ValueError('Legacy hold/ride mode requires --policy; use --all-actions for 116 actions.')
        policy = json.loads(args.policy.read_text())
    prompts = {}
    for filename, detections, claims in records(args):
        if filename in prompts:
            raise ValueError('Duplicate image filename.')
        prompts[filename] = build_prompt(filename, detections, claims, policy, args.max_relations,
                                         args.object_threshold, args.max_object_names)
    if not prompts:
        raise ValueError('No completed image records found.')
    summary = {'variant': 'upt_object_relation_prompt_all_actions_v2' if args.all_actions else METHOD,
               'supported_actions': len(ACTION_TEXT) if args.all_actions else 2,
               'threshold_calibrated': False if args.all_actions else 'Calibration-fitted; precision not guaranteed on new images',
               'images': len(prompts), 'modes': dict(Counter(e['mode'] for e in prompts.values())),
               'selected_relations': sum(len(e['relations']) for e in prompts.values()),
               'selected_predicates': dict(Counter(c['predicate'] for e in prompts.values() for c in e['relations'])),
               'above_threshold_claims': sum(e['above_threshold_claims'] for e in prompts.values()),
               'capped_claims': sum(e['capped_claims'] for e in prompts.values()),
               'policy_sha256': identity(policy), 'policy': policy,
               'source': str(args.audit_zip or args.cache),
               'source_sha256': digest(args.audit_zip or args.cache / 'cache_identity.json'),
               'source_scope': '3374 reused relation audit images, not full COCO' if args.audit_zip else 'completed full-cache shards',
               'partial_allowed': args.allow_partial, 'object_threshold': args.object_threshold,
               'max_relations': args.max_relations, 'max_object_names': args.max_object_names,
               'label_based_selection': False, 'training': False, 'caption_metrics': False,
               'limits': [('All 116 positive HICO actions; common native-score threshold is exploratory, NOT calibrated.'
                           if args.all_actions else 'Holding/riding only; other cached verbs not used.'),
                          'Candidate previews include rejected relations and are never fed into prompt text.',
                          'Object fallback threshold/caps are fixed engineering settings, not validated precision.',
                          'No claim of full scene coverage, instance grounding by decoder, or caption improvement.']}
    if args.cache:
        meta = json.loads((args.cache / 'cache_identity.json').read_text())
        summary['expected_images'] = sum(meta['identity']['split_images'].values())
        summary['complete'] = len(prompts) == summary['expected_images']
    else:
        summary['complete'] = False
    # Hash-order alternating modes gives a reproducible review, not a representative sample.
    ordered = {mode: sorted((n for n, e in prompts.items() if e['mode'] == mode),
                            key=lambda n: hashlib.sha256(('upt-review-v1:' + n).encode()).hexdigest())
               for mode in ('object_relation', 'object_only', 'no_objects')}
    review = []
    while len(review) < min(50, len(prompts)):
        for mode in ordered:
            if ordered[mode] and len(review) < 50:
                name = ordered[mode].pop(0)
                review.append({'filename': name, **prompts[name], 'human_status': '', 'human_notes': ''})
    args.output.mkdir(parents=True)
    for name, value in [('prompt_cache.json', prompts), ('prompt_summary.json', summary), ('prompt_review_50.json', review)]:
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    cards = []
    for row in review:
        source = review_image(row['filename'], args.output, args.review_images)
        previews = '<br>'.join(html.escape(c['text']) + f" | {c['score']:.3f} | " +
                              ('qua ngưỡng' if c['passes_locked_threshold'] else 'bị từ chối') for c in row['candidate_preview'])
        rectangles = []
        index = {o['index']: o for o in row['objects']}
        for c in row['relations']:
            for key, color in [('subject_index', '#1565c0'), ('object_index', '#c75a00')]:
                x1, y1, x2, y2 = index[c[key]]['bbox']
                rectangles.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="{color}" stroke-width=".005"/>')
        cards.append(f'<article><h2>{html.escape(row["filename"])}</h2><div class="photo"><img loading="lazy" src="{source}"><svg viewBox="0 0 1 1" preserveAspectRatio="none">{"".join(rectangles)}</svg></div><p><b>Prompt:</b> {html.escape(row["prompt"])}</p><p>Mode: {row["mode"]}</p><details><summary>Ứng viên trước bộ lọc (không phải prompt)</summary>{previews}</details></article>')
    page = '<!doctype html><meta charset="utf-8"><title>UPT prompt review 50</title><style>body{font:18px Arial;background:#eef2f6;max-width:1000px;margin:25px auto;padding:15px}article{background:white;padding:20px;margin:20px 0;border-radius:12px}.photo{position:relative}.photo img{width:100%;display:block}.photo svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}p{line-height:1.5}</style><h1>50 prompt UPT để kiểm tra</h1><p>Mẫu có chủ đích, chia theo mode; không đại diện phân phối ảnh. Ảnh tải từ COCO qua Internet. Box xanh: người; cam: vật trong quan hệ được chọn. Bản này chỉ dùng holding/riding với ngưỡng cũ, chưa chứng minh tăng chất lượng caption.</p>' + ''.join(cards)
    if args.review_images is not None:
        page = page.replace('Ảnh tải từ COCO qua Internet.', 'Ảnh đã lưu kèm, xem được offline.')
    if args.all_actions:
        page = page.replace('Bản này chỉ dùng holding/riding với ngưỡng cũ, chưa chứng minh tăng chất lượng caption.',
                            f'Bản mở rộng 116 hành động, ngưỡng thử nghiệm {args.relation_threshold}; chưa hiệu chuẩn hoặc chứng minh tăng chất lượng caption.')
    (args.output / 'prompt_review_50.html').write_text(page, encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--audit-zip', type=Path)
    source.add_argument('--cache', type=Path)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--all-actions', action='store_true', help='Use all 116 positive HICO actions; full-cache input only.')
    parser.add_argument('--relation-threshold', type=float, default=.5, help='Exploratory native score threshold in all-actions mode.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review-images', type=Path, help='Local COCO image directory; copy 50 review images for offline viewing.')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--max-relations', type=int, default=3)
    parser.add_argument('--object-threshold', type=float, default=.5)
    parser.add_argument('--max-object-names', type=int, default=8)
    run(parser.parse_args())
