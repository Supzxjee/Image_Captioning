"""Frozen CPU inference + prompt selection + visual review, without GT labels."""
import argparse
import html
import io
import json
import zipfile
from pathlib import Path
from collections import Counter
import torch
from captioning.instance_contrast_alignment import ResidualInstanceAlignment
from captioning.selected_relation_prompts import build_selected_prompts
from captioning.detector_candidates import person_object_pairs
from evaluate_frozen_residual import sha, newline_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--protocol', default='audit/residual_frozen_evaluation_protocol.json')
    parser.add_argument('--image-list', default='audit/residual_full_candidate_images_20.json')
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    if sha(args.checkpoint) != protocol['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.source) as archive:
        def read(base):
            matches = [n for n in archive.namelist() if n.endswith(base)]
            if len(matches) != 1:
                raise ValueError(f'Expected one {base}.')
            return archive.read(matches[0])
        raw = read('data/evaluation_pairs.json')
        rows = json.loads(raw)
        inventory = json.loads(read('data/object_inventory.json'))
        summary = json.loads(read('data/full_candidate_summary.json'))
        import hashlib
        if hashlib.sha256(raw).hexdigest() != summary['pair_json_sha256']:
            raise ValueError('Pair hash mismatch.')
        if summary['image_list_sha256'] not in newline_hashes(args.image_list):
            raise ValueError('Audit image selection changed.')
        config = json.loads(read('manifest.json'))['configuration']
        if (config['model'] != 'openai/clip-vit-base-patch16'
                or config['model_revision'] != '57c216476eefef5ab752ec549e440a49ae4ae5f3'
                or config['training'] is not False):
            raise ValueError('Feature model/protocol mismatch.')
        selected = json.loads(Path(args.image_list).read_text())['images']
        if set(inventory) != {x['filename'] for x in selected} or len(rows) != summary['candidate_pairs']:
            raise ValueError('Inventory/image coverage mismatch.')
        expected_pairs = [r for image in selected for r in person_object_pairs(image, inventory[image['filename']])]
        if rows != expected_pairs:
            raise ValueError('Candidate pool is incomplete or uses different instances.')
        if any('labels' in r or 'gt_subject_box' in r or 'gt_object_box' in r for r in rows):
            raise ValueError('Full candidate inference must not use GT-labelled matched pool.')
        records = []
        for shard in range(2):
            records += torch.load(io.BytesIO(read(f'features_{shard}.pt')), map_location='cpu', weights_only=True)
        for filename in inventory:
            stem = Path(filename).stem
            for suffix in ('full', 'marked'):
                name = stem+'_'+suffix+'.jpg'
                (output / name).write_bytes(read('data/'+name))
    records.sort(key=lambda r: r['index'])
    if [r['index'] for r in records] != list(range(len(rows))):
        raise ValueError('Feature coverage invalid.')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['scale'] != protocol['scale'] or checkpoint['relations'] != protocol['relations']:
        raise ValueError('Model scale/vocabulary mismatch.')
    model = ResidualInstanceAlignment(scale=checkpoint['scale'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    torch.set_num_threads(2)
    if rows:
        views, text = torch.stack([r['views'] for r in records]), torch.stack([r['text'] for r in records])
        if views.shape[1:] != (3, 512) or text.shape[1:] != (6, 512):
            raise ValueError('Invalid feature shapes.')
        if not torch.isfinite(views).all() or not torch.isfinite(text).all():
            raise ValueError('Nonfinite features.')
        if not torch.allclose(views.norm(dim=-1), torch.ones(views.shape[:2]), atol=1e-3):
            raise ValueError('Image features not normalized.')
        with torch.inference_mode():
            scores, base, delta = model(views, text)
        predictions = [{**row, 'scores': scores[i].tolist(), 'clip': base[i].tolist(),
                         'correction': delta[i].tolist()} for i, row in enumerate(rows)]
    else:
        predictions = []
    prompts = build_selected_prompts(predictions, checkpoint['thresholds'], max_relations=3,
                                      object_inventory=inventory)
    (output / 'selected_prompt_cache.json').write_text(json.dumps(prompts, indent=2), encoding='utf-8')
    (output / 'all_candidate_predictions.json').write_text(json.dumps(predictions, indent=2), encoding='utf-8')
    import csv
    with (output / 'selected_claim_review.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = ['filename', 'subject_instance', 'subject', 'relation', 'object_instance', 'object',
                  'score', 'threshold', 'prompt', 'human_status', 'human_notes']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for filename, entry in prompts.items():
            for claim in entry['relations']:
                writer.writerow({k: ({**claim, 'filename': filename}).get(k, '') for k in fields})
    cards = []
    for filename, entry in prompts.items():
        cards.append('<article><h3>'+html.escape(filename)+'</h3><img src="'+Path(filename).stem+
                     '_full.jpg"><img src="'+Path(filename).stem+'_marked.jpg"><pre>'+html.escape(
                         json.dumps(entry, ensure_ascii=False, indent=2))+'</pre></article>')
    (output / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Full candidate audit</title>'
        '<style>body{font:16px Arial;margin:24px}img{max-width:48%}pre{white-space:pre-wrap}'
        'article{border-bottom:1px solid #ccc;padding:20px}</style><h1>Review required, no ground-truth claims</h1>'
        + ''.join(cards), encoding='utf-8')
    result = {'variant': 'frozen_detector_full_candidate_audit_20_v1',
               'images': len(prompts), 'candidate_pairs': len(rows),
               'modes': dict(Counter(e['mode'] for e in prompts.values())),
               'selected_claims': sum(len(e['relations']) for e in prompts.values()),
               'capped_claims': sum(e['capped_claims'] for e in prompts.values()),
               'thresholds': checkpoint['thresholds'], 'checkpoint_sha256': sha(args.checkpoint),
               'source_zip_sha256': sha(args.source), 'preparation_summary': summary,
               'notes': 'All candidates within retained top10 detector objects, no GT matching/selection. '
                        'No automatic accuracy/precision: human visual review required. '
                        'Same previously evaluated image cohort, qualitative audit only; no train/refit.'}
    (output / 'full_candidate_audit_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
