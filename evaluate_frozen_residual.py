"""Frozen CPU evaluation on a pre-registered unseen-image feature archive."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
import torch
from captioning.instance_contrast_alignment import ResidualInstanceAlignment
from run_instance_alignment_diagnostic import metrics, edge_metrics
from run_residual_center_ablation import flips_vs_base


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(4*1024**2):
            digest.update(block)
    return digest.hexdigest()


def newline_hashes(path):
    """Accept only LF/CRLF representations of the exact frozen manifest bytes."""
    raw = Path(path).read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    return {hashlib.sha256(value).hexdigest() for value in (raw, lf, lf.replace(b'\n', b'\r\n'))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--protocol', default='audit/residual_frozen_evaluation_protocol.json')
    parser.add_argument('--exclusions', default='audit/instance_alignment_seen_images.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--detector-matched', action='store_true')
    parser.add_argument('--reference-pairs', default='audit/residual_new_image_pairs_500.json')
    parser.add_argument('--reference-predictions', default='')
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    if sha(args.checkpoint) != protocol['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint SHA mismatch.')
    exclusion_hashes = newline_hashes(args.exclusions)
    if protocol['exclusion_sha256'] not in exclusion_hashes:
        raise ValueError('Exclusion manifest changed.')
    seen = set(json.loads(Path(args.exclusions).read_text())['image_ids'])
    with zipfile.ZipFile(args.source) as archive:
        def read(base):
            matches = [n for n in archive.namelist() if n.endswith(base)]
            if len(matches) != 1:
                raise ValueError(f'Expected one {base}.')
            return archive.read(matches[0])
        raw = read('data/evaluation_pairs.json')
        rows = json.loads(raw)
        data_summary = json.loads(read('data/detector_pair_summary.json' if args.detector_matched
                                       else 'data/evaluation_pair_summary.json'))
        if hashlib.sha256(raw).hexdigest() != data_summary['pair_json_sha256']:
            raise ValueError('Pair hash mismatch.')
        if not args.detector_matched and data_summary['exclusion_sha256'] not in exclusion_hashes:
            raise ValueError('Source used different exclusions.')
        config = json.loads(read('manifest.json'))['configuration']
        if (config['model'] != 'openai/clip-vit-base-patch16'
                or config['model_revision'] != '57c216476eefef5ab752ec549e440a49ae4ae5f3'
                or config['training'] is not False):
            raise ValueError('Frozen feature model/protocol mismatch.')
        if not args.detector_matched:
            if data_summary['seed'] != protocol['selection_seed'] or len(rows) != protocol['evaluation_pairs']:
                raise ValueError('Evaluation selection protocol mismatch.')
        else:
            if data_summary['source_pairs_sha256'] not in newline_hashes(args.reference_pairs):
                raise ValueError('Different reference cohort.')
            if data_summary['reference_pairs'] != 500 or data_summary['matched_pairs'] != len(rows) or not rows:
                raise ValueError('Matched pair coverage mismatch.')
            references = json.loads(Path(args.reference_pairs).read_text())
            gt = {(r['subject_id'], r['object_id']): r for r in references}
            for row in rows:
                reference = gt.get((row['subject_id'], row['object_id']))
                if reference is None or any(row[k] != reference[k] for k in (
                        'labels', 'filename', 'image_id', 'subject', 'object')):
                    raise ValueError('Matched pair changed GT labels/identity.')
        records = []
        for shard in range(2):
            records += torch.load(io.BytesIO(read(f'features_{shard}.pt')), map_location='cpu', weights_only=True)
    if {r['image_id'] for r in rows} & seen:
        raise ValueError('Previously used image included.')
    if any(r['caption_split'] not in {'train', 'restval'} for r in rows):
        raise ValueError('Caption heldout included.')
    if len({(r['subject_id'], r['object_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate pairs.')
    if any(len(r['labels']) != 6 or any(v not in (-1, 0, 1) for v in r['labels']) for r in rows):
        raise ValueError('Invalid labels.')
    records.sort(key=lambda r: r['index'])
    if [r['index'] for r in records] != list(range(len(rows))):
        raise ValueError('Feature coverage invalid.')
    views, text = torch.stack([r['views'] for r in records]), torch.stack([r['text'] for r in records])
    if views.shape[1:] != (3, 512) or text.shape[1:] != (6, 512):
        raise ValueError('Invalid feature shape.')
    if not torch.isfinite(views).all() or not torch.isfinite(text).all():
        raise ValueError('Nonfinite feature.')
    if not torch.allclose(views.norm(dim=-1), torch.ones(views.shape[:2]), atol=1e-3):
        raise ValueError('Image features not normalized.')
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['relations'] != protocol['relations'] or checkpoint['scale'] != protocol['scale']:
        raise ValueError('Model vocabulary/scale mismatch.')
    model = ResidualInstanceAlignment(scale=checkpoint['scale'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        scores, base, delta = model(views, text)
    labels = torch.tensor([r['labels'] for r in rows])
    summary = {'variant': 'frozen_residual_new_image_evaluation_v1', 'protocol': protocol,
               'source_zip_sha256': sha(args.source), 'pairs': len(rows),
               'images': len({r['image_id'] for r in rows}), 'seen_image_overlap': 0,
               'training_performed': False, 'threshold_refit': False,
               'exclusion_hash_verification': 'Exact frozen content; LF/CRLF-only variants accepted',
               'flips_vs_clip': flips_vs_base(scores, base, rows),
               'notes': 'New for listed scorer experiments/audits. GT boxes and curated pairs; '
                        'not caption-heldout/official HOI role AP or proof of novelty.'}
    for name, values, thresholds in [('clip', base, protocol['clip_thresholds']),
                                     ('residual', scores, checkpoint['thresholds'])]:
        accepts = values >= torch.tensor(thresholds)
        summary[name] = {'thresholds': thresholds, 'metrics': metrics(values, labels, thresholds),
                         'edges': edge_metrics(values, rows),
                         'accepted_claims_all': int(accepts.sum()),
                         'accepted_unknown_claims': int((accepts & (labels < 0)).sum())}
    if args.detector_matched:
        summary['variant'] = 'frozen_residual_detector_matched_evaluation_v1'
        summary['matching_summary'] = data_summary
        summary['notes'] = ('Conditional evaluation on class/IoU-matched detector boxes, not full HOI detection AP. '
                            'Pair/positive coverage includes missing-cache cases reported separately. '
                            'Extra detector proposals do not automatically become negatives.')
        if args.reference_predictions:
            saved = json.loads(Path(args.reference_predictions).read_text())
            indexed = {(r['subject_id'], r['object_id']): r for r in saved}
            subset = [indexed[(r['subject_id'], r['object_id'])] for r in rows]
            for reference, row in zip(subset, rows):
                if reference['labels'] != row['labels'] or reference['image_id'] != row['image_id']:
                    raise ValueError('Reference predictions mismatch.')
            gt_scores = torch.tensor([r['scores'] for r in subset])
            gt_clip = torch.tensor([r['clip'] for r in subset])
            summary['gt_boxes_on_identical_matched_subset'] = {
                'residual': {'metrics': metrics(gt_scores, labels, checkpoint['thresholds']),
                             'edges': edge_metrics(gt_scores, rows)},
                'clip': {'metrics': metrics(gt_clip, labels, protocol['clip_thresholds']),
                         'edges': edge_metrics(gt_clip, rows)}}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'frozen_evaluation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'predictions.json').write_text(json.dumps([
        {**row, 'scores': scores[i].tolist(), 'clip': base[i].tolist(), 'correction': delta[i].tolist()}
        for i, row in enumerate(rows)], indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
