"""Separate actor verifier development trial on cached V2 and detector features."""
import argparse
import io
import json
import hashlib
import zipfile
from pathlib import Path
from collections import Counter
import torch
from captioning.actor_compatibility import ActorCompatibility, CompactActorCompatibility, actor_supervision, actor_loss, pair_geometry, actor_edge_metrics
from captioning.supervised_relations import RELATIONS
from captioning.selected_relation_prompts import build_selected_prompts
from run_instance_alignment_diagnostic import fit_thresholds, edge_metrics, metrics
from evaluate_frozen_residual import sha
from captioning.feature_archive import open_feature_archive


def read_archive(path, pair_names):
    with open_feature_archive(path) as archive:
        def read(name):
            files = [f for f in archive.namelist() if f.endswith(name)]
            if len(files) != 1:
                raise ValueError(name)
            return archive.read(files[0])
        raw = [read('data/'+name) for name in pair_names]
        rows = sum([json.loads(value) for value in raw], [])
        records = sum([torch.load(io.BytesIO(read(f'features_{i}.pt')), map_location='cpu', weights_only=True)
                       for i in range(2)], [])
        records.sort(key=lambda r: r['index'])
        if [r['index'] for r in records] != list(range(len(rows))):
            raise ValueError('Coverage mismatch.')
        views = torch.stack([r['views'] for r in records])
        text = torch.stack([r['text'] for r in records])
        if views.shape[1:] != (3, 512) or not torch.isfinite(views).all():
            raise ValueError('Invalid visual features.')
        if not torch.allclose(views.norm(dim=-1), torch.ones(views.shape[:2]), atol=1e-3):
            raise ValueError('Unnormalized features.')
        if pair_names == ['train_pairs.json', 'val_pairs.json']:
            summary = json.loads(read('data/pair_summary.json'))
            if summary['variant'] == 'instance_contrast_alignment_diagnostic_v2':
                config = json.loads(read('diagnostic_summary.json'))
                if config['variant'] != summary['variant']:
                    raise ValueError('Diagnostic provenance mismatch.')
            elif summary['variant'] == 'actor_compatibility_all_endpoints_v1':
                config = json.loads(read('manifest.json'))['configuration']
                if config['training'] is not False or config['model_revision'] != '57c216476eefef5ab752ec549e440a49ae4ae5f3':
                    raise ValueError('Actor feature model mismatch.')
            elif summary['variant'] == 'triplet_detector_training_features_v1':
                config=json.loads(read('manifest.json'))['configuration']
                if config['training'] is not False or config['model_revision']!='57c216476eefef5ab752ec549e440a49ae4ae5f3':
                    raise ValueError('Detector training feature model mismatch.')
                from evaluate_frozen_residual import newline_hashes
                protocol_path='audit/triplet_detector_training_protocol.json'
                protocol=json.loads(Path(protocol_path).read_text())
                if config['protocol_sha256'] not in newline_hashes(protocol_path):raise ValueError('Training protocol changed.')
                for split,data in zip(('train','val'),raw):
                    reference=json.loads(Path(f'audit/triplet_detector_{split}_reference_pairs.json').read_text())
                    lookup={(r['image_id'],r['subject_id'],r['object_id']):r for r in reference}
                    for row in json.loads(data):
                        original=lookup[(row['image_id'],row['subject_id'],row['object_id'])]
                        if any(row[k]!=original[k] for k in ('labels','filename','subject','object','caption_split')):
                            raise ValueError('Detector feature changed supervision.')
                        if row['gt_subject_box']!=original['subject_box'] or row['gt_object_box']!=original['object_box']:
                            raise ValueError('Detector source changed GT boxes.')
                        if row['subject_match_iou']<protocol['min_iou'] or row['object_match_iou']<protocol['min_iou']:
                            raise ValueError('Insufficient matching IoU.')
            else:
                raise ValueError('Unsupported actor training archive.')
            for split, data in zip(('train', 'val'), raw):
                if hashlib.sha256(data).hexdigest() != summary[split]['pair_json_sha256']:
                    raise ValueError('Training pair hash mismatch.')
        else:
            summary = json.loads(read('data/full_candidate_summary.json'))
            if hashlib.sha256(raw[0]).hexdigest() != summary['pair_json_sha256']:
                raise ValueError('Candidate hash mismatch.')
            config = json.loads(read('manifest.json'))['configuration']
            if config['training'] is not False or config['model_revision'] != '57c216476eefef5ab752ec549e440a49ae4ae5f3':
                raise ValueError('Candidate model mismatch.')
        if text.shape[1:] != (6,512) or not torch.isfinite(text).all():
            raise ValueError('Invalid text features.')
        return rows, views, text, [len(json.loads(data)) for data in raw]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-source', required=True)
    p.add_argument('--candidate-source', required=True)
    p.add_argument('--scored', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--compact', action='store_true', help='Linear150parameter diagnostic, fixed CLIP similarities.')
    p.add_argument('--all-actor-edges', action='store_true', help='Use every confirmed actor edge in the supplied archive.')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    rows, views, text, lengths = read_archive(args.train_source, ['train_pairs.json', 'val_pairs.json'])
    n = lengths[0]
    train, dev = rows[:n], rows[n:]
    if {r['image_id'] for r in train} & {r['image_id'] for r in dev}:
        raise ValueError('Train/dev image leakage.')
    if any(r['caption_split'] not in ('train', 'restval') for r in rows):
        raise ValueError('Caption heldout leakage.')
    labels, edges = actor_supervision(train, exhaustive=args.all_actor_edges)
    geometry = pair_geometry(rows)
    model = CompactActorCompatibility() if args.compact else ActorCompatibility()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    history = []
    for epoch in range(30):
        optimizer.zero_grad()
        logits = model(views[:n], geometry[:n], text[:n])
        loss, bce, ranking = actor_loss(logits, labels, edges)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        history.append({'epoch': epoch+1, 'loss': float(loss.detach()),
                        'bce': float(bce.detach()), 'ranking': float(ranking.detach())})
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        logits = model(views, geometry, text)
        activations = logits.sigmoid()  # Activation only, not calibrated probability.
    thresholds = fit_thresholds(activations[:n], labels)
    support = [{'positive': int((labels[:, c] == 1).sum()), 'negative': int((labels[:, c] == 0).sum())}
               for c in range(6)]
    enabled = [s['positive'] >= 5 and s['negative'] >= 5 for s in support]
    dev_labels, _ = actor_supervision(dev, exhaustive=args.all_actor_edges)
    candidates, candidate_views, candidate_text, _ = read_archive(args.candidate_source, ['evaluation_pairs.json'])
    old_summary = json.loads((Path(args.scored)/'full_candidate_audit_summary.json').read_text())
    if sha(args.candidate_source) != old_summary['source_zip_sha256']:
        raise ValueError('Candidate archive changed.')
    if {r['image_id'] for r in candidates} & {r['image_id'] for r in train}:
        raise ValueError('Candidate/train image overlap.')
    predictions = json.loads((Path(args.scored)/'all_candidate_predictions.json').read_text())
    if len(predictions) != len(candidates) or any(any(a[k] != b[k] for k in b) for a,b in zip(predictions,candidates)):
        raise ValueError('Candidate order mismatch.')
    with torch.inference_mode():
        candidate_logits = model(candidate_views, pair_geometry(candidates), candidate_text)
        candidate_activations = candidate_logits.sigmoid()
    # Noncompetitive veto permits multiple supported actors; no inferred unknown negatives.
    modified = []
    decisions = []
    for i, row in enumerate(predictions):
        scores = row['scores'].copy()
        for c in range(6):
            allowed = not enabled[c] or candidate_activations[i,c] >= thresholds[c]
            if scores[c] >= old_summary['thresholds'][c]:
                decisions.append({'filename': row['filename'], 'subject_id': row['subject_id'],
                    'object_id': row['object_id'], 'relation': RELATIONS[c], 'original_score': scores[c],
                    'actor_activation': float(candidate_activations[i,c]),
                    'actor_threshold': thresholds[c], 'enabled': enabled[c], 'allowed': bool(allowed)})
            if not allowed:
                scores[c] = -2.0
        modified.append({**row, 'scores': scores})
    # All original inventory objects are retained, even those without remaining relations.
    original_prompts = json.loads((Path(args.scored)/'selected_prompt_cache.json').read_text())
    inventory = {filename: [{'label': o['label'], 'bbox': o['bbox'], 'confidence': o['confidence']}
                            for o in entry['objects']] for filename,entry in original_prompts.items()}
    prompts = build_selected_prompts(modified, old_summary['thresholds'], object_inventory=inventory)
    architecture = 'compact_actor_compatibility_v1' if args.compact else 'actor_compatibility_v1'
    result = {'variant': architecture, 'seed':42,'epochs':30,
        'parameters':sum(p.numel() for p in model.parameters()),
        'training_pairs':n, 'training_actor_edges':len(edges),
        'all_actor_edges':args.all_actor_edges,
        'training_actor_images':len({train[int(e[0])]['image_id'] for e in edges}),
        'support':dict(zip(RELATIONS,support)), 'enabled':dict(zip(RELATIONS,enabled)),
        'thresholds':dict(zip(RELATIONS,thresholds)), 'history':history,
        'dev_actor_endpoint_metrics':metrics(activations[n:],dev_labels,thresholds),
        'dev_actor_edges':actor_edge_metrics(logits[n:],dev,exhaustive=args.all_actor_edges),
        'candidate_selected_claims':sum(len(e['relations']) for e in prompts.values()),
        'candidate_modes':dict(Counter(e['mode'] for e in prompts.values())),
        'train_source_sha256':sha(args.train_source), 'candidate_source_sha256':sha(args.candidate_source),
        'frozen_residual_checkpoint_sha256':old_summary['checkpoint_sha256'],
        'notes':'Only confirmed same-object actor edge endpoints supervise verifier. Negatives mean annotated action absence, '
                'not exhaustive wrong-object labels. Union never enters actor head. Scores are sigmoid activations, '
                'not calibrated probabilities. Training-only threshold fit; >=5pos/5neg to enable veto, '
                'otherwise baseline passes through unverified. Fixed30epochs, no dev selection. '
                'Previously inspected development sets, no fresh heldout/caption/novelty claim. '
                'Actor edge margin metric in reused helper uses0.2 logits, not original cosine scale.'}
    output = Path(args.output);output.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'architecture':architecture,'thresholds':thresholds,
                'enabled':enabled,'relations':RELATIONS},output/'actor_head.pt')
    for name,value in [('actor_summary.json',result),('candidate_veto_decisions.json',decisions),
                       ('verified_prompt_cache.json',prompts),('dev_actor_predictions.json',[
                           {**row,'actor_logits':logits[n+i].tolist(),'actor_activation':activations[n+i].tolist()}
                           for i,row in enumerate(dev)])]:
        (output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2),flush=True)
    print('CANDIDATE DECISIONS',json.dumps(decisions,indent=2),flush=True)


if __name__ == '__main__':
    main()
