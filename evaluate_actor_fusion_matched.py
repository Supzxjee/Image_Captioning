"""Conditional detector transfer audit with frozen actor/fusion heads and thresholds."""
import argparse,io,json,zipfile,hashlib
from pathlib import Path
import torch
from captioning.actor_compatibility import ActorCompatibility,pair_geometry
from captioning.instance_contrast_alignment import ResidualInstanceAlignment
from evaluate_frozen_residual import sha,newline_hashes
from run_instance_alignment_diagnostic import metrics
from run_actor_soft_fusion_cached import edges_report


def main():
    p=argparse.ArgumentParser()
    for name in ('source','actor-directory','fusion-directory','residual-head','reference-predictions','training-source','output'):
        p.add_argument('--'+name,required=True)
    args=p.parse_args();torch.set_num_threads(2)
    fs=json.loads((Path(args.fusion_directory)/'fusion_summary.json').read_text())
    if sha(args.training_source)!=fs['train_source_sha256']:
        raise ValueError('Verifier training archive changed.')
    with zipfile.ZipFile(args.training_source) as train_archive:
        train_files=[n for n in train_archive.namelist() if n.endswith('data/train_pairs.json')]
        if len(train_files)!=1:raise ValueError('Training metadata ambiguous.')
        train_ids={r['image_id'] for r in json.loads(train_archive.read(train_files[0]))}
    baseline_summary=json.loads((Path(args.reference_predictions).parent/'frozen_evaluation_summary.json').read_text())
    if sha(args.source)!=baseline_summary['source_zip_sha256']:
        raise ValueError('Transfer archive differs from verified R55 source.')
    if sha(args.residual_head)!=fs['residual_head_sha256'] or sha(Path(args.actor_directory)/'actor_head.pt')!=fs['actor_head_sha256']:
        raise ValueError('Frozen scorer changed.')
    with zipfile.ZipFile(args.source) as z:
        def read(name):
            files=[f for f in z.namelist() if f.endswith(name)]
            if len(files)!=1:raise ValueError(name)
            return z.read(files[0])
        raw=read('data/evaluation_pairs.json');rows=json.loads(raw)
        provenance=json.loads(read('data/detector_pair_summary.json'))
        if hashlib.sha256(raw).hexdigest()!=provenance['pair_json_sha256']:raise ValueError('Pair hash changed.')
        config=json.loads(read('manifest.json'))['configuration']
        if config['training'] is not False or config['model_revision']!='57c216476eefef5ab752ec549e440a49ae4ae5f3':
            raise ValueError('Feature model changed.')
        records=sum([torch.load(io.BytesIO(read(f'features_{i}.pt')),map_location='cpu',weights_only=True) for i in range(2)],[])
    if train_ids & {r['image_id'] for r in rows}:
        raise ValueError('Transfer/training image overlap.')
    records.sort(key=lambda r:r['index'])
    if [r['index'] for r in records]!=list(range(len(rows))):raise ValueError('Feature coverage mismatch.')
    views=torch.stack([r['views'] for r in records]);text=torch.stack([r['text'] for r in records])
    if not torch.isfinite(views).all() or not torch.isfinite(text).all():raise ValueError('Nonfinite features.')
    ac=torch.load(Path(args.actor_directory)/'actor_head.pt',map_location='cpu',weights_only=True)
    fc=torch.load(Path(args.fusion_directory)/'fusion_head.pt',map_location='cpu',weights_only=True)
    rc=torch.load(args.residual_head,map_location='cpu',weights_only=True)
    actor=ActorCompatibility();actor.load_state_dict(ac['model']);actor.eval().requires_grad_(False)
    residual=ResidualInstanceAlignment(scale=rc['scale']);residual.load_state_dict(rc['model']);residual.eval().requires_grad_(False)
    with torch.no_grad():
        base,_,_=residual(views,text);a=actor(views,pair_geometry(rows),text)
        combined=(((torch.stack([base,a],dim=-1)-fc['mean'])/fc['std'])*fc['weights']).sum(-1)+fc['bias']
        fusion=combined.sigmoid()
    reference=json.loads(Path(args.reference_predictions).read_text())
    if len(reference)!=len(rows) or any(any(x[k]!=r[k] for k in ('image_id','subject_id','object_id','labels')) for x,r in zip(reference,rows)):
        raise ValueError('Reference order/labels mismatch.')
    if not torch.allclose(base,torch.tensor([r['scores'] for r in reference]),atol=1e-6):
        raise ValueError('Frozen R55 scores mismatch.')
    labels=torch.tensor([r['labels'] for r in rows]);before=base>=torch.tensor(rc['thresholds'])
    after=before&(fusion>=torch.tensor(fc['thresholds']))
    def counts(mask):
        tp=int((mask&(labels==1)).sum());fp=int((mask&(labels==0)).sum())
        return {'positive':tp,'negative':fp,'unknown':int((mask&(labels<0)).sum()),
                'precision_on_known':tp/(tp+fp) if tp+fp else None,'recall_on_known':tp/int((labels==1).sum())}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    result={'variant':'frozen_actor_fusion_detector_matched_transfer_development_v1','pairs':len(rows),
        'images':len({r['image_id'] for r in rows}),'source_sha256':sha(args.source),
        'fusion_head_sha256':sha(Path(args.fusion_directory)/'fusion_head.pt'),
        'training_performed':False,'threshold_refit':False,'before':counts(before),'after':counts(after),
        'transfer_train_image_overlap':0,
        'lost_positives':int((before&~after&(labels==1)).sum()),'removed_negatives':int((before&~after&(labels==0)).sum()),
        'residual_actor_edges':edges_report(base,rows),'fusion_actor_edges':edges_report(fusion,rows),
        'notes':'Same reused R55 class/IoU-matched detector cohort; conditional transfer audit, not fresh heldout or detection AP. '
                'Known-label precision excludes unknown links; no unknown converted to negative. No training/refitting here.'}
    (output/'transfer_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (output/'transfer_predictions.json').write_text(json.dumps([{**r,'fusion_activation':fusion[i].tolist(),
        'baseline_accept':before[i].tolist(),'fused_accept':after[i].tolist()} for i,r in enumerate(rows)],indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
