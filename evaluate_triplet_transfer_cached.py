"""Frozen conditional detector transfer for the text-conditioned verifier and ablation."""
import argparse,io,json,zipfile,hashlib
from pathlib import Path
import torch
from captioning.triplet_verifier import TripletVerifier
from captioning.actor_compatibility import pair_geometry,actor_edge_metrics
from evaluate_frozen_residual import sha
from run_instance_alignment_diagnostic import metrics


def main():
    p=argparse.ArgumentParser()
    for name in ('source','head-directory','reference-directory','v2-source','actor-source','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args();torch.set_num_threads(2)
    summary=json.loads((Path(args.head_directory)/'triplet_summary.json').read_text())
    train_ids=set()
    for name in ('v2','actor'):
        source=getattr(args,name+'_source')
        if sha(source)!=summary['sources'][name]:raise ValueError('Training source changed.')
        with zipfile.ZipFile(source) as z:
            files=[f for f in z.namelist() if f.endswith('data/train_pairs.json')]
            if len(files)!=1:raise ValueError('Training pair source ambiguous.')
            train_ids.update(r['image_id'] for r in json.loads(z.read(files[0])))
    reference_dir=Path(args.reference_directory)
    baseline_summary=json.loads((reference_dir/'frozen_evaluation_summary.json').read_text())
    if sha(args.source)!=baseline_summary['source_zip_sha256']:raise ValueError('Transfer source changed.')
    with zipfile.ZipFile(args.source) as z:
        def read(name):
            files=[f for f in z.namelist() if f.endswith(name)]
            if len(files)!=1:raise ValueError(name)
            return z.read(files[0])
        raw=read('data/evaluation_pairs.json');rows=json.loads(raw)
        provenance=json.loads(read('data/detector_pair_summary.json'))
        if hashlib.sha256(raw).hexdigest()!=provenance['pair_json_sha256']:raise ValueError('Pair hash mismatch.')
        configuration=json.loads(read('manifest.json'))['configuration']
        if configuration['training'] is not False or configuration['model_revision']!='57c216476eefef5ab752ec549e440a49ae4ae5f3':raise ValueError('Feature model changed.')
        records=sum([torch.load(io.BytesIO(read(f'features_{i}.pt')),map_location='cpu',weights_only=True) for i in range(2)],[])
    if train_ids & {r['image_id'] for r in rows}:raise ValueError('Training/transfer overlap.')
    records.sort(key=lambda r:r['index'])
    if [r['index'] for r in records]!=list(range(len(rows))):raise ValueError('Feature order invalid.')
    views=torch.stack([r['views'] for r in records]);text=torch.stack([r['text'] for r in records])
    if not torch.isfinite(views).all() or not torch.isfinite(text).all():raise ValueError('Nonfinite features.')
    cp=torch.load(Path(args.head_directory)/'triplet_head.pt',map_location='cpu',weights_only=True)
    model=TripletVerifier();model.load_state_dict(cp['model']);model.eval().requires_grad_(False)
    if cp.get('relation_only_text',False):text=cp['relation_bank'][None].expand(len(rows),-1,-1)
    with torch.no_grad():logits=model(views,pair_geometry(rows),text);scores=logits.sigmoid()
    reference=json.loads((reference_dir/'predictions.json').read_text())
    if len(rows)!=len(reference) or any(any(r[k]!=s[k] for k in ('image_id','subject_id','object_id','labels')) for r,s in zip(rows,reference)):
        raise ValueError('Reference pair identity changed.')
    labels=torch.tensor([r['labels'] for r in rows]);before=torch.tensor([r['scores'] for r in reference])>=torch.tensor(baseline_summary['residual']['thresholds'])
    after=before&(scores>=torch.tensor(cp['thresholds']))
    def counts(mask):
        tp=int((mask&(labels==1)).sum());fp=int((mask&(labels==0)).sum())
        return {'positive':tp,'negative':fp,'unknown':int((mask&(labels<0)).sum()),'precision_on_known':tp/(tp+fp) if tp+fp else None,
                'recall_on_known':tp/int((labels==1).sum())}
    result={'variant':summary['variant'],'pairs':len(rows),'images':len({r['image_id'] for r in rows}),
        'head_sha256':sha(Path(args.head_directory)/'triplet_head.pt'),'source_sha256':sha(args.source),
        'train_image_overlap':0,'training_performed':False,'threshold_refit':False,
        'before':counts(before),'after':counts(after),'lost_positives':int((before&~after&(labels==1)).sum()),
        'removed_negatives':int((before&~after&(labels==0)).sum()),'verifier_actor_edges':actor_edge_metrics(logits,rows,exhaustive=True),
        'verifier_metrics_all_known':metrics(scores,labels,cp['thresholds']),
        'notes':'ReusedR55 matched detector pool; sequential development,not freshheldout/detectionAP/caption quality. Unknown excluded from precision; frozen thresholds, no fitting here.'}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    (output/'transfer_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='verifier_metrics_all_known'},indent=2),flush=True)


if __name__=='__main__':main()
