"""USER-RUN CPU saved-caption metrics only. No Torch/model/checkpoint or TEST."""
import argparse
import gzip
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from saved_quality_data import check_rows,chair_counts

CHAIR_SHA='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def checked_payload(path,manifest):
    if sha(path)!=manifest['payload_sha256']:raise ValueError('Changed sealed caption package')
    payload=json.loads(gzip.decompress(path.read_bytes()))
    refs={int(k):v for k,v in payload['reference_ptb'].items()}
    if len(refs)!=5000 or sum(map(len,refs.values()))!=25010 or set(payload['models'])!={'CONTROL10','PROMPT9','PROMPT10'}:raise ValueError('Wrong fixed VALID5000 package')
    for name,model in payload['models'].items():
        check_rows(model['rows'],set(refs))
        if model['expected_CIDEr']!=manifest['expected_CIDEr'][name]:raise ValueError('Replay target changed')
    return payload,refs


def paired(a,b,resamples=2000):
    if set(a)!=set(b) or not a:raise ValueError('Unpaired metric IDs')
    ids=sorted(a);d=np.array([b[i]['CIDEr']-a[i]['CIDEr'] for i in ids],np.float64)
    h=np.array([int(b[i]['hallucination'])-int(a[i]['hallucination']) for i in ids],np.float64)
    if not np.isfinite(d).all():raise ValueError('Nonfinite per-image CIDEr')
    rng=np.random.default_rng(42);means=[];hallu=[]
    for _ in range(resamples//100):
        take=rng.integers(len(ids),size=(100,len(ids)));means.extend(d[take].mean(1));hallu.extend(h[take].mean(1))
    return dict(CIDEr_delta=float(d.mean()),CIDEr_paired95=np.quantile(means,[.025,.975]).tolist(),CHAIRs_delta=float(h.mean()),CHAIRs_paired95=np.quantile(hallu,[.025,.975]).tolist(),repaired=int((h==-1).sum()),introduced=int((h==1).sum()),resamples=resamples,conditional_on_fixed_corpus_IDF_and_selected_checkpoints=True)


def run(args):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'));payload,refs=checked_payload(args.payload,manifest)
    if sha(args.chair/'chair.pkl')!=CHAIR_SHA or importlib.metadata.version('pycocoevalcap')!='1.2':raise ValueError('Evaluator identity differs')
    args.output.mkdir(parents=True,exist_ok=True);summaries={};per={};mismatches=[]
    save(args.output/'evaluator_provenance.json',dict(packages={k:importlib.metadata.version(k) for k in ('pycocoevalcap','numpy','nltk')},tokenizer_sha256=sha(Path(inspect.getfile(PTBTokenizer))),cider_sha256=sha(Path(inspect.getfile(Cider))),chair_cache_sha256=CHAIR_SHA,reference_tokens_reused_exactly=True))
    for name,model in payload['models'].items():
        rows=model['rows'];hyp=PTBTokenizer().tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in rows})
        cider,values=Cider().compute_score(refs,hyp)
        if abs(float(cider)-model['expected_CIDEr'])>1e-7:mismatches.append(dict(model=name,expected=model['expected_CIDEr'],actual=float(cider)))
        save(args.output/'replay_mismatches.json',mismatches)
        if mismatches:raise ValueError('CIDEr replay failed; keep diagnostics, no tolerance change')
        bleu,_=Bleu(4).compute_score(refs,hyp);rouge,_=Rouge().compute_score(refs,hyp)
        inp=args.output/(name+'_chair_input.json');out=args.output/(name+'_chair_output.json')
        save(inp,[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
        subprocess.run([sys.executable,'-u',args.chair/'chair.py','--cap_file',inp,'--cache',args.chair/'chair.pkl','--save_path',out],cwd=args.chair,check=True,timeout=600)
        chair=json.loads(out.read_text(encoding='utf-8'));metric,counts=chair_counts(chair,rows)
        summaries[name]=dict(CIDEr=float(cider),Bleu_4=float(bleu[3]),ROUGE_L=float(rouge),**metric,counts=counts,mean_words=float(np.mean([len(r['caption'].split()) for r in rows])),epoch=model['epoch'])
        # CIDEr values follow reference dictionary insertion order, not rank-concat caption order.
        scores={i:float(v) for i,v in zip(refs,values)};sentences={r['image_id']:r for r in chair['sentences']}
        per[name]={r['coco_id']:dict(CIDEr=scores[r['coco_id']],hallucination=bool(sentences[r['coco_id']]['mscoco_hallucinated_words'])) for r in rows}
        if abs(np.mean(list(scores.values()))-cider)>1e-12:raise ValueError('Per-image CIDEr aggregation differs')
        save(args.output/(name+'_metrics.json'),summaries[name]);save(args.output/(name+'_per_image.json'),per[name]);print('METRICS',name,summaries[name],flush=True)
    comparisons={name+'_minus_CONTROL10':paired(per['CONTROL10'],per[name]) for name in ('PROMPT9','PROMPT10')}
    save(args.output/'comparison.json',dict(status='PROMPT_SCRATCH_QUALITY_COMPLETE',metrics=summaries,paired=comparisons,replay_mismatches=mismatches,model_executed=False,checkpoints_loaded=False,new_generation=False,test_used=False,limits=['One seed, reused VALID5000; exploratory comparison, not TEST.','Best epochs locked by prior CIDEr; not reselected using CHAIR.','CHAIR evaluates objects, not relation or attribute correctness.']))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('payload','manifest','chair','output'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
