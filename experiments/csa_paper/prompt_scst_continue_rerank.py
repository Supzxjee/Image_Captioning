"""Same frozen reranker; added MID phase and predeclared checkpoint selection."""
import argparse
from pathlib import Path
import shutil
from real_bank_data import read,save,sha
from prompt_scst_continue_data import PHASES,select,source_checkpoint,SOURCE
import prompt_scst_rerank as original


def score(cfg):
    original.score(cfg)
    out=Path(cfg['output']);report=read(out/'result.json');result=report['metrics']
    for mode in ('decoder_rank0','fixed_rerank'):
        for key in ('CIDEr','CHAIRs','CHAIRi','Recall'):
            if abs(result['original'][mode][key]-SOURCE['expected_metrics'][mode][key])>1e-7:raise ValueError('Original source60 metric replay differs:'+mode+'/'+key)
    selected,eligible=select({p:m['fixed_rerank'] for p,m in result.items()})
    paths={'original':source_checkpoint(cfg['checkpoint_root'],cfg['arm']),'mid':out/'mid.bin','end':out/'checkpoint.bin'}
    for p,n in (('mid','mid_receipt.json'),('end','checkpoint_receipt.json')):
        receipt=read(out/n)
        if receipt['updates']!=PHASES[p] or sha(paths[p])!=receipt['sha256']:raise ValueError('Wrong selected checkpoint '+p)
    shutil.copy2(paths[selected],out/'best.bin')
    lock=dict(selected_name=selected,eligible_names=eligible,sha256=sha(out/'best.bin'),test_used=False,
        improved=result[selected]['fixed_rerank']['CIDEr']>result['original']['fixed_rerank']['CIDEr'],
        rule='max fixed-rerank CIDEr with CHAIRs/i<=original and Recall>=original; earlier ties; original fallback')
    save(out/'locked_selection.json',lock)
    report.update(status='PROMPT_SCST_CONTINUE_RERANK_COMPLETE',selection=lock,
        limitations=['One training seed and reused VALID; not independent TEST.',
                    'Continuation within PROMPT, not an isolated prompt-effect ablation.',
                    'CHAIR evaluates objects, not relation correctness.'])
    save(out/'result.json',report)
    print('CONTINUATION COMPLETE; selected',selected,'eligible',eligible,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','prefetch','worker','score'));p.add_argument('--config',required=True)
    a=p.parse_args();cfg=read(a.config);cfg['phase_steps']=PHASES
    if a.mode=='score':score(cfg)
    else:getattr(original,a.mode)(cfg)
