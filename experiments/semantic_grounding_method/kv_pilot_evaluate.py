"""User-run CPU official CIDEr/CHAIR and observed semantic diagnostics."""
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
from kv_pilot_data import save,digest,semantic_metrics,POLICY


def paired_ci(values):
    values = np.asarray(values,dtype=float)
    rng = np.random.default_rng(20261030)
    means = []
    for _ in range(10):
        means.extend(values[rng.integers(0,len(values),(1000,len(values)))].mean(1).tolist())
    return np.quantile(means,[.025,.975]).tolist()


def paired_false(a,b):
    repaired = sum(x==1 and y==0 for x,y in zip(a,b))
    introduced = sum(x==0 and y==1 for x,y in zip(a,b))
    n = repaired+introduced
    p = min(1.,2*sum(math.comb(n,i) for i in range(min(repaired,introduced)+1))/(2**n)) if n else 1.
    return dict(repaired=repaired,introduced=introduced,exact_mcnemar_p=p)


def semantic_report(records,groups):
    results = {}
    ids = sorted({r['image_id'] for r in records})
    for key in ('pre_score','effective_score','C0_decoder_score','end_decoder_score'):
        result = semantic_metrics(records,groups,key)
        rng = np.random.default_rng(20261030)
        aucs,concords,valid_groups = [],[],[]
        for _ in range(2000):
            draw = rng.choice(ids,len(ids),replace=True)
            unique,counts = np.unique(draw,return_counts=True)
            weights = dict(zip(unique.tolist(),counts.tolist()))
            b = semantic_metrics(records,groups,key,weights)
            if b['macro_family_auc'] is not None:aucs.append(b['macro_family_auc'])
            if b['query_concordance'] is not None:concords.append(b['query_concordance'])
            valid_groups.append(b['valid_query_groups'])
        result.update(macro_family_auc_image_bootstrap95=np.quantile(aucs,[.025,.975]).tolist() if aucs else None,
            query_concordance_image_bootstrap95=np.quantile(concords,[.025,.975]).tolist() if concords else None,
            bootstrap_draws=2000,valid_auc_draws=len(aucs),valid_concordance_draws=len(concords),
            bootstrap_valid_query_groups_min=min(valid_groups),bootstrap_valid_query_groups_max=max(valid_groups))
        results[key] = result
    return results


def evaluate(cfg):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    out = Path(cfg['output'])
    workers = [json.loads((out/f'worker_rank{r}.json').read_bytes()) for r in (0,1)]
    if any(w['status'] != 'ENDPOINT_500_COMPLETE' or w['updates'] != 500 for w in workers):
        raise ValueError('Incomplete endpoint; do not score a subset')
    for key in ('initial_digest','final_digest','initial_frozen_digest','final_frozen_digest','common_text_sha256'):
        if workers[0][key] != workers[1][key]:raise ValueError('Rank identity mismatch '+key)
    if workers[0]['initial_frozen_digest'] != workers[0]['final_frozen_digest']:raise ValueError('Frozen state drift')
    meta = json.loads((Path(cfg['prepared_root'])/'metadata.json').read_bytes())
    expected = {meta['rows'][i]['image_id']:meta['rows'][i] for i in meta['dev_indices']}
    ptb = PTBTokenizer()
    refs = ptb.tokenize({i:[dict(caption=s) for s in r['references']] for i,r in sorted(expected.items())})
    metrics,per_cider,flags = {},{},{}
    predictions = {}
    for condition in ('C0','end'):
        rows = sorted(sum([json.loads((out/f'{condition}_rank{rank}.json').read_bytes()) for rank in (0,1)],[]),key=lambda r:r['image_id'])
        if len(rows) != 1000 or {r['image_id']:r['coco_id'] for r in rows} != {i:r['coco_id'] for i,r in expected.items()}:
            raise ValueError('Wrong full1000 prediction cohort')
        predictions[condition] = rows
        save(out/f'{condition}_predictions.json',rows)
        hyp = ptb.tokenize({r['image_id']:[dict(caption=r['caption'])] for r in rows})
        cider,per = Cider().compute_score(refs,hyp)
        per_cider[condition] = np.asarray(per)
        save(out/f'{condition}_per_image_cider.json',[dict(image_id=r['image_id'],coco_id=r['coco_id'],CIDEr=float(v)) for r,v in zip(rows,per)])
        save(out/f'{condition}_chair_input.json',[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
        subprocess.run([sys.executable,'-u',str(Path(cfg['chair'])/'chair.py'),'--cap_file',str(out/f'{condition}_chair_input.json'),
            '--cache',str(Path(cfg['chair'])/'chair.pkl'),'--save_path',str(out/f'{condition}_chair_output.json')],cwd=cfg['chair'],check=True)
        chair = json.loads((out/f'{condition}_chair_output.json').read_bytes())
        sentences = chair['sentences']
        if len(sentences) != 1000 or {s['image_id'] for s in sentences} != {r['coco_id'] for r in rows}:
            raise ValueError('CHAIR image cohort differs')
        counts = dict(hallucinated_captions=sum(bool(s['mscoco_hallucinated_words']) for s in sentences),
            hallucinated_mentions=sum(len(s['mscoco_hallucinated_words']) for s in sentences),
            object_mentions=sum(len(s['mscoco_generated_words']) for s in sentences),
            recalled_unique=sum(len(set(s['mscoco_generated_words'])&set(s['mscoco_gt_words'])) for s in sentences),
            reference_unique=sum(len(set(s['mscoco_gt_words'])) for s in sentences))
        ratios = dict(CHAIRs=counts['hallucinated_captions']/1000,
            CHAIRi=counts['hallucinated_mentions']/max(1,counts['object_mentions']),
            Recall=counts['recalled_unique']/max(1,counts['reference_unique']))
        if any(abs(v-chair['overall_metrics'][k]) > 1e-12 for k,v in ratios.items()):
            raise ValueError('CHAIR aggregation mismatch')
        by_id = {s['image_id']:int(bool(s['mscoco_hallucinated_words'])) for s in sentences}
        flags[condition] = np.asarray([by_id[r['coco_id']] for r in rows])
        metrics[condition] = dict(CIDEr=float(cider),counts=counts,chair=ratios,
            mean_words=float(np.mean([len(r['caption'].split()) for r in rows])))
    aux = json.loads((Path(cfg['prepared_root'])/'auxiliary_facts.json').read_bytes())
    held = sum([json.loads((out/f'held_rank{r}.json').read_bytes())['records'] for r in (0,1)],[])
    if len(held) != 99 or {r['fact_id'] for r in held} != {r['fact_id'] for r in aux['partitions']['held']}:
        raise ValueError('Held facts incomplete')
    save(out/'held_records.json',held)
    semantic = semantic_report(held,aux['multipositive']['held'])
    save(out/'semantic_metrics.json',semantic)
    off = sum([json.loads((out/f'reader_off_rank{r}.json').read_bytes()) for r in (0,1)],[])
    if len(off) != 64 or len({r['coco_id'] for r in off}) != 64:raise ValueError('Reader-off cohort incomplete')
    end_by = {r['coco_id']:r['caption'] for r in predictions['end']}
    changed = sum(r['caption'] != end_by[r['coco_id']] for r in off)
    c0,end = metrics['C0'],metrics['end']
    effect = semantic['effective_score']
    sem_pass = effect['macro_family_auc'] >= .6 and effect['query_concordance'] >= .6
    guards = end['chair']['Recall'] >= c0['chair']['Recall']-.01 and .95 <= end['mean_words']/c0['mean_words'] <= 1.05
    summary = dict(status='ENDPOINT_500_EVALUATED_WAIT_MATCHED_DE',arm=cfg['arm'],updates=500,policy=POLICY,
        metrics=metrics,semantic=semantic,reader_off_changed_captions=changed,reader_off_images=64,
        end_minus_C0_CIDEr=float((per_cider['end']-per_cider['C0']).mean()),
        end_minus_C0_CIDEr_paired95=paired_ci(per_cider['end']-per_cider['C0']),
        end_minus_C0_CHAIRs=float((flags['end']-flags['C0']).mean()),
        end_minus_C0_CHAIRs_paired95=paired_ci(flags['end']-flags['C0']),
        false_caption_pairing=paired_false(flags['C0'],flags['end']),
        operational_semantic_screen=sem_pass,recall_length_guard=guards,
        full_continue_gate='WAIT_BOTH_ARMS: require E>D/E>C0 CIDEr, E<C0/E<=D false captions, semantics/branch/guards',
        initial_digest=workers[0]['initial_digest'],final_digest=workers[0]['final_digest'],frozen_digest=workers[0]['final_frozen_digest'],
        common_text_sha256=workers[0]['common_text_sha256'],checkpoint_sha256=digest(out/'checkpoint.bin'),
        source_commit=cfg['source_commit'],source_assets=cfg['source_assets'],
        prepared_receipt_sha256=digest(Path(cfg['prepared_root'])/'cpu_preparation.json'),
        pilot_preparation_payload_sha256=json.loads((out/'preparation.json').read_bytes())['payload_sha256'],
        effective_held_prompt_content_sha256=json.loads((out/'preparation.json').read_bytes())['effective_held_prompt_content_sha256'],
        test_used=False,one_seed=True,reused_development=True,relation_caption_truth='NOT_ESTABLISHED',
        directional_generalization='NOT_ESTABLISHED',quota_accounting=POLICY['quota_accounting'])
    save(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
