"""Reconcile historical caption results; recompute paired CHAIR from returned outputs.

No checkpoint inference, caption scorer execution, or historical metric invention.
"""
import argparse,hashlib,json,zipfile
from pathlib import Path
import numpy as np


def sha_bytes(data):return hashlib.sha256(data).hexdigest()


def chair_aggregate(payload):
    sentences=payload['sentences'];ids=[s['image_id'] for s in sentences]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate image IDs.')
    hallucinated=mentions=recall=gt=0
    for s in sentences:
        hw=s['mscoco_hallucinated_words'];words=s['mscoco_generated_words'];truth=s['mscoco_gt_words']
        flag=int(bool(hw))
        if s['metrics']['CHAIRs']!=flag:raise ValueError('Sentence hallucination flag differs.')
        hallucinated+=flag;mentions+=len(words);gt+=len(truth)
        recall+=len(set(words)&set(truth))
    result={'CHAIRs':hallucinated/len(sentences),'CHAIRi':sum(len(s['mscoco_hallucinated_words']) for s in sentences)/mentions,
        'Recall':recall/gt}
    if any(abs(result[k]-payload['overall_metrics'][k])>1e-12 for k in result):raise ValueError('Stored CHAIR aggregate is inconsistent.')
    return result,{'captions':len(sentences),'hallucinated_captions':hallucinated,'recognized_mentions':mentions,
        'hallucinated_mentions':sum(len(s['mscoco_hallucinated_words']) for s in sentences),'recalled_gt_objects':recall,'gt_objects':gt}


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    registry=json.loads(Path(args.registry).read_text(encoding='utf-8'))
    historical={r[0]:r for r in registry['rows']}
    root='e2_qualitative_analysis/'
    names=[root+x for x in ('qformer_chair_output.json','e2_chair_output.json','qformer_chair_input.json','e2_chair_input.json','transition_summary.json')]
    with zipfile.ZipFile(args.archive) as archive:raw={n:archive.read(n) for n in names}
    q,e=(json.loads(raw[root+n]) for n in ('qformer_chair_output.json','e2_chair_output.json'))
    qm,qcounts=chair_aggregate(q);em,ecounts=chair_aggregate(e)
    qlookup={s['image_id']:s for s in q['sentences']};elookup={s['image_id']:s for s in e['sentences']}
    if set(qlookup)!=set(elookup) or len(qlookup)!=5000:raise ValueError('Not the same5000 images.')
    for name,lookup in [('qformer_chair_input.json',qlookup),('e2_chair_input.json',elookup)]:
        entries=json.loads(raw[root+name]);input_lookup={x['image_id']:x['caption'] for x in entries}
        if set(input_lookup)!=set(lookup) or any(input_lookup[i]!=lookup[i]['caption'] for i in lookup):raise ValueError('Input/output captions differ.')
    if any(set(qlookup[i]['mscoco_gt_words'])!=set(elookup[i]['mscoco_gt_words']) for i in qlookup):raise ValueError('CHAIR ground-truth objects differ.')
    counts={'both_clean':0,'improved_to_clean':0,'regressed_to_hallucinated':0,'both_hallucinated':0}
    for image_id in sorted(qlookup):
        a,b=qlookup[image_id]['metrics']['CHAIRs'],elookup[image_id]['metrics']['CHAIRs']
        key='both_hallucinated' if a and b else 'improved_to_clean' if a else 'regressed_to_hallucinated' if b else 'both_clean'
        counts[key]+=1
    reported=json.loads(raw[root+'transition_summary.json'])
    if counts!=reported['transitions']:raise ValueError('Transition summary differs.')
    for metric_id,values in [('M12',qm),('M19',em)]:
        record=historical[metric_id]
        if any(abs(record[column]-values[k])>1e-9 for column,k in [(11,'CHAIRs'),(12,'CHAIRi'),(13,'Recall')]):raise ValueError('Registry CHAIR differs beyond recorded rounding.')
    generator=np.random.default_rng(42)
    bootstrap=generator.multinomial(5000,[counts[k]/5000 for k in counts],size=20000)
    delta=(bootstrap[:,2]-bootstrap[:,1])/5000
    interval=np.quantile(delta,[.025,.975]).tolist()
    checked_documents=['EVALUATE_QFORMER_TEST.md','QFORMER_OBJECT_ALIGNMENT.md','QFORMER_CASCADE_ALIGNMENT.md',
        'QFORMER_FIXED_RERANKING_ABLATION.md','EVALUATE_CASCADE_CHAIR.md']
    summary={'variant':'caption_progress_reconciliation_20261015','registry_sha256':sha_bytes(Path(args.registry).read_bytes()),
        'archive_sha256':sha_bytes(Path(args.archive).read_bytes()),'archive_member_sha256':{n:sha_bytes(b) for n,b in raw.items()},
        'document_sha256':{n:sha_bytes(Path(n).read_bytes()) for n in checked_documents},
        'paired_chair':{'baseline':qm,'e2_v1':em,'baseline_counts':qcounts,'e2_counts':ecounts,'transitions':counts,
            'delta_e2_minus_baseline':{k:em[k]-qm[k] for k in qm},'bootstrap_images':5000,'bootstrap_replicates':20000,
            'bootstrap_seed':42,'chair_s_delta_95_percentile_interval':interval,
            'interpretation':'Numerical improvement; paired image-bootstrap interval includes zero. Reaggregated existing CHAIR outputs; not a fresh scorer run.'},
        'historical_deltas':{
            'e2_v1_vs_qformer_test_cider':historical['M19'][10]-historical['M12'][10],
            'fixed_reranking_vs_qformer_test_cider':historical['M13'][10]-historical['M12'][10],
            'e2_v21_vs_qformer_test_cider':historical['M24'][10]-historical['M12'][10],
            'e2_v1_vs_continuation_validation_cider':historical['M17'][10]-historical['M18'][10],
            'e3_vs_e2_v21_validation_cider':historical['M25'][10]-historical['M23'][10]},
        'evidence_status':{'M12_M19_CHAIR':'same5000 IDs/captions/GT objects and aggregate arithmetic verified from uploaded artifact',
            'caption_metrics':'historical registry/documents; not recomputed without original reference/candidate bundles',
            'M13_reranking':'historical registry and implementation report agree; raw reranked captions/candidates not present in inspected local artifacts',
            'M24_V21_M25_E3':'registry and user-provided logs; no new inference'},
        'new_caption_model_evaluated':False,'correction':'The earlier blanket statement that nothing beat the baseline was incorrect. Historical E2 and fixed reranking have numerical gains; recent relation prototypes have not surpassed the best historical pipeline.'}
    target=output/'caption_progress_audit.json';target.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',required=True);parser.add_argument('--registry',default='audit/caption_metrics_registry_20261015.json');parser.add_argument('--output',required=True)
    run(parser.parse_args())
