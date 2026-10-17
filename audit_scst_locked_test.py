"""Recompute locked test metrics from user review, no model inference/training."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from run_scst_object_locked_test import SELECTED_LEGACY_SHA,validate_selected


def audit(source,output):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None
        read=lambda n:json.loads(z.read(n))
        cfg,summary=read('protocol.json'),read('run_summary.json')
        assert cfg['code_commit']=='21846bc4602f581e9ed2d29424a80f750e5fe3e9'
        assert cfg['variant']=='scst_object_w05_locked_test5000' and cfg['evaluation_split']=='test'
        assert cfg['locked_before_test'] and not cfg['test_based_selection'] and not cfg['training']
        assert summary['status']=='LOCKED_TEST_COMPLETE' and summary['test_used'] and not summary['training']
        assert not summary['checkpoint_changed_after_test']
        assert summary['protocol_sha256']==hashlib.sha256(z.read('protocol.json')).hexdigest()
        identity=json.loads((Path(__file__).parent/'audit/hoi_validation_input_identity.json').read_text())['input_sha256']
        assert all(v==(SELECTED_LEGACY_SHA if k=='selected_checkpoint' else identity[k]) for k,v in cfg['input_sha256'].items())
        validate_selected(cfg['selected_metadata'])
        assert cfg['beam_size']==cfg['candidate_count']==5
        assert cfg['fixed_weights']=={'decoder':.3,'clip':.4,'object_consistency':.3}
        assert cfg['primary_policy']==summary['primary_policy']=='fixed'
        rows=cfg['groups']['test'];ids=[r['image_id'] for r in rows]
        assert len(rows)==len(set(ids))==len({r['coco_id'] for r in rows})==5000
        gt=read('test_gt.json');refs={i:[] for i in ids}
        for r in gt['annotations']:refs[r['image_id']].append({'caption':r['caption']})
        assert all(len(v)==5 for v in refs.values())
        tokens=PTBTokenizer().tokenize(refs)
        metrics={};per={};chairs={};captions={}
        for arm in ('original','scst_object'):
            prefix=arm+'/'
            beams,scored=read(prefix+'beams.json'),read(prefix+'clip_beams.json')
            assert beams['metadata']['split']=='test' and beams['metadata']['count']==5000
            data=beams['data'];assert [r['image_id'] for r in data]==ids==[r['image_id'] for r in scored['data']]
            assert all(len(r['candidates'])==5 for r in data)
            for a,b in zip(data,scored['data']):assert a['candidates']==[{k:v for k,v in c.items() if k!='clipscore'} for c in b['candidates']]
            worker=read(prefix+'beams_worker_config.json')
            expected=cfg['input_sha256']['checkpoint' if arm=='original' else 'selected_checkpoint']
            assert worker['checkpoint_sha256']==beams['metadata']['checkpoint_sha256']==expected
            assert worker['split']=='test' and worker['images']==rows
            for rank in range(2):
                assert [json.loads(line) for line in z.read(prefix+f'beams_{rank}.jsonl').decode().splitlines()]==data[rank::2]
            for policy in ('rank0','fixed'):
                name=arm+'_'+policy;pred=read(prefix+policy+'_predictions.json')
                assert [r['image_id'] for r in pred]==ids
                assert all(p['caption'] in {c['caption'] for c in r['candidates']} for p,r in zip(pred,data))
                if policy=='rank0':assert all(p['caption']==r['candidates'][0]['caption'] for p,r in zip(pred,data))
                text=PTBTokenizer().tokenize({r['image_id']:[{'caption':r['caption']}] for r in pred})
                cider,scores=Cider().compute_score(tokens,text);bleu,_=Bleu(4).compute_score(tokens,text);rouge,_=Rouge().compute_score(tokens,text)
                saved=read(prefix+policy+'_metrics.json');assert saved==summary['metrics'][name]['caption']
                for k,v in {'CIDEr':cider,'ROUGE_L':rouge,**{f'Bleu_{i+1}':b for i,b in enumerate(bleu)}}.items():assert abs(saved[k]-v)<1e-10
                ch=read(prefix+policy+'_chair_output.json');sentences=ch['sentences'];lookup={r['image_id']:r for r in sentences}
                assert len(sentences)==len(lookup)==5000 and set(lookup)=={r['coco_id'] for r in rows}
                assert all(lookup[r['coco_id']]['caption']==p['caption'] for r,p in zip(rows,pred))
                bad=sum(bool(r['mscoco_hallucinated_words']) for r in sentences);hall=sum(len(r['mscoco_hallucinated_words']) for r in sentences)
                mentions=sum(len(r['mscoco_generated_words']) for r in sentences)
                correct=sum(len(set(r['mscoco_generated_words'])&set(r['mscoco_gt_words'])) for r in sentences)
                total=sum(len(set(r['mscoco_gt_words'])) for r in sentences)
                agg={'CHAIRs':bad/5000,'CHAIRi':hall/mentions,'Recall':correct/total}
                assert agg==ch['overall_metrics']==summary['metrics'][name]['chair']
                metrics[name]={'caption':saved,'chair':agg,'hallucinated_captions':bad,'hallucinated_mentions':hall,
                               'object_mentions':mentions,'correct_unique_objects':correct,'reference_unique_objects':total}
                per[name]=np.asarray(scores);chairs[name]=lookup;captions[name]=pred
                print('Audited',name,flush=True)
        paired={}
        for policy in ('rank0','fixed'):
            a,b='original_'+policy,'scst_object_'+policy;delta=per[b]-per[a];rng=np.random.default_rng(20261017)
            ci=np.percentile([delta[rng.integers(0,5000,5000)].mean() for _ in range(5000)],[2.5,97.5])
            assert abs(float(delta.mean())-summary['deltas'][policy]['CIDEr'])<1e-10
            paired[policy]={'CIDEr_delta':float(delta.mean()),'conditional_bootstrap95':ci.tolist(),
                'hallucination_removed':sum(bool(chairs[a][i]['mscoco_hallucinated_words']) and not chairs[b][i]['mscoco_hallucinated_words'] for i in chairs[a]),
                'hallucination_introduced':sum(bool(chairs[b][i]['mscoco_hallucinated_words']) and not chairs[a][i]['mscoco_hallucinated_words'] for i in chairs[a]),
                'changed_captions':sum(x['caption']!=y['caption'] for x,y in zip(captions[a],captions[b]))}
        result={'status':'AUDIT_PASS','source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                'source_bytes':Path(source).stat().st_size,'metrics':metrics,'paired_comparisons':paired,
                'selected_checkpoint_sha256':SELECTED_LEGACY_SHA,'test_images':5000,'test_used':True,'training':False,
                'limits':['One seed;historical test exposure;no independent unseen-test claim.',
                         'CIDEr intervals conditional on checkpoint/imagepool,fixedIDF;not training-seed uncertainty.',
                         'Weights/fullsource/detections absent: no actual tensorforward or rerankformula replay.',
                         'METEOR cross-checked only;CHAIR reaggregated,notretokenized.',
                         'SCST+CE newtrained control not evaluated;no isolated objectreward contribution established.']}
        Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(paired,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.source,a.output)
