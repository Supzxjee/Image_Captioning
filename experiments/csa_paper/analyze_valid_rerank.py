"""Saved VALID candidates/scores only; no model or official metric execution."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from captioning.multiscore_reranking import rerank_multiscore

HERE=Path(__file__).resolve().parent
CACHE=HERE.parents[2]/'.review_cache/prompt_scst_full1'
OUT=HERE/'valid_rerank_diagnostic'


def repeated_trigram(caption):
    tokens=re.findall(r'[a-z0-9]+',caption.lower())
    grams=[tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
    return len(grams)!=len(set(grams))


def main():
    OUT.mkdir(exist_ok=True)
    bundle=dict(schema='saved_VALID5000_candidates_v1',test_used=False,training=False,arms={})
    summary={}; checks=[]; examples=[]; refs=None
    def check(n,ok):
        if not ok:raise ValueError(n)
        checks.append(n)
    for arm,version in [('CONTROL',180),('PROMPT',60)]:
        cached=json.loads((CACHE/(arm+'.json')).read_text(encoding='utf-8'))
        files=cached['files'];rd=lambda n:json.loads(files[n])
        state=rd('progress.json');result=rd('result.json');prep=rd('rerank_prepared.json')
        check(arm+'/complete',rd('run_status.json')['complete'] and state['updates']==14161 and not result['test_used'])
        for n,h in state['validation_files'].items():check(arm+'/sha/'+n,hashlib.sha256(files[n].encode()).hexdigest()==h)
        check(arm+'/rerank-input-sha',hashlib.sha256(files['rerank_inputs.json'].encode()).hexdigest()==prep['input_sha256'])
        thisrefs=rd('validation_references.json');ids={r['coco_id'] for r in thisrefs}
        check(arm+'/refs',len(thisrefs)==len(ids)==5000 and sum(len(r['references']) for r in thisrefs)==25010)
        check(arm+'/reference-sha',hashlib.sha256(files['validation_references.json'].encode()).hexdigest()==prep['references_sha256'])
        if refs is not None:check('same refs',thisrefs==refs)
        refs=thisrefs
        candidates=rd('evaluations/end/candidates_rank0.json')+rd('evaluations/end/candidates_rank1.json')
        check(arm+'/candidate-ids',len(candidates)==5000 and {r['coco_id'] for r in candidates}==ids)
        scored={r['coco_id']:r for rank in (0,1) for r in map(json.loads,files[f'rerank_scores_rank{rank}.jsonl'].splitlines())}
        raw={r['coco_id']:r['CIDEr'] for r in rd('evaluations/end/cider_per_image.json')}
        rerank={r['coco_id']:r['CIDEr'] for r in rd('reranked/end/cider_per_image.json')}
        inputmap={r['coco_id']:r for r in rd('rerank_inputs.json')}
        ch0={r['image_id']:r for r in rd('evaluations/end/chair_output.json')['sentences']}
        ch1={r['image_id']:r for r in rd('reranked/end/chair_output.json')['sentences']}
        check(arm+'/coverage',all(set(m)==ids for m in [scored,raw,rerank,inputmap,ch0,ch1]))
        records=[]
        for row in sorted(candidates,key=lambda r:r['coco_id']):
            i=row['coco_id'];cs=row['candidates'];clip=scored[i]['phases']['end']['clipscore']
            check(arm+f'/{i}/ranks',len(cs)==5 and [c['rank'] for c in cs]==list(range(5)))
            selected,enriched=rerank_multiscore(dict(candidates=[dict(c,clipscore=s) for c,s in zip(cs,clip)]),inputmap[i]['detected'],.4,.3,1.)
            k=selected['original_rank']
            check(arm+f'/{i}/selector',k==scored[i]['phases']['end']['selected_rank'] and cs[k]['caption']==ch1[i]['caption'] and cs[0]['caption']==ch0[i]['caption'])
            for c in enriched:
                c['D_contribution']=.3*c['decoder_score'];c['C_contribution']=.4*c['clip_score_normalized'];c['O_contribution']=.3*c['object_score']
                c['repeated_trigram']=repeated_trigram(c['caption'])
                c.pop('ids',None)
            rec=dict(coco_id=i,selected_rank=k,candidates=enriched,detected=inputmap[i]['detected'],
                image_path=inputmap[i]['image_path'],image_sha256=inputmap[i]['image_sha256'],
                rank0_CIDEr=raw[i],selected_CIDEr=rerank[i],delta_CIDEr=rerank[i]-raw[i],
                rank0_CHAIR=bool(ch0[i]['mscoco_hallucinated_words']),selected_CHAIR=bool(ch1[i]['mscoco_hallucinated_words']))
            records.append(rec)
        eps=1e-10
        flat=lambda key:[r['candidates'][r['selected_rank']][key] for r in records]
        gains=[r['delta_CIDEr'] for r in records]
        check(arm+'/mean-rank0',abs(sum(raw.values())/5000-result['metrics']['end']['decoder_rank0']['CIDEr'])<1e-10)
        check(arm+'/mean-selected',abs(sum(rerank.values())/5000-result['metrics']['end']['fixed_rerank']['CIDEr'])<1e-10)
        summary[arm]=dict(source_version=version,images=5000,
            rank0_CIDEr_x100=sum(raw.values())/50,selected_CIDEr_x100=sum(rerank.values())/50,
            changed_rank=sum(r['selected_rank']!=0 for r in records),selected_rank_counts=dict(Counter(r['selected_rank'] for r in records)),
            cider_improved=sum(d>eps for d in gains),cider_worsened=sum(d < -eps for d in gains),cider_tied=sum(abs(d)<=eps for d in gains),
            positive_CIDEr_contribution_x100=sum(max(0,d) for d in gains)/50,
            negative_CIDEr_contribution_x100=sum(min(0,d) for d in gains)/50,
            rank0_selected_oracle_x100=sum(max(r['rank0_CIDEr'],r['selected_CIDEr']) for r in records)/50,
            chair_flags_removed=sum(r['rank0_CHAIR'] and not r['selected_CHAIR'] for r in records),
            chair_flags_introduced=sum(not r['rank0_CHAIR'] and r['selected_CHAIR'] for r in records),
            identical_five=sum(len({c['caption'] for c in r['candidates']})==1 for r in records),
            mean_unique_captions=sum(len({c['caption'] for c in r['candidates']}) for r in records)/5000,
            any_repeated_trigram=sum(any(c['repeated_trigram'] for c in r['candidates']) for r in records),
            selected_repeated_trigram=sum(flat('repeated_trigram')),
            decoder_rank0_repeated_trigram=sum(r['candidates'][0]['repeated_trigram'] for r in records),
            selected_repeats_clean_alternative=sum(r['candidates'][r['selected_rank']]['repeated_trigram'] and any(not c['repeated_trigram'] for c in r['candidates']) for r in records),
            object_score_all_equal=sum(len({c['object_score'] for c in r['candidates']})==1 for r in records),
            clip_spread_below_001=sum(max(c['clipscore'] for c in r['candidates'])-min(c['clipscore'] for c in r['candidates'])<.01 for r in records))
        pools={
            'selected_lower_CIDEr':lambda r:r['delta_CIDEr']<-.5,
            'introduced_CHAIR':lambda r:not r['rank0_CHAIR'] and r['selected_CHAIR'],
            'selected_repeat_with_clean_alternative':lambda r:r['candidates'][r['selected_rank']]['repeated_trigram'] and any(not c['repeated_trigram'] for c in r['candidates'])}
        for group,predicate in pools.items():
            eligible=[r for r in records if predicate(r)]
            picked=sorted(eligible,key=lambda r:hashlib.sha256(f'valid-rerank-v1/{arm}/{group}/{r["coco_id"]}'.encode()).hexdigest())[:2]
            for r in picked:examples.append(dict(arm=arm,group=group,**r))
        bundle['arms'][arm]=dict(source_version=version,archive_sha256=cached['archive_sha256'],references=thisrefs,records=records)
    raw=json.dumps(bundle,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    # Deterministic gzip text package: no images, checkpoints, tokens or secrets.
    payload=gzip.compress(raw,mtime=0)
    (OUT/'VALID_CANDIDATES.json.gz').write_bytes(payload)
    saved=dict(status='STORED_VALID_RERANK_DIAGNOSTIC_COMPLETE',checks=len(checks),test_used=False,summary=summary,
        bundle_sha256=hashlib.sha256(payload).hexdigest(),bundle_bytes=len(payload),
        limits=['CIDEr and CHAIR values reused for rank0 and selected only. All-five oracle not yet measured.',
                'Repeated trigram is a diagnostic heuristic, not grammatical/visual truth.',
                'Rank0/selected oracle uses references and is not deployable.',
                'No selector weights changed, no official scorer/model run, no user report edited.'])
    (OUT/'RESULT.json').write_text(json.dumps(saved,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'review_candidates.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(saved,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
