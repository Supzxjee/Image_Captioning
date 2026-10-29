"""Analyze mixed AI/human panel ratings, retaining annotation provenance."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from make_panel import HERE,OUT,assemble,digest

ARMS=('A','I','J')

def bounds(rows,field,value):
    definite=sum(r[field]==value for r in rows); unknown=sum(r[field]=='uncertain' for r in rows)
    return [definite/len(rows),(definite+unknown)/len(rows)]

def aggregate(rows):
    positive=[r for r in rows if r['role']=='positive'];negative=[r for r in rows if r['role']=='negative']
    rc=sum(r['q_claim']=='yes' for r in positive);fr=sum(r['q_claim']=='yes' for r in negative)
    covered=[r for r in rows if r['target_relation']=='yes']
    correctness=Counter(r['relation_correctness'] for r in covered)
    rb=bounds(positive,'q_claim','yes');fb=bounds(negative,'q_claim','yes')
    return {'images':len(rows),'positive_images':len(positive),'negative_images':len(negative),
      'asserted_on_positive':rc,'asserted_on_negative':fr,'RC':rc/len(positive),'FR':fr/len(negative),'Y':rc/len(positive)-fr/len(negative),
      'q_uncertain_positive':sum(r['q_claim']=='uncertain' for r in positive),'q_uncertain_negative':sum(r['q_claim']=='uncertain' for r in negative),
      'RC_uncertainty_bounds':rb,'FR_uncertainty_bounds':fb,'Y_uncertainty_bounds':[rb[0]-fb[1],rb[1]-fb[0]],
      'target_relation_covered':len(covered),'coverage':len(covered)/len(rows),'coverage_uncertainty_bounds':bounds(rows,'target_relation','yes'),
      'correct':correctness['correct'],'wrong':correctness['wrong'],'uncertain_when_covered':correctness['uncertain'],
      'target_relation_uncertain':sum(r['target_relation']=='uncertain' for r in rows),'not_mentioned':sum(r['target_relation']=='no' for r in rows),
      'correct_fraction_when_covered':correctness['correct']/len(covered) if covered else None,
      'confirmed_human_arm_captions':sum(r['confirmed_by_human'] for r in rows)}

def paired_vectors(rows,pairs):
    lookup={(r['pair_id'],r['role']):r for r in rows}
    return np.array([[float(lookup[(p,'positive')]['q_claim']=='yes'),float(lookup[(p,'negative')]['q_claim']=='yes'),
      sum(lookup[(p,role)]['target_relation']=='yes' for role in ('positive','negative'))/2,
      sum(lookup[(p,role)]['target_relation']=='yes' and lookup[(p,role)]['relation_correctness']=='correct' for role in ('positive','negative'))/2,
      sum(lookup[(p,role)]['target_relation']=='yes' and lookup[(p,role)]['relation_correctness']=='wrong' for role in ('positive','negative'))/2] for p in pairs])

def main():
    public,key,_=assemble();cached=json.loads((OUT/'imported_user_review.json').read_text(encoding='utf8'));review=cached['review']
    assert review['panel_sha256']==public['panel_sha256']
    proposal_raw=(HERE/'ASSISTANT_PANEL_REVIEW.json').read_bytes();proposal=json.loads(proposal_raw)
    assert review['assistant_proposal_sha256']==digest(proposal_raw)
    ann=json.loads(json.dumps(review['annotations']))
    assert set(ann)=={u['id'] for c in public['cards'] for u in c['captions']}
    adjudications=[]
    # User chat clarification on 2026-10-29: keep target relation uncertain for both.
    for index in (6,49):
        uid=public['cards'][index-1]['captions'][0]['id'];r=ann[uid]
        assert r['target_relation']=='uncertain' and r['confirmed_by_human'] is True
        assert r['relation_correctness']=={6:'correct',49:'wrong'}[index]
        adjudications.append({'image':index,'caption':'X','id':uid,'original_relation_correctness':r['relation_correctness'],
          'analysis_relation_correctness':'uncertain','reason':'User explicitly retained uncertain target-pair identity in chat; correct/wrong is not scored as definite for unidentified target. Original uploaded JSON retained unchanged.'})
        r['relation_correctness']='uncertain'
    allowed={'q_claim':{'yes','no','uncertain'},'target_relation':{'yes','no','uncertain'},'relation_correctness':{'correct','wrong','uncertain','not_applicable'}}
    for r in ann.values():
        assert all(r[k] in v for k,v in allowed.items())
        assert isinstance(r['confirmed_by_human'],bool)
        assert not(r['q_claim']=='yes' and r['target_relation']!='yes')
        assert not(r['target_relation']=='no' and r['relation_correctness']!='not_applicable')
        assert not(r['target_relation']=='yes' and r['relation_correctness']=='not_applicable')
        assert not(r['target_relation']=='uncertain' and r['relation_correctness']!='uncertain')
    expanded={arm:[dict(k,**ann[k['id']]) for k in key if arm in k['arms']] for arm in ARMS}
    assert all(len(v)==64 and len({(r['pair_id'],r['role']) for r in v})==64 for v in expanded.values())
    metrics={arm:aggregate(rows) for arm,rows in expanded.items()}
    strata={p:next(k['stratum'] for k in key if k['pair_id']==p) for p in {k['pair_id'] for k in key}}
    pairs=sorted(strata);groups=[np.array([i for i,p in enumerate(pairs) if strata[p]==s]) for s in sorted(set(strata.values()))]
    assert [len(g) for g in groups]==[16,16]
    rng=np.random.default_rng(20261029);draws=np.concatenate([rng.choice(g,(10000,len(g))) for g in groups],axis=1)
    vectors={a:paired_vectors(r,pairs) for a,r in expanded.items()};contrasts={}
    names=['RC','FR','coverage','correct_per_image','wrong_per_image']
    for left,right in [('J','A'),('J','I')]:
        delta=vectors[left]-vectors[right];boot=delta[draws].mean(1);point=delta.mean(0)
        contrasts[left+'_minus_'+right]={n:{'difference':float(point[i]),'paired95_ci':np.quantile(boot[:,i],[.025,.975]).tolist()} for i,n in enumerate(names)}
        contrasts[left+'_minus_'+right]['Y']={'difference':float(point[0]-point[1]),'paired95_ci':np.quantile(boot[:,0]-boot[:,1],[.025,.975]).tolist()}
        lb=metrics[left]['Y_uncertainty_bounds'];rb=metrics[right]['Y_uncertainty_bounds']
        contrasts[left+'_minus_'+right]['Y']['conservative_annotation_bounds']=[lb[0]-rb[1],lb[1]-rb[0]]
    confirmed=[uid for uid,r in ann.items() if r['confirmed_by_human']]
    updated=[uid for uid,r in review['annotations'].items() if any(r[k]!=proposal['annotations'][uid][k] for k in allowed)]
    result={'status':'MIXED_ANNOTATION_SCREEN_NO_J_SEMANTIC_GAIN_OVER_A','date':'2026-10-29','panel_sha256':public['panel_sha256'],
      'source_review_sha256':cached['source_sha256'],'annotation_provenance':'AI-proposed; human reviewed selected priority cases, not independent human gold',
      'unique_ratings':175,'human_confirmed_unique':len(confirmed),'AI_only_unique':175-len(confirmed),'human_changed_labels_unique':len(updated),
      'adjudications':adjudications,'metrics':metrics,'by_stratum':{s:{a:aggregate([r for r in rows if r['stratum']==s]) for a,rows in expanded.items()} for s in sorted(set(strata.values()))},
      'paired_contrasts':contrasts,'bootstrap':{'draws':10000,'seed':20261029,'unit':'32 complete pairs; stratified resampling16 per stratum','unknown_q_point_rule':'Only definite yes counts; unknowns retained in bounds','limits':'Conditional on mixed labels, not annotation reliability, seed robustness or generalization. Bounds conservative, may allow inconsistent assignments to shared unknowns.'},
      'recommendation':'Keep A; stop current I/J recipe. This panel gives no reason to run conditional J_bypass diagnostic or extend training.',
      'limits':['158 of175 unique ratings remain AI-only; 17 human confirmations are targeted, not a random agreement sample.',
                'Source VSR labels define positive/negative strata; pair labels were not re-audited in this caption panel.',
                '64 images/32 reused pairs, one training seed. No independent test or broad causal/novelty claim.',
                'RC/FR measure target assertion expression, not full-caption accuracy; wrong nouns/other relations can fall outside the target pair.']}
    assert len(confirmed)==17 and len(updated)==3
    (HERE/'SEMANTIC_PANEL_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    lines=['# Semantic caption panel: mixed AI/human screening','',
      '2026-10-29. All32 frozen pairs/64 images retained. 175 unique ratings cover192 arm-captions. 17 unique ratings confirmed by the user;158 remain AI-only. This is not an independent human evaluation. Three caption-label records changed relative to the proposal; two unresolved target-identity cases remain uncertain after explicit chat clarification.','',
      '| Arm | Target assertion on positive /32 (RC) | On negative /32 (FR) | Y=RC-FR | Target relation mentioned /64 | Correct / wrong among mentions | Unknown target /64 |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,m in metrics.items():lines.append(f"| {arm} | {m['asserted_on_positive']}/32 ({m['RC']:.2%}) | {m['asserted_on_negative']}/32 ({m['FR']:.2%}) | {m['Y']:.2%} | {m['target_relation_covered']}/64 | {m['correct']} / {m['wrong']} | {m['target_relation_uncertain']} |")
    lines+=['','J expresses the true target assertion on fewer positive images than A (13 versus15), without reducing definite false assertions (1 each). Definite target-relation coverage is equal (28/64), while correct/wrong counts do not favor J (25/3 versus26/2). This is not a semantic gain over A.','',
      'Unknown q claims: A1 positive+1 negative; I0+0; J0+1. Conservative Y bounds:']
    for a,m in metrics.items():lines.append(f"- {a}: [{m['Y_uncertainty_bounds'][0]:.2%}, {m['Y_uncertainty_bounds'][1]:.2%}].")
    lines+=['','## Paired effects, percentage points','', '| Contrast | Metric | Difference | Pair bootstrap95% interval |','|---|---|---:|---|']
    for contrast,values in contrasts.items():
        for metric,v in values.items():lines.append(f"| {contrast} | {metric} | {100*v['difference']:+.2f} | [{100*v['paired95_ci'][0]:+.2f}, {100*v['paired95_ci'][1]:+.2f}] |")
    lines+=['','Intervals use10000 stratified whole-pair bootstrap draws, seed20261029. They are conditional on the mixed annotations and do not establish annotation accuracy or population significance. Unknown claims are treated as not-definite-yes in point estimates, with separate bounds. All strata and denominators are retained in the JSON.','',
      '## Decision','', 'Keep A. Consistent with Astra\'s post-result rule, stop the current I/J configuration rather than automatically running seed2, full training or J_bypass. This screening result does not rule out the general shared-readout idea; it does not justify more GPU on this recipe. A different experiment needs a separately stated hypothesis.','',
      '## Human clarification','', '#6X and#49X retain uncertain target identity at the user\'s explicit request. The original uploaded judgments correct/wrong are preserved in the input cache; for analysis their correctness remains unresolved, not definite. No user file was overwritten.','', '## Limits','']+['- '+x for x in result['limits']]
    (HERE/'SEMANTIC_PANEL_RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'status':result['status'],'human_confirmed':len(confirmed),'metrics':metrics,'contrasts':contrasts},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
