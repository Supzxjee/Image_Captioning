"""Instance matching and coverage-aware evaluation for a frozen external HOI model."""
from collections import Counter
import torch

VERB_NAMES=['hold','ride','sit_on','carry','eat','talk_on']


def node_key(label,box):return (label,tuple(box))


def box_iou(a,b):
    intersection=(torch.minimum(a[:,None,2:],b[None,:,2:])-torch.maximum(a[:,None,:2],b[None,:,:2])).clamp_min(0).prod(-1)
    area_a=(a[:,2:]-a[:,:2]).clamp_min(0).prod(-1)
    area_b=(b[:,2:]-b[:,:2]).clamp_min(0).prod(-1)
    return intersection/(area_a[:,None]+area_b[None,:]-intersection).clamp_min(1e-12)


def match_inventory(rows,prediction,object_names,minimum_iou=.5):
    from scipy.optimize import linear_sum_assignment
    keys=sorted({node_key(r[role],r[role+'_box']) for r in rows for role in ('subject','object')})
    if not len(prediction['boxes']):return {},{'source_nodes':len(keys),'matched':0}
    boxes=torch.tensor([k[1] for k in keys])
    height,width=prediction['size'].tolist()
    detected=prediction['boxes']/torch.tensor([width,height,width,height])
    iou=box_iou(boxes,detected)
    names=[object_names[int(i)].replace('_',' ') for i in prediction['labels']]
    same=torch.tensor([[k[0]==name for name in names] for k in keys])
    weights=iou.clone();weights[~same]=-2
    aa,bb=linear_sum_assignment(-weights.numpy())
    mapping={keys[int(i)]:{'index':int(j),'iou':float(iou[i,j])} for i,j in zip(aa,bb) if same[i,j] and iou[i,j]>=minimum_iou}
    return mapping,{'source_nodes':len(keys),'matched':len(mapping),'detected_nodes':len(detected)}


def target_case(case,rows,prediction,mapping,verbs,correspondence,clip_logits):
    i,j,c=case['positive_row_index'],case['alternate_row_index'],case['relation_index']
    positive,alternate=rows[i],rows[j]
    keys=[node_key(positive['subject'],positive['subject_box']),node_key(positive['object'],positive['object_box']),node_key(alternate['object'],alternate['object_box'])]
    result={'case_id':case['case_id'],'image_id':positive['image_id'],'relation_index':c,
        'same_class':positive['object']==alternate['object'],'clip_gap':float(clip_logits[i,c]-clip_logits[j,c]),
        'localized':False,'ontology_supported':False,'missing_nodes':[role for role,key in zip(('subject','positive','alternate'),keys) if key not in mapping]}
    if result['missing_nodes']:return result
    s,p,n=[mapping[key]['index'] for key in keys]
    lookup={(int(a),int(b)):k for k,(a,b) in enumerate(zip(prediction['subject_indices'],prediction['object_indices']))}
    if (s,p) not in lookup or (s,n) not in lookup:
        result['missing_pairs']=True;return result
    verb=verbs.index(VERB_NAMES[c]);pp,nn=lookup[s,p],lookup[s,n]
    classes=[int(prediction['labels'][t]) for t in (p,n)]
    supported=[verb in correspondence[t] for t in classes]
    raw=prediction['logits'][:,verb]
    confidence=prediction['scores'][s].pow(2.8)*prediction['scores'][[p,n]].pow(2.8)
    output=prediction['official_output']
    official={(int(a),int(b),int(v)):score for (a,b),v,score in zip(output['pairing'].T,output['labels'],output['scores'])}
    positive_score=official.get((s,p,verb),raw.new_tensor(0.))
    alternate_score=official.get((s,n,verb),raw.new_tensor(0.))
    result.update(localized=True,ontology_supported=all(supported),endpoint_ontology_supported=supported,
        matched_iou=[mapping[k]['iou'] for k in keys],raw_gap=float(raw[pp]-raw[nn]),
        official_gap=float(positive_score-alternate_score),
        detection_confidence_gap=float(confidence[0]-confidence[1]))
    return result


def aggregate_cases(items):
    covered=[r for r in items if r['localized'] and r['ontology_supported']]
    def compare(values,field):
        return {'cases':len(values),'images':len({r['image_id'] for r in values}),
            'wins':sum(r[field]>0 for r in values),'ties':sum(r[field]==0 for r in values),
            'win_rate':sum(r[field]>0 for r in values)/len(values) if values else None}
    comparisons={field:compare(covered,field) for field in ('clip_gap','raw_gap','official_gap','detection_confidence_gap')}
    same=[r for r in covered if r['same_class']]
    pairs={name:{'cases':sum((r['clip_gap']<=0 and r['raw_gap']>0) if name=='fixed' else (r['clip_gap']>0 and r['raw_gap']<=0) for r in covered),
        'images':len({r['image_id'] for r in covered if ((r['clip_gap']<=0 and r['raw_gap']>0) if name=='fixed' else (r['clip_gap']>0 and r['raw_gap']<=0))})} for name in ('fixed','regressed')}
    return {'total_cases':len(items),'total_images':len({r['image_id'] for r in items}),
        'localized_cases':sum(r['localized'] for r in items),'localized_and_ontology_supported':len(covered),
        'coverage':len(covered)/len(items) if items else None,
        'missing_node_roles':dict(Counter(role for r in items for role in r['missing_nodes'])),
        'localized_but_ontology_unsupported':sum(r['localized'] and not r['ontology_supported'] for r in items),
        'matched_comparisons':comparisons,'matched_same_class':{f:compare(same,f) for f in ('clip_gap','raw_gap','official_gap','detection_confidence_gap')},
        'raw_covered_and_correct_of_all':sum(r['raw_gap']>0 for r in covered),
        'raw_paired_changes_vs_clip':pairs,
        'go_to_larger_audit':len(covered)>=.8*len(items) and comparisons['raw_gap']['wins']>=comparisons['clip_gap']['wins']+3,
        'notes':'Unlocalized/ontology-unsupported cases stay unscored, not false. Same-subset CLIP comparison only. Go rule fixed before inference; not production promotion.'}


def exploratory_pairwise_fallback(items):
    """Post-hoc pairwise decision diagnostic, NOT a unified per-candidate score.

    Route without correctness labels. No local gap thresholds or hand-picked IDs.
    A production total ordering/acceptance calibration has not been implemented.
    """
    chosen=[{**r,'fallback_gap':r['raw_gap'] if r['localized'] and r['ontology_supported'] else r['clip_gap']} for r in items]
    def count(subset):
        return {'cases':len(subset),'clip_wins':sum(r['clip_gap']>0 for r in subset),
            'fallback_wins':sum(r['fallback_gap']>0 for r in subset),'ties':sum(r['fallback_gap']==0 for r in subset)}
    return {'all':count(chosen),'same_class':count([r for r in chosen if r['same_class']]),
        'teacher_routed':sum(r['localized'] and r['ontology_supported'] for r in chosen),
        'clip_fallback':sum(not(r['localized'] and r['ontology_supported']) for r in chosen),
        'status':'Exploratory rule added after observing teacher results; not independent validation, calibrated acceptance, unified scores or a caption improvement.'}
