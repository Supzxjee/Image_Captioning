"""Natural target substitutions from cached candidates, not labelled counterfactual truth."""
import argparse,json,itertools,statistics
from collections import defaultdict,Counter
from pathlib import Path
from captioning.supervised_relations import RELATIONS,contrast_edges
from evaluate_frozen_residual import sha


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union


def describe(values):
    return {'count':len(values),'mean':statistics.mean(values) if values else None,
            'median':statistics.median(values) if values else None}


def supervision(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row['image_id'],row['subject_id'])].append(row)
    counts=Counter();examples=[]
    for (image,subject),items in groups.items():
        for c,relation in enumerate(RELATIONS):
            positive=[r for r in items if r['labels'][c]==1]
            for anchor in positive:
                for alternate in items:
                    if alternate['object_id']==anchor['object_id']:continue
                    status=alternate['labels'][c]
                    counts[str(status)]+=1
                    if status==-1 and len(examples)<12:
                        examples.append({'image_id':image,'subject_id':subject,'relation':relation,
                            'positive_target':anchor['object_id'],'alternate_target':alternate['object_id'],'alternate_label':-1})
    return {'positive_anchor_alternate_target_label_counts':dict(counts),
            'confirmed_positive_to_zero_target_contrasts':counts['0'],'unknown_examples':examples,
            'existing_contrast_types':dict(Counter(e[-1] for e in contrast_edges(rows,max_per_positive=len(rows)))),
            'notes':'Same image,subject,predicate, differenttarget. Unknown alternatives remain unknown, not invented negatives.'}


def main():
    parser=argparse.ArgumentParser()
    for name in ('region-directory','triplet-directory','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();root=Path(args.region_directory);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    rows=json.loads(Path('audit/interaction_region_full100_pairs.json').read_text())
    models={};provenance={}
    for name in ('union_only','regions'):
        directory=root/name;summary=json.loads((directory/'region_summary.json').read_text())
        predictions=json.loads((directory/'candidate_predictions.json').read_text())
        if sha(directory/'region_head.pt')!=summary['head_sha256']:raise ValueError('Region head changed.')
        if len(predictions)!=len(rows) or any(any(row[k]!=p[k] for k in row) for row,p in zip(rows,predictions)):
            raise ValueError('Cached candidate identity/order changed.')
        models[name]=([p['scores'] for p in predictions],summary['thresholds'])
        provenance[name]={'head_sha256':summary['head_sha256'],'prediction_sha256':sha(directory/'candidate_predictions.json')}
    directory=Path(args.triplet_directory);summary=json.loads((directory/'triplet_summary.json').read_text())
    predictions=json.loads((directory/'candidate_predictions.json').read_text())
    if len(predictions)!=len(rows) or any(any(row[k]!=p[k] for k in row) for row,p in zip(rows,predictions)):
        raise ValueError('Triplet candidate identity/order changed.')
    models['old_triplet']=([p['triplet_activation'] for p in predictions],[summary['thresholds'][r] for r in RELATIONS])
    provenance['old_triplet']={'head_sha256':sha(directory/'triplet_head.pt'),'prediction_sha256':sha(directory/'candidate_predictions.json')}
    if provenance['old_triplet']['head_sha256']!='de8124a30555b39eabdf45cdf649d599788174361c94bd8036ebed714f61fab4':
        raise ValueError('Preserved triplet head changed.')
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[(row['image_id'],row['subject_id'],row['object'])].append(i)
    groups={key:indices for key,indices in groups.items() if len({rows[i]['object_id'] for i in indices})>=2}
    pairs=[(i,j) for indices in groups.values() for i,j in itertools.combinations(indices,2)
           if rows[i]['object_id']!=rows[j]['object_id']]
    separated=[(i,j) for i,j in pairs if iou(rows[i]['object_box'],rows[j]['object_box'])<.8]
    result={'variant':'natural_target_instance_sensitivity_v1','training_performed':False,'threshold_refit':False,
        'models':{},'source_provenance':provenance,'same_class_groups':len(groups),'same_class_pair_comparisons':len(pairs),
        'low_overlap_pair_comparisons':len(separated),'geometry_overlap_cutoff':.8,
        'supervision':{split:supervision(json.loads(Path(f'audit/{filename}.json').read_text())) for split,filename in
            (('full_gt_train','triplet_detector_train_reference_pairs'),('matched_detector_train','interaction_region_train_pairs'))},
        'notes':'Natural candidate substitutions, same image/subject/objectclass but target,crop andgeometry change together. '
                'Scores are sigmoid activations, not probabilities; scale differs across models. Sensitivity/multiple acceptance is not accuracy or proof of one true target. '
                'IoU<.8 subgroup excludes nearly duplicate boxes geometrically, not validated instances. Reuseddevelopment;no labels added or pipeline changed.'}
    for name,(scores,thresholds) in models.items():
        by_relation={}
        for c,relation in enumerate(RELATIONS):
            active=[(i,j) for i,j in separated if max(scores[i][c],scores[j][c])>=thresholds[c]]
            common_active=[(i,j) for i,j in separated if any(
                max(values[i][c],values[j][c])>=limits[c] for values,limits in models.values())]
            by_relation[relation]={'all_low_overlap_abs_score_difference':describe([abs(scores[i][c]-scores[j][c]) for i,j in separated]),
                'at_least_one_accepted_abs_difference':describe([abs(scores[i][c]-scores[j][c]) for i,j in active]),
                'same_common_active_pairs_abs_difference':describe([abs(scores[i][c]-scores[j][c]) for i,j in common_active]),
                'both_accepted':sum(scores[i][c]>=thresholds[c] and scores[j][c]>=thresholds[c] for i,j in active),
                'threshold_disagreement':sum((scores[i][c]>=thresholds[c])!=(scores[j][c]>=thresholds[c]) for i,j in active)}
        result['models'][name]=by_relation
    rank_comparisons={}
    for a,b in itertools.combinations(models,2):
        eligible=0;changed=0
        for indices in groups.values():
            for c in range(6):
                if not any(models[a][0][i][c]>=models[a][1][c] or models[b][0][i][c]>=models[b][1][c] for i in indices):continue
                # Ties break by the same stable index; not a claim of calibrated cross-model equivalence.
                eligible+=1
                changed+=max(indices,key=lambda i:(models[a][0][i][c],-i))!=max(indices,key=lambda i:(models[b][0][i][c],-i))
        rank_comparisons[a+' vs '+b]={'active_group_predicates':eligible,'different_top_target':changed}
    result['target_rank_comparisons']=rank_comparisons
    examples=[]
    for key,indices in groups.items():
        if key[0]!=303429:continue
        for i in indices:
            examples.append({**rows[i],'model_scores':{name:dict(zip(RELATIONS,scores[i])) for name,(scores,_) in models.items()}})
    result['pizza_target_case']=examples
    (output/'target_sensitivity_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('models','pizza_target_case')},indent=2),flush=True)
    for name,statistics_ in result['models'].items():print(name,'holding',json.dumps(statistics_['holding']),flush=True)


if __name__=='__main__':main()
