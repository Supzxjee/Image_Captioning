"""Pure saved-score operations; no model or evaluator imports."""
import hashlib
import math

ORIGINAL=(.4,.3)  # CLIP, object; decoder = 1-C-O

def split_ids(ids):
    if len(ids)!=5000 or len(set(ids))!=5000:raise ValueError('Expected 5000 unique VALID IDs')
    ordered=sorted(ids,key=lambda i:(hashlib.sha256(f'valid-weight-v1/42/{i}'.encode()).hexdigest(),i))
    return dict(TUNE=ordered[:2000],CHECK=ordered[2000:])

def grid():return [(c/10,o/10) for c in range(11) for o in range(11-c)]

def choose(row,weights):
    c,o=weights;d=1.0-c-o
    if min(c,o)<0 or c+o>1:raise ValueError('Invalid weights')
    candidates=row['candidates']
    if len(candidates)!=5:raise ValueError('Expected five candidates')
    scores=[d*x['decoder_score']+c*x['clip_score_normalized']+o*x['object_score'] for x in candidates]
    if not all(math.isfinite(s) for s in scores):raise ValueError('Nonfinite candidate score')
    return max(range(5),key=lambda r:(scores[r],-r))

def chair_counts(sentence):
    generated=sentence['mscoco_generated_words'];gt=set(sentence['mscoco_gt_words'])
    h=len(sentence['mscoco_hallucinated_words'])
    return [int(h>0),h,len(generated),len(set(generated)&gt),len(gt)]

def metrics(rows,cider,chair,ids,weights):
    counts=[0]*5;total=0
    for i in ids:
        rank=choose(rows[i],weights);total+=cider[i][rank]
        for j,v in enumerate(chair[i][rank]):counts[j]+=v
    if not ids or counts[2]<=0 or counts[4]<=0:raise ValueError('Empty metric denominator')
    return dict(CIDEr=total/len(ids),CHAIRs=counts[0]/len(ids),CHAIRi=counts[1]/counts[2],
                Recall=counts[3]/counts[4],counts=counts,images=len(ids))

def guards(candidate,baseline):
    return all(candidate[a]['CIDEr']>=baseline[a]['CIDEr']-1e-12 and
               candidate[a]['CHAIRs']<=baseline[a]['CHAIRs']+1e-12 and
               candidate[a]['CHAIRi']<=baseline[a]['CHAIRi']+1e-12 and
               candidate[a]['Recall']>=baseline[a]['Recall']-1e-12 for a in baseline)

def mean_cider(results):return sum(v['CIDEr'] for v in results.values())/len(results)

def fit(tune_scores):
    """Accept only TUNE aggregates. No CHECK data is passed to this function."""
    baseline=next(x['arms'] for x in tune_scores if tuple(x['weights'])==ORIGINAL)
    eligible=[x for x in tune_scores if guards(x['arms'],baseline)]
    best=max(mean_cider(x['arms']) for x in eligible)
    ties=[x for x in eligible if mean_cider(x['arms'])>=best-1e-12]
    return min(ties,key=lambda x:(sum(abs(a-b) for a,b in zip(x['weights'],ORIGINAL)),*x['weights']))

def accept(check,baseline):return guards(check,baseline) and mean_cider(check)>mean_cider(baseline)+1e-12
