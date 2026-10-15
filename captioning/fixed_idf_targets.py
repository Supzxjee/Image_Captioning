"""Reference gain targets with IDF frozen from fit images only.

Published evaluation still uses the normal scorer on its own evaluation corpus.
"""
from collections import Counter
import numpy as np


def fixed_scores(references,hypotheses,df,reference_count):
    from pycocoevalcap.cider.cider_scorer import CiderScorer
    if references.keys()!=hypotheses.keys() or len(references)>reference_count:
        raise ValueError('Fixed-IDF scorer count or IDs differ.')
    scorer=CiderScorer()
    for i in references:scorer+=(hypotheses[i][0],references[i])
    actual=len(references)
    # The upstream normalizer uses len(crefs). Empty internal pairs fix that
    # count to the fit corpus without adding anything to the frozen DF table.
    # They are not calibration images or labels; discard their scores below.
    for _ in range(reference_count-actual):scorer+=('', [''])
    scorer.document_frequency.update(df)
    return np.asarray(scorer.compute_cider()[:actual])


def targets(records,gt,fit_idf=None):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider_scorer import precook
    refs={r['image_id']:[] for r in records}
    for a in gt['annotations']:refs[a['image_id']].append({'caption':a['caption']})
    if not all(len(v)==5 for v in refs.values()):raise ValueError('Need5 references per image.')
    token_refs=PTBTokenizer().tokenize(refs)
    tokens=PTBTokenizer().tokenize({r['image_id']:[{'caption':c['caption']} for c in r['candidates']] for r in records})
    if fit_idf is None:
        df=Counter()
        for sentences in token_refs.values():df.update(set(g for sentence in sentences for g in precook(sentence)))
        fit_idf={'fit_reference_images':len(records),'document_frequency':[[list(g),n] for g,n in sorted(df.items())]}
    df={tuple(g):n for g,n in fit_idf['document_frequency']}
    result=np.zeros((len(records),5))
    for rank in range(5):
        pred={r['image_id']:[tokens[r['image_id']][min(rank,len(r['candidates'])-1)]] for r in records}
        result[:,rank]=fixed_scores(token_refs,pred,df,fit_idf['fit_reference_images'])
    return result,fit_idf
