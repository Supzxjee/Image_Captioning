"""USER-RUN official metric replay on saved texts; no model inference."""
import argparse
import gzip
import json
from pathlib import Path
from real_bank_data import HERE,read,save,sha
from saved_quality_data import check_rows


def run(output):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    baseline=read(output/'result.json')
    if baseline['status']!='PLAIN10_KARPATHY_TEST_COMPLETE':raise ValueError('Incomplete baselineTEST')
    package=read(HERE/'CSA10_TEST_TEXT_PACKAGE.json');bundle=HERE/'csa10_test_saved_text.json.gz'
    if sha(bundle)!=package['bundle_sha256']:raise ValueError('Changed CSA TEST savedtext')
    csa=json.loads(gzip.decompress(bundle.read_bytes()))
    if csa['source_review_sha256']!=package['source_review_sha256']:raise ValueError('Wrong CSA TEST provenance')
    rows=read(output/'test_rank0.json')+read(output/'test_rank1.json');metadata=read(output/'test_metadata.json');ids={r['coco_id'] for r in metadata}
    for data in (rows,csa['rows']):check_rows(data,ids)
    ptb=PTBTokenizer();ref=ptb.tokenize({r['coco_id']:[dict(caption=s) for s in r['captions']] for r in metadata})
    expected=read(HERE/'csa10_test_review_text/result.json')['metrics']
    replay={}
    for name,data,target in (('plain',rows,baseline['metrics']),('csa',csa['rows'],expected)):
        hyp=ptb.tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in data})
        cider,_=Cider().compute_score(ref,hyp);bleu,_=Bleu(4).compute_score(ref,hyp);rouge,_=Rouge().compute_score(ref,hyp)
        actual=dict(CIDEr=float(cider),Bleu_4=float(bleu[3]),ROUGE_L=float(rouge));replay[name]=actual
        save(output/'metric_replay_partial.json',replay)
        if any(abs(v-target[k])>1e-7 for k,v in actual.items()):raise ValueError('Saved metric replay differs: '+name)
    if baseline['metrics']['counts']['reference_unique']!=expected['counts']['reference_unique']:raise ValueError('CHAIR reference coverage differs')
    flags={str(r['image_id']):bool(r['mscoco_hallucinated_words']) for r in read(output/'test_chair_output.json')['sentences']}
    old=csa['hallucination']
    paired=dict(plain_to_csa_repaired=sum(flags[str(i)] and not old[str(i)] for i in ids),plain_to_csa_introduced=sum(not flags[str(i)] and old[str(i)] for i in ids))
    save(output/'comparison.json',dict(status='PLAIN10_VS_CSA10_TEST_COMPLETE',images=5000,epoch=10,beam=5,max_new_tokens=20,
        plain=baseline['metrics'],csa=expected,csa_minus_plain={k:expected[k]-baseline['metrics'][k] for k in ('CIDEr','Bleu_4','ROUGE_L','CHAIRs','CHAIRi','Recall','NLL')},
        paired_hallucination=paired,metric_replay=replay,selection='exact10 prelocked',new_training=False,
        limitations=['One seed; encoder parameter capacity differs.','CHAIR assesses objects, not general relational truth.','Inherited CSA versus plain is not a novel alignment method.']))
    print('PLAIN10 vs CSA10 TEST COMPARISON COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
