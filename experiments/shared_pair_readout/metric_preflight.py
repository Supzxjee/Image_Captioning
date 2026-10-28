"""User-run CPU metric dependencies before spending time on model training."""
import math
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from captioning.object_reward import recognized_objects


def main():
    ptb=PTBTokenizer()
    refs=ptb.tokenize({0:[dict(caption='a dog is on a chair')],1:[dict(caption='a bicycle is beside a bus')]})
    hyp=ptb.tokenize({0:[dict(caption='a dog is on a chair')],1:[dict(caption='a bicycle is beside a bus')]})
    for scorer in (Cider(),Bleu(4),Meteor(),Rouge()):
        value,_=scorer.compute_score(refs,hyp)
        if isinstance(value,(list,tuple)):assert all(math.isfinite(float(v)) for v in value)
        else:assert math.isfinite(float(value))
    assert 'dog' in recognized_objects('a dog is on a chair','aliases_v2')
    print('CPU metric Java/PTB/CIDEr/BLEU/METEOR/ROUGE/object-parser preflight PASS',flush=True)


if __name__=='__main__':main()
