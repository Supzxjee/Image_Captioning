"""Score saved full-VALID captions through existing evaluators, no model inference."""
import argparse
from pathlib import Path
from real_bank_data import read,save
import score_full


def score(cfg):
    score_full.score(cfg)
    path=Path(cfg['output'])/'result.json';r=read(path)
    if r['method']!='csa_structure_patch_prompt_grouped_per_caption_accelerate_v1':raise ValueError('Wrong fulltraining method')
    r.update(status='CSA_PER_CAPTION_FULL_XE10_COMPLETE',framework='Accelerate1.15.0',precision='FP32',feature_sharing=False,sampler='image_grouped_42_plus_epoch')
    r['limits']+=['Restored CSA structure with CLIP substitutions and added UPT prompts, not exact originalCSA replication.',
                  'Grouped batches change sampling versus random-caption baseline.']
    save(path,r)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('probe','score'));p.add_argument('--config',required=True)
    a=p.parse_args();(score_full.probe if a.mode=='probe' else score)(read(a.config))
