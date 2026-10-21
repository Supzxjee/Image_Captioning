"""One GV arm, using the existing G data/staging/worker/evaluator paths."""
import argparse
import json
from pathlib import Path
import run_region as region
from pilot_data import save

BASE_BUILD=region.build_model


def prepare(config):
    cfg=json.loads(Path(config).read_text())
    if cfg['arm']!='GV':raise ValueError('Expected one GV arm')
    temporary=Path(config).with_name('gv_as_g_preparation.json')
    save(temporary,dict(cfg,arm='G'))
    region.prepare(temporary)
    protocol=Path(cfg['output'])/'protocol.json'
    cfg=json.loads(protocol.read_text())
    cfg.update(arm='GV',checkpoint_format='vg_visual_only_caption_v1',
        auxiliary_gradient_policy='Visual projection/Q-Former: caption+.1phrase gradient; decoder: caption gradient only. Auxiliary uses current decoder parameters detached with functional_call; memory remains differentiable.',
        promotion_requires='GV beats matched A under locked numeric gates and blind review; causal attribution to gradient routing additionally requires reliable GV-G improvement. No full/tuning/seed rescue if fails.',
        locked_numeric_gate=dict(GV_minus_A_CIDEr_min=.01,GV_minus_A_paired95_lower_gt=0.,
            GV_CIDEr_gt_original=True,CHAIRs_increase_max=0.,CHAIRi_increase_max=0.,Recall_drop_max=.01,
            attribution_requires_GV_minus_G_paired95_lower_gt=0.,blind_review_required=True),
        comparison_limits='GV changes auxiliary gradient routing only relative to G. Decoder can still change indirectly via visual features during caption training. VG noise and contextualized CLIP patches remain; no novelty or gain established.')
    save(protocol,cfg)
    print('GV READY: same G supervision; auxiliary decoder PARAMETER gradients disabled, memory gradient enabled.',flush=True)


def build_model(cfg,tokenizer):
    from visual_only_model import VisualOnlyModel
    from captioning.scst_runtime import model_digest
    original=BASE_BUILD(dict(cfg,arm='G'),tokenizer)
    model=VisualOnlyModel(original.base).cuda().eval()
    model.initial_full_sha256=model_digest(model)
    region.LIVE_MODEL=model
    return model


def worker(protocol):
    if json.loads(Path(protocol).read_text())['arm']!='GV':raise ValueError('Expected GV')
    region.build_model=build_model
    region.worker(protocol)


def evaluate(protocol):
    region.evaluate(protocol)
    cfg=json.loads(Path(protocol).read_text())
    path=Path(cfg['output'])/'run_summary.json'
    result=json.loads(path.read_text())
    result.update(status='GV_PILOT_COMPLETE_WAIT_MATCHED_A_G_AUDIT',
        auxiliary_gradient_policy=cfg['auxiliary_gradient_policy'])
    save(path,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','evaluate','archive'])
    p.add_argument('--config',required=True);args=p.parse_args()
    (region.hidden.archive if args.mode=='archive' else globals()[args.mode])(args.config)
