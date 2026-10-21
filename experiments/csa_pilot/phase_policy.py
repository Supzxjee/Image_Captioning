"""Locked two-phase screening recipe; not an optimized training schedule."""
import hashlib
import json

UPDATES = 1250
ADAM = dict(weight_decay=.01,betas=[.9,.999],eps=1e-8,amsgrad=False,maximize=False,
            foreach=None,capturable=False,differentiable=False,fused=None)


def schedule(phase, step):
    if phase not in (1,2) or not 1 <= step <= UPDATES:
        raise ValueError('Invalid phase/update.')
    return dict(base_lr=0. if phase == 1 else 5e-6,
                core_lr=1e-4*min(step/100,1.) if phase == 1 else 2e-5,
                scst_weight=.05 if phase == 2 and step > 200 else 0.)


def contract(cfg):
    # Output locations/mount paths may change on resume; input identities/math may not.
    value = dict(format='csa_two_phase_v1',updates_per_phase=UPDATES,global_batch=8,seed=42,
                 adam=ADAM,clip=1.,phase1_warmup=100,phase2_ce_warmup=200,
                 objective='CE+.05SCST(CIDErD+.5object_count_v2)',
                 csa_commit=cfg['csa_commit'],project_commit=cfg['project_commit'],
                 implementation_sha256=cfg['implementation_sha256'],
                 cohort_sha256=cfg['cohort_sha256'],input_sha256=cfg['input_sha256'])
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def freeze_base(model, phase):
    if phase not in (1,2):
        raise ValueError(phase)
    # Both phases start from a fresh baseline loader, retaining its original freeze flags.
    if phase == 1:
        model.base.requires_grad_(False)
    model.eval()


def parameter_groups(model, phase):
    groups = []
    if phase == 2:
        groups.append(dict(params=[p for p in model.base.parameters() if p.requires_grad],lr=5e-6))
    groups.append(dict(params=list(model.extra.parameters()),lr=schedule(phase,1)['core_lr']))
    return groups
