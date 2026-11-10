"""Shared neural/NumPy fusion algebra and saved gate diagnostics, no Torch import."""
import math
import numpy as np


def fuse_memory(visual,objects,relations,branch,strength):
    delta=branch*(objects-visual)+(1-branch)*(relations-visual)
    return visual+strength*delta


def update_gate_stats(state,values):
    a=np.asarray(values,dtype=np.float64).reshape(-1)
    if not len(a) or not np.isfinite(a).all() or (a<0).any() or (a>1).any():raise ValueError('Invalid semantic strength')
    state=state or dict(count=0,total=0.,minimum=1.,maximum=0.,below_point1=0,above_point9=0)
    return dict(count=state['count']+len(a),total=state['total']+float(a.sum()),minimum=min(state['minimum'],float(a.min())),maximum=max(state['maximum'],float(a.max())),
        below_point1=state['below_point1']+int((a<.1).sum()),above_point9=state['above_point9']+int((a>.9).sum()))


def merge_gate_stats(states):
    if not states or any(s['count']<=0 or not math.isfinite(s['total']) for s in states):raise ValueError('Missing semantic gate diagnostics')
    count=sum(s['count'] for s in states)
    return dict(patches=count,mean=sum(s['total'] for s in states)/count,minimum=min(s['minimum'] for s in states),maximum=max(s['maximum'] for s in states),
        fraction_below_point1=sum(s['below_point1'] for s in states)/count,fraction_above_point9=sum(s['above_point9'] for s in states)/count,
        interpretation='Learned residual strength, not a calibrated confidence or grounding accuracy.')
