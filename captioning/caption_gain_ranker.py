"""Predict reference-CIDEr gain relative to fixed and abstain conservatively.

Calibration corrects selected-proposal error on disjoint train images. It does
not guarantee selective precision or safety under train/validation shift.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from .learned_caption_ranker import CaptionRanker,select_rank


class CaptionGainRanker(CaptionRanker):
    def forward(self,batch):
        representation=self.represent(batch)
        baseline=select_rank(batch['fixed_scores'],batch['candidate_mask'])
        index=torch.arange(len(baseline),device=baseline.device)
        relative=representation-representation[index,baseline][:,None,:]
        utility=self.head(relative).squeeze(-1)
        gains=utility-utility[index,baseline][:,None]
        return gains.masked_fill(~batch['candidate_mask'],float('-inf')),baseline


def gain_loss(gains,baseline,targets,mask):
    index=torch.arange(len(baseline),device=baseline.device)
    target_gain=targets-targets[index,baseline][:,None]
    valid=mask.clone();valid[index,baseline]=False
    if not valid.any():return gains[index,baseline].sum()*0
    error=F.smooth_l1_loss(gains[valid],target_gain[valid],beta=.1,reduction='none')
    weights=torch.where(target_gain[valid]<0,2.,1.)
    return (error*weights).sum()/weights.sum()


def propose(gains,baseline,correction=0.):
    if not math.isfinite(correction) or correction<0:raise ValueError('Invalid correction.')
    best=gains.argmax(-1);index=torch.arange(len(best),device=best.device)
    return torch.where(gains[index,best]>correction,best,baseline)


def calibrate(gains,baselines,targets,coverage=.9):
    """Quantile of error for the model-selected proposal; not oracle selection."""
    if not 0<coverage<1:raise ValueError('Invalid nominal coverage.')
    predicted=np.asarray(gains,dtype=float);truth=np.asarray(targets,dtype=float)
    if predicted.shape!=truth.shape or len(predicted)==0:raise ValueError('Invalid calibration arrays.')
    baselines=np.asarray(baselines);rows=np.arange(len(predicted));proposal=predicted.argmax(1)
    chosen=np.where(predicted[rows,proposal]>0,proposal,baselines)
    gain=truth[rows,chosen]-truth[rows,baselines]
    errors=predicted[rows,chosen]-gain
    if not np.isfinite(errors).all():raise ValueError('Nonfinite calibration proposal error.')
    k=math.ceil((len(errors)+1)*coverage)
    if k>len(errors):raise ValueError('Calibration set too small for finite quantile.')
    correction=max(0.,float(np.sort(errors)[k-1]))
    return {'correction':correction,'calibration_images':len(errors),'nominal_error_coverage':coverage,
            'quantile_order':k,'calibration_raw_proposal_changes':int((chosen!=baselines).sum()),
            'calibration_error_covered':float((errors<=correction).mean()),
            'validation_used':False,'selective_precision_guaranteed':False,
            'limitation':'Empirical train-image correction;no guarantee under distribution shift or conditional on switching.'}
