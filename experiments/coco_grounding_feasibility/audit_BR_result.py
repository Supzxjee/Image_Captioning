"""Matched B/R/A source, exposure and metric audit; NumPy only, no model execution."""
import json
import numpy as np
from audit_R_result import HERE,ROOT,load,verify_run,paired


def main():
    ac=load(ROOT/'experiments/vg_region_caption/review/loaded_A.json')
    bc,rc=[load(HERE/f'review/loaded_{arm}.json') for arm in ('B','R')]
    b,a,bcfg,bs,bh=verify_run(bc,ac,'B')
    r,_,rcfg,rs,rh=verify_run(rc,ac,'R')
    differences=[k for k in bcfg if bcfg[k]!=rcfg.get(k)]
    assert set(bcfg)==set(rcfg) and set(differences)=={'arm','output','stage','preparation_seconds'}
    assert bc['sources']==rc['sources']
    for rank in (0,1):
        for x,y in zip(bh[rank],rh[rank]):
            assert x['local_image_ids']==y['local_image_ids']
            for key in ('local_spans','global_spans','global_person_spans','scst_weight'):
                assert x[key]==y[key],key
    comparisons={name:paired(left['on_per_image_cider.json'],right['on_per_image_cider.json'])
        for name,left,right in [('R_minus_B',r,b),('B_minus_A',b,a),('R_minus_A',r,a)]}
    metrics=dict(original=rs['metrics']['origin'],A=a['run_summary.json']['metrics']['on'],
        B=bs['metrics']['on'],R=rs['metrics']['on'])
    spatial_gate=comparisons['R_minus_B']['paired95_percentile'][0]>0
    ra=comparisons['R_minus_A'];R=metrics['R'];A=metrics['A']
    gates=dict(R_minus_A_at_least_001=ra['delta']>=.01,R_minus_A_paired95_lower_positive=ra['paired95_percentile'][0]>0,
        R_mean_CIDEr_above_original=R['CIDEr']>metrics['original']['CIDEr'],
        CHAIRs_not_above_A=R['chair']['CHAIRs']<=A['chair']['CHAIRs'],
        CHAIRi_not_above_A=R['chair']['CHAIRi']<=A['chair']['CHAIRi'],
        Recall_at_least_A_minus_001=R['chair']['Recall']>=A['chair']['Recall']-.01)
    def transition(old,new):
        hx={s['image_id'] for s in old['on_chair_output.json']['sentences'] if s['mscoco_hallucinated_words']}
        hy={s['image_id'] for s in new['on_chair_output.json']['sentences'] if s['mscoco_hallucinated_words']}
        return dict(cleared=len(hx-hy),new=len(hy-hx),shared=len(hx&hy))
    result=dict(status='MATCHED_B_R_COMPLETE_NO_LOCKED_CAPTION_GAIN_OR_SPATIAL_ATTRIBUTION',
        audit='PASS',archives=dict(B=bc['archive_sha256'],R=rc['archive_sha256']),
        metrics=metrics,comparisons=comparisons,locked_R_numeric_gates=gates,
        locked_R_numeric_gate_pass=all(gates.values()),bbox_position_attribution_gate_pass=spatial_gate,
        protocol_differences=sorted(differences),same_staged_feature_bytes=bcfg['stage_sha256']==rcfg['stage_sha256'],
        same_source_bundles=True,same_actual_B_R_image_order_and_span_schedule=True,
        exposures_per_arm=dict(images=10000,spans=17904,person_spans=4476,updates=1250),
        hallucination_transitions=dict(A_to_B=transition(a,b),B_to_R=transition(b,r),A_to_R=transition(a,r)),
        clipping_fraction_preclip_norm_above1={arm:float(np.mean([s['grad_norm']>1 for s in h[0]])) for arm,h in [('B',bh),('R',rh)]},
        checkpoint_sha256_reported=dict(B=bs['checkpoint_sha256'],R=rs['checkpoint_sha256']),
        checkpoint_bytes_independently_verified=False,test_used=False,local_model_execution=False,
        recommendation='Do not promote R or scale this recipe based on these results. Keep A as caption baseline; ask Astra to choose a discriminating diagnostic or alternative alignment approach.',
        limits=['Reused development1000, one seed; image bootstrap does not quantify training-seed variance.',
            'B/R actual image order and staged feature hash match; historical A actual order was not byte-logged.',
            'A gate failure does not prove all visual-semantic alignment approaches fail; this fixed pilot did not establish a gain.',
            'Weak noun/class attention-path supervision is not causal grounding or verified relation truth.'])
    (HERE/'BR_RESULT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','comparisons','locked_R_numeric_gates','bbox_position_attribution_gate_pass','hallucination_transitions']},indent=2))


if __name__=='__main__':main()
