"""Matched G/S/A comparison from previously cached review data. No model execution."""
import json
from pathlib import Path
from evidence_data import save
from evidence_metrics import paired

HERE=Path(__file__).resolve().parent


def compare():
    cached={arm:json.loads((HERE/f'review/loaded_{arm}.json').read_text(encoding='utf-8'))['json'] for arm in ('G','S')}
    results={arm:json.loads((HERE/f'{arm}_RESULT.json').read_text(encoding='utf-8')) for arm in ('G','S')}
    g,s=cached['G'],cached['S'];gc,sc=g['protocol.json'],s['protocol.json']
    assert results['G']['status']=='AUDIT_PASS_G_FAILS_A_GATE'
    assert results['S']['status']=='AUDIT_PASS_S_FAILS_A_GATE'
    allowed={'arm','labels','output','preparation_seconds','stage'}
    differences={k for k in set(gc)|set(sc) if gc.get(k)!=sc.get(k)}
    assert differences==allowed,differences
    for rank in (0,1):
        assert g[f'runtime_rank{rank}.json']==s[f'runtime_rank{rank}.json']
        assert g[f'optimizer_rank{rank}.json']==s[f'optimizer_rank{rank}.json']
        assert g[f'worker_complete_rank{rank}.json']['initial_head_sha256']==s[f'worker_complete_rank{rank}.json']['initial_head_sha256']
        gh,sh=g[f'history_rank{rank}.json'],s[f'history_rank{rank}.json']
        for x,y in zip(gh,sh):
            for key in ('step','epoch','local_image_ids','local_regions','global_regions','global_observed_labels','global_images_seen','scst_weight'):
                assert x[key]==y[key],(rank,key,x['step'])
    for name in ('origin_predictions.json','origin_per_image_cider.json','origin_chair_output.json','origin_metrics.json','groups.json','label_receipt.json','train_object_targets.json'):
        assert g[name]==s[name],name
    ids=[r['image_id'] for r in g['on_per_image_cider.json']]
    assert ids==[r['image_id'] for r in s['on_per_image_cider.json']]
    comparison=paired([r['CIDEr'] for r in s['on_per_image_cider.json']],[r['CIDEr'] for r in g['on_per_image_cider.json']])
    pred={arm:{r['image_id']:r['caption'] for r in data['on_predictions.json']} for arm,data in cached.items()}
    flags={arm:{r['image_id'] for r in data['on_chair_output.json']['sentences'] if r['mscoco_hallucinated_words']} for arm,data in cached.items()}
    result=dict(status='MATCHED_AUDIT_PASS_BOTH_ARMS_FAIL_KEEP_A',
        source_zip_sha256={arm:r['zip_sha256'] for arm,r in results.items()},
        common_source_sha256=gc['implementation_sha256'],common_stage_sha256=gc['stage_sha256'],
        common_label_receipt_sha256=gc['label_receipt_sha256'],common_source_checkpoint_sha256=gc['source_checkpoint_sha256'],
        common_initial_head_tensor_sha256=results['G']['initial_head_tensor_sha256'],
        protocol_differing_keys=sorted(differences),
        matched_checks=['exact source and locked hyperparameters','identical staged tensor bytes','identical label/reward data and cohorts',
            'identical rank runtime and optimizer options','identical initial head','identical per-rank image order and auxiliary event counts',
            'identical A5000 captions/per-image CIDEr/CHAIR'],
        metrics=dict(A=g['origin_metrics.json'],G=g['on_metrics.json'],S=s['on_metrics.json']),
        paired_CIDEr=dict(G_minus_A=results['G']['on_minus_A'],S_minus_A=results['S']['on_minus_A'],S_minus_G=comparison),
        S_minus_G_paired95_lower_positive=comparison['paired95'][0]>0,
        caption_comparison=dict(G_S_changed=sum(pred['G'][i]!=pred['S'][i] for i in ids),
            hallucination_flags_common=len(flags['G']&flags['S']),G_only=len(flags['G']-flags['S']),S_only=len(flags['S']-flags['G'])),
        gates=dict(G_vs_A=results['G']['A_gate'],S_vs_A=results['S']['S_A_gate'],spatial_caption_benefit=comparison['paired95'][0]>0),
        decision='Keep A. Stop this fixed noun-evidence configuration; no automatic full run, coefficient sweep or extra seed.',
        interpretation='Both variants increase object recall and hallucination and lower CIDEr. The negative outcome is not exclusive to bbox pooling; the responsible common component is not identified.',
        controls='S/A failed, so inference controls were correctly skipped. No image-specific benefit claim is supported.',
        test_used=False,limits=['One seed and historically reused validation. Paired image CI does not quantify training-seed uncertainty.',
            'Source/input/receipt audits do not independently verify missing head/checkpoint bytes.',
            'CHAIR flags are annotation-based; no new human image review was performed.',
            'A negative configuration does not establish that spatial supervision or visual-semantic alignment is impossible.'])
    save(HERE/'GS_RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='common_source_sha256'},ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':compare()
