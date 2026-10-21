"""Audit saved GV against completed A/G/T. No model or new metric execution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_g_review import HERE,load,index,paired,gates
from audit_gt_review import compare


def audit_gv(a,g,t,v):
    previous=compare(a,g,t)
    base,files=g['json'],v['json'];gp,vp=base['protocol.json'],files['protocol.json']
    changed=sorted(k for k in set(gp)|set(vp) if gp.get(k)!=vp.get(k))
    assert set(changed)=={'stage','locked_numeric_gate','checkpoint_format','auxiliary_gradient_policy',
        'promotion_requires','output','implementation_sha256','comparison_limits','arm'},changed
    assert vp['arm']=='GV'
    for name,value in g['source_hashes'].items():assert v['source_hashes'][name]==value,name
    extra=set(v['source_hashes'])-set(g['source_hashes'])
    assert extra=={'source/'+n for n in ['visual_only_model.py','run_visual_only.py','test_visual_only_model.py','test_visual_only_ddp.py']}
    for name in extra:
        expected=hashlib.sha256((HERE/Path(name).name).read_text(encoding='utf-8').encode()).hexdigest()
        assert v['source_hashes'][name]==expected,name
    for name in ['auxiliary_manifest.json','groups.json','hidden_cache_receipt.json','human_review_cohort.json',
                 'train_object_targets.json','origin_predictions.json','origin_per_image_cider.json','origin_chair_output.json']:
        assert files[name]==base[name],name
    summary=files['run_summary.json']
    assert summary['protocol_sha256']==v['root_hashes']['protocol.json']
    assert not vp['test_used'] and not summary['test_used']
    for name,expected in vp['implementation_sha256'].items():assert v['source_hashes']['source/'+name]==expected
    for name in ['initial_base_sha256','evaluation_reference_tokens_sha256','cohort_sha256',
                 'auxiliary_schedule_sha256','auxiliary_payload_sha256']:
        assert summary[name]==base['run_summary.json'][name],name
    workers=[files[f'worker_complete_rank{rank}.json'] for rank in (0,1)]
    for name in ['initial_base_sha256','initial_full_sha256','final_full_sha256','final_base_sha256']:
        assert workers[0][name]==workers[1][name]
    history=[]
    for rank,w in enumerate(workers):
        assert w['status']=='COMPLETE' and w['updates']==1250 and w['source_arm']=='GV'
        assert w['trainable_base_parameters']==33520400 and w['trainable_branch_parameters']==0 and not w['test_used']
        assert w['final_full_sha256']==summary['final_full_sha256']
        assert w['initial_full_sha256']==base[f'worker_complete_rank{rank}.json']['initial_full_sha256']
        for name in [f'runtime_rank{rank}.json',f'optimizer_rank{rank}.json']:
            assert files[name]==base[name],name
        rows=files[f'history_rank{rank}.json'];history.extend(rows)
        assert [r['step'] for r in rows]==list(range(1,1251))
        for row,original in zip(rows,base[f'history_rank{rank}.json']):
            assert np.isfinite([x for x in row.values() if isinstance(x,(int,float))]).all()
            assert row['aux_decoder_gradient_enabled'] is False
            assert row['phrase_weight']==.1 and row['aux_samples']==4
            assert row['mean_aux_patches']==row['mean_original_region_patches']==original['mean_aux_patches']
            assert row['scst_weight']==(0 if row['step']<=200 else .05)
            expected=row['ce_loss']+row['scst_weight']*row['scst_loss']+.1*row['phrase_ce_loss']
            assert abs(row['loss']-expected)<2e-6
        assert rows[0]['ce_loss']==base[f'history_rank{rank}.json'][0]['ce_loss']
        assert rows[0]['phrase_ce_loss']==base[f'history_rank{rank}.json'][0]['phrase_ce_loss']
    required=['test_aux_uses_current_decoder_not_an_initial_snapshot',
        'test_aux_value_and_visual_gradient_match_G_but_decoder_gradient_stops',
        'test_combined_decoder_gradient_equals_caption_only_CE_and_SCST']
    assert all(any(name in line and line.endswith('... ok') for line in v['notebook_log'].splitlines()) for name in required)
    assert 'TWO_GPU_GV_SMOKE_PASS' in v['notebook_log']
    scores=index(files['on_per_image_cider.json'],'CIDEr');captions=index(files['on_predictions.json'],'caption')
    ids=sorted(scores);assert set(ids)==set(captions)==set(index(base['on_predictions.json'],'caption'))
    metric=summary['metrics']['on'];assert abs(np.mean(list(scores.values()))-metric['CIDEr'])<1e-12
    sentences=files['on_chair_output.json']['sentences']
    assert len(sentences)==1000 and {r['image_id'] for r in sentences}=={r['coco_id'] for r in files['groups.json']['val']}
    counts=dict(images=1000,hallucinated_captions=sum(bool(r['mscoco_hallucinated_words']) for r in sentences),
        hallucinated_mentions=sum(len(r['mscoco_hallucinated_words']) for r in sentences),
        object_mentions=sum(len(r['mscoco_generated_words']) for r in sentences),
        recalled_unique_objects=sum(len(set(r['mscoco_generated_words'])&set(r['mscoco_gt_words'])) for r in sentences),
        reference_unique_objects=sum(len(set(r['mscoco_gt_words'])) for r in sentences))
    assert metric['counts']==counts
    chair=dict(CHAIRs=counts['hallucinated_captions']/1000,CHAIRi=counts['hallucinated_mentions']/counts['object_mentions'],
               Recall=counts['recalled_unique_objects']/counts['reference_unique_objects'])
    assert all(abs(chair[k]-metric['chair'][k])<1e-12 for k in chair)
    metrics=dict(previous['metrics'],GV=metric);comparisons=dict(previous['comparisons'])
    for name,artifact in [('A',a),('G',g),('T',t)]:
        comparisons['GV_minus_'+name]=paired(scores,index(artifact['json']['on_per_image_cider.json'],'CIDEr'),ids)
    checks=gates(dict(A=metrics['A'],G=metric,original=metrics['original']),comparisons['GV_minus_A'])
    checks['GV_minus_G_paired_lower_positive']=comparisons['GV_minus_G']['ci95'][0]>0
    assert sum(r['aux_samples'] for r in history)==10000
    return dict(status='MATCHED_GV_A_G_T_ARTIFACT_AUDIT_PASS',
        decision='NUMERIC_PASS_PENDING_BLIND_REVIEW' if all(checks.values()) else 'STOP_VG_REGION_CE_CURRENT_CACHE_SCALE',
        metrics=metrics,comparisons=comparisons,gate_checks=checks,archives=dict(previous['archives'],GV=v['archive']),
        GV_G_protocol_difference_keys=changed,original_G_sources_preserved=True,
        gradient_preflight_pass=True,gradient_flag_false_rank_steps=len(history),
        auxiliary=dict(draws=10000,mean_patches=float(np.mean([r['mean_aux_patches'] for r in history])),
            first100_phrase_loss=float(np.mean([r['phrase_ce_loss'] for r in history if r['step']<=100])),
            last100_phrase_loss=float(np.mean([r['phrase_ce_loss'] for r in history if r['step']>1150]))),
        training_seconds_by_rank=[files[f'history_rank{rank}.json'][-1]['seconds'] for rank in (0,1)],
        global_gradient_clipping=dict(threshold=1.,rank_steps_above_threshold=sum(r['grad_norm']>1 for r in history),
            limit='Direct decoder auxiliary gradients stop, but global clipping can change the scale of caption decoder updates. Visual-memory adaptation also has indirect effects.'),
        test_used=False,model_or_training_executed_locally=False,
        interpretation='GV fails locked baseline improvement and does not support the proposed decoder-gradient harm explanation in this configuration. Point estimates below A/G are not proof of inferiority because paired CIs include zero.',
        limits=['Reused dev1000, one seed; paired image bootstrap excludes seed variance and selection bias.',
            'No visual/human factual review; CHAIR measures objects, not relation correctness.',
            'GV isolates direct auxiliary decoder gradient, not all indirect decoder update effects (visual path and global gradient clipping).',
            'Checkpoint bytes absent; checkpoint/state hashes are worker-reported.'])


def markdown(r):
    lines=['# GV — kiểm tra đường gradient phrase phụ','','**Theo tiêu chí khóa trước: dừng VG region-CE với cache và quy mô hiện tại. GV chạy hợp lệ nhưng chưa vượt A hoặc G.**','',
        '| Nhánh | CIDEr ↑ | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Caption ảo giác |','|---|---:|---:|---:|---:|---:|']
    for name in ['original','A','G','T','GV']:
        m=r['metrics'][name];c=m['chair'];lines.append(f"| {name} | {m['CIDEr']:.6f} | {100*c['CHAIRs']:.2f}% | {100*c['CHAIRi']:.3f}% | {100*c['Recall']:.3f}% | {m['counts']['hallucinated_captions']}/1000 |")
    lines+=['','CIDEr giữ thang evaluator, không nhân100. A là caption CE/SCST đối chứng; GV chỉ bỏ gradient phụ trực tiếp vào tham số decoder.','',
        '| So sánh | Δ CIDEr | Paired95% CI |','|---|---:|---|']
    for name in ['GV_minus_A','GV_minus_G','GV_minus_T']:
        d=r['comparisons'][name];lines.append(f"| {name} | {d['delta']:+.6f} | [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] |")
    lines+=['','Bootstrap20000lần,seed42,1000ảnh ghép theoID; dev đã dùng nhiều lần, một seed, không phải xác nhận test độc lập.','',
        'GV−A −.012292, GV−G −.015181, cả hai CI chứa0. Không có gain đủ tiêu chí, nhưng cũng chưa chứng minh GV chắc chắn kém hơn A/G. Recall nhỉnh hơn A (1492vs1490đối tượng unique), trong khi CIDEr thấp hơn và caption ảo giác53vs52, không đủ coi cấu hình thành công.','',
        'Giả thuyết gradient phrase vào decoder là nguyên nhân chính làm mất lợi ích chưa được ủng hộ bởi phép thử này. Không suy ra mọi giám sát vùng hoặc mọi hướng căn chỉnh bất khả thi. Theo thỏa thuận trước: không full train, không tự thử freeze/trọng số/seed khác để cứu chuỗi này; báo Astra chọn hướng tiếp từ tóm tắt.','',
        '## Tính hợp lệ và giới hạn cơ chế','',
        'Hai rank đủ1250steps, lịchCE/SCST+.1phrase đúng và loss hữu hạn. Cùng init/cohort/stage hash/payload/schedule/targets/optimizer/runtime/original predictions/evaluator với G; mọi source gốc G giữ nguyên, chỉ thêm4sourceGV. Mọi rank-step ghi aux_decoder_gradient_enabled=false; synthetic tests thực sự qua trong log: aux loss/visual gradient khớpG và decoder.grad=None; decoder gradient combinedCE/SCST khớpcaption-only; current weights/parameter restoration;2GPU DDP smoke. Đây là test synthetic và source audit, không đo tách gradient trên từng minibatch thật.','',
        f"Caveat: global gradient clipping norm1 được giữ như G, có {r['global_gradient_clipping']['rank_steps_above_threshold']}/{r['gradient_flag_false_rank_steps']}rank-steps vượt ngưỡng. Dù decoder không nhận gradient phrase trực tiếp, loss phụ vẫn có thể đổi hệ số clipping của cập nhật decoder; visual memory đổi cũng ảnh hưởng nhiệm vụ caption. Do đó không diễn giải đây là decoder được cách ly hoàn toàn khỏi VG.",'',
        f"Aux10000draws,mean{r['auxiliary']['mean_patches']:.3f}patch; phraseCE mean100step đầu{r['auxiliary']['first100_phrase_loss']:.4f}→100step cuối{r['auxiliary']['last100_phrase_loss']:.4f}, trên minibatch khác nhau, không heldoutmetric. Training≈{max(r['training_seconds_by_rank'])/60:.2f}phút, chưa gồm staging/IDF/eval.",'',
        'Không chạy model, inference hay training mới trên local. CHAIR không chấm quan hệ; chưa human visual review. Checkpoint không có trong reviewZIP, hashes do worker báo cáo.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arm in ['a','g','t','gv']:p.add_argument('--'+arm,type=Path,required=True)
    args=p.parse_args();r=audit_gv(load(args.a,'A'),load(args.g,'G'),load(args.t,'T'),load(args.gv,'GV'))
    (HERE/'GV_RESULT.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'GV_RESULT.md').write_text(markdown(r),encoding='utf-8')
    print(json.dumps({k:r[k] for k in ['status','decision','gate_checks','auxiliary','global_gradient_clipping']},indent=2))
