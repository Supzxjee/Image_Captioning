"""Matched G/T/A saved-artifact comparison; never executes a model."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_g_review import HERE,load,audit,index,paired


def compare(a,g,t):
    previous=audit(a,g)
    gt,tt=g['json'],t['json']
    gp,tp=gt['protocol.json'],tt['protocol.json']
    diff=sorted(k for k in set(gp)|set(tp) if gp.get(k)!=tp.get(k))
    assert diff==['arm','output','stage'],diff
    assert tp['arm']=='T' and g['source_hashes']==t['source_hashes']
    for name in ['auxiliary_manifest.json','groups.json','hidden_cache_receipt.json','human_review_cohort.json',
                 'train_object_targets.json','origin_predictions.json','origin_per_image_cider.json','origin_chair_output.json']:
        assert gt[name]==tt[name],name
    summary=tt['run_summary.json']
    assert t['root_hashes']['protocol.json']==summary['protocol_sha256']
    for name,expected in tp['implementation_sha256'].items():
        assert t['source_hashes']['source/'+name]==expected,name
    for key in ['initial_base_sha256','evaluation_reference_tokens_sha256','cohort_sha256',
                'auxiliary_schedule_sha256','auxiliary_payload_sha256']:
        assert summary[key]==gt['run_summary.json'][key],key
    assert not summary['test_used'] and not tp['test_used']
    workers=[tt[f'worker_complete_rank{rank}.json'] for rank in (0,1)]
    for key in ['initial_base_sha256','initial_full_sha256','final_full_sha256','final_base_sha256']:
        assert workers[0][key]==workers[1][key]
    histories=[]
    for rank,worker in enumerate(workers):
        assert worker['status']=='COMPLETE' and worker['updates']==1250 and worker['source_arm']=='T'
        assert worker['trainable_base_parameters']==33520400 and worker['trainable_branch_parameters']==0
        assert not worker['test_used'] and worker['final_full_sha256']==summary['final_full_sha256']
        assert worker['initial_full_sha256']==gt[f'worker_complete_rank{rank}.json']['initial_full_sha256']
        for filename in [f'runtime_rank{rank}.json',f'optimizer_rank{rank}.json']:
            assert tt[filename]==gt[filename],filename
        history=tt[f'history_rank{rank}.json'];histories.extend(history)
        assert [row['step'] for row in history]==list(range(1,1251))
        for row,gr in zip(history,gt[f'history_rank{rank}.json']):
            assert np.isfinite([v for v in row.values() if isinstance(v,(int,float))]).all()
            assert row['scst_weight']==(0 if row['step']<=200 else .05)
            assert row['phrase_weight']==.1 and row['aux_samples']==4
            assert row['mean_aux_patches']==196 and row['mean_original_region_patches']==gr['mean_original_region_patches']
            expected=row['ce_loss']+row['scst_weight']*row['scst_loss']+.1*row['phrase_ce_loss']
            assert abs(row['loss']-expected)<2e-6
        assert history[0]['ce_loss']==gt[f'history_rank{rank}.json'][0]['ce_loss']
    metric=summary['metrics']['on']
    cider=index(tt['on_per_image_cider.json'],'CIDEr');caption=index(tt['on_predictions.json'],'caption')
    ids=sorted(cider)
    assert set(cider)==set(caption)==set(index(gt['on_predictions.json'],'caption'))
    assert abs(np.mean(list(cider.values()))-metric['CIDEr'])<1e-12
    chair=tt['on_chair_output.json']['sentences']
    assert len(chair)==1000 and {r['image_id'] for r in chair}=={r['coco_id'] for r in tt['groups.json']['val']}
    counts=dict(images=1000,hallucinated_captions=sum(bool(r['mscoco_hallucinated_words']) for r in chair),
        hallucinated_mentions=sum(len(r['mscoco_hallucinated_words']) for r in chair),
        object_mentions=sum(len(r['mscoco_generated_words']) for r in chair),
        recalled_unique_objects=sum(len(set(r['mscoco_generated_words'])&set(r['mscoco_gt_words'])) for r in chair),
        reference_unique_objects=sum(len(set(r['mscoco_gt_words'])) for r in chair))
    assert counts==metric['counts']
    for k,value in dict(CHAIRs=counts['hallucinated_captions']/1000,
        CHAIRi=counts['hallucinated_mentions']/counts['object_mentions'],
        Recall=counts['recalled_unique_objects']/counts['reference_unique_objects']).items():
        assert abs(value-metric['chair'][k])<1e-12
    metrics=dict(previous['metrics'],T=metric)
    scores={'T':cider,'G':index(gt['on_per_image_cider.json'],'CIDEr'),
        'A':index(a['json']['on_per_image_cider.json'],'CIDEr'),
        'original':index(a['json']['origin_per_image_cider.json'],'CIDEr')}
    comparisons=dict(previous['comparisons'])
    for l,r in [('G','T'),('T','A'),('T','original')]:
        comparisons[l+'_minus_'+r]=paired(scores[l],scores[r],ids)
    checks=dict(previous['gate_checks'],G_minus_T_paired_lower_positive=comparisons['G_minus_T']['ci95'][0]>0)
    assert sum(r['aux_samples'] for r in histories)==10000
    return dict(status='MATCHED_G_T_A_ARTIFACT_AUDIT_PASS',
        decision='NUMERIC_PASS_PENDING_BLIND_REVIEW' if all(checks.values()) else 'NO_GO_CURRENT_G_CONFIGURATION_VS_A',
        metrics=metrics,comparisons=comparisons,gate_checks=checks,
        archives=dict(previous['archives'],T=t['archive']),G_T_protocol_differences=diff,
        G_T_sources_identical=True,G_T_runtime_optimizer_stage_targets_originals_matched=True,
        T_auxiliary=dict(draws=10000,mean_patches=196.,
            first100_phrase_loss=float(np.mean([r['phrase_ce_loss'] for r in histories if r['step']<=100])),
            last100_phrase_loss=float(np.mean([r['phrase_ce_loss'] for r in histories if r['step']>1150]))),
        G_auxiliary=previous['auxiliary'],
        training_seconds=dict(G=previous['training_seconds_by_rank'],
            T=[tt[f'history_rank{rank}.json'][-1]['seconds'] for rank in (0,1)]),
        changed_G_T_captions=sum(index(gt['on_predictions.json'],'caption')[i]!=caption[i] for i in ids),
        gpu_preflight_pass=previous['gpu_preflight_pass'] and all(s in t['notebook_log'] for s in
            ['TWO_GPU_REGION_CAPTION_SMOKE_PASS G','TWO_GPU_REGION_CAPTION_SMOKE_PASS T']),
        test_used=False,model_executed_locally=False,
        interpretation='Region conditioning outperforms the same-text whole-image auxiliary control on reused dev; G does not reliably beat A, and T drops below A. This is not a successful caption-quality gain or proven semantic grounding.',
        limits=['Reused dev1000 and one seed; paired image bootstrap excludes training-seed variance and selection bias.',
            'Positive G-T can reflect mitigating harm from poorly localized auxiliary supervision; cannot infer G improves baseline.',
            'CLIP patches contextualized; mask is not complete isolation of visual evidence.',
            'CHAIR evaluates object hallucination, not relation truth; blind visual review not performed.',
            'Checkpoint bytes absent; state/checkpoint hashes worker-reported.'])


def markdown(r):
    lines=['# VG region-caption — đối chứng A/G/T hoàn tất','','**Quyết định: chưa mở rộng cấu hình G. Chỉ định vùng tốt hơn đối chứng cùng phrase toàn ảnh T, nhưng chưa cải thiện đáng tin cậy so với A.**','',
        '| Nhánh | CIDEr ↑ | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Caption ảo giác |','|---|---:|---:|---:|---:|---:|']
    for name in ['original','A','G','T']:
        m=r['metrics'][name];c=m['chair']
        lines.append(f"| {name} | {m['CIDEr']:.6f} | {100*c['CHAIRs']:.2f}% | {100*c['CHAIRi']:.3f}% | {100*c['Recall']:.3f}% | {m['counts']['hallucinated_captions']}/1000 |")
    lines+=['','CIDEr giữ thang evaluator, không nhân100. Original là checkpoint đầu; A cùng lịch caption CE/SCST không có phrase phụ, G thêm phrase phụ theo vùng, T thêm đúng phrase đó nhưng đọc toàn ảnh.','',
        '| So sánh | Δ CIDEr | Paired95% CI |','|---|---:|---|']
    for name in ['G_minus_A','G_minus_T','T_minus_A','G_minus_original','T_minus_original']:
        d=r['comparisons'][name];lines.append(f"| {name} | {d['delta']:+.6f} | [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] |")
    lines+=['','Bootstrap20000lần,seed42,ghép theo1000imageIDs development đã dùng nhiều lần; không phải test độc lập hay nhiều seed.','',
        '## Kết luận có thể rút ra','',
        'G−T +.022654 có CI dương trong phép resampling ảnh này, nên chỉ định vùng có tín hiệu hữu ích so với dùng cùng phrase với toàn ảnh. Nhưng G−A chỉ+.002888, CI chứa0 và CHAIRs/i tăng nhẹ, nên chưa đạt gate nâng chất lượng baseline. T−A −.019766 với CI âm: một phần khoảng cách G/T đến từ đối chứng T giảm chất lượng. Không diễn giải G>T thành G tốt hơn baseline hoặc đã chứng minh grounding đúng.','',
        'Giả thuyết cần trao đổi với Astra: chỉ định vùng có thể giảm sự không khớp giữa phrase cục bộ và ảnh toàn cảnh; nguồn VG yếu hoặc mục tiêu phụ cũng có thể gây nhiễu. Đây là giả thuyết cơ chế, chưa kiểm chứng bằng visual review hoặc ablation khác. Không tự tăng dữ liệu/tune/full train để cứu kết quả.','',
        '## Audit','',
        'G/T protocol chỉ khác arm,output,stage path; source runtime hoàn toàn giống nhau, cùng payload/lịch phrase,stage hash,caption cohort,targets,optimizer,runtime và original predictions. Hai rank T đủ1250updates,tham số như A/G,loss hữu hạn đúng công thức; T dùng196patch ở mọi rank-step, G trung bình32.258patch; mean số patch vùng gốc khớp từng step G/T. Không thêm tham số hoặc dùng vùng VG khi inference caption. Synthetic CPU/model/DDP preflight qua trong log.','',
        f"Training G≈{max(r['training_seconds']['G'])/60:.2f}phút, T≈{max(r['training_seconds']['T'])/60:.2f}phút; không gồm staging,khởi tạoIDF/evaluation. G/T khác{r['changed_G_T_captions']}/1000caption, chưa chấm đúng/sai bằng ảnh.",'',
        f"Phrase loss G: {r['G_auxiliary']['first100_phrase_loss']:.4f}→{r['G_auxiliary']['last100_phrase_loss']:.4f}; T: {r['T_auxiliary']['first100_phrase_loss']:.4f}→{r['T_auxiliary']['last100_phrase_loss']:.4f} (mean100step đầu/cuối trên minibatch khác nhau, không heldoutmetric). Loss thấp hơn không thay thế gain caption hoặc bằng chứng grounding.",'',
        'Đã chỉ xử lý saved artifacts trên CPU; không chạy model, sinh caption, hay huấn luyện mới. CHAIR không chấm quan hệ. Checkpoint không nằm trong reviewZIP; hash do worker báo cáo. Chờ Astra quyết định hướng tiếp theo từ tóm tắt, không yêu cầu đọc lại rawfiles.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arm in ('a','g','t'):p.add_argument('--'+arm,type=Path,required=True)
    args=p.parse_args()
    result=compare(load(args.a,'A'),load(args.g,'G'),load(args.t,'T'))
    (HERE/'G_T_A_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'G_T_A_RESULT.md').write_text(markdown(result),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','decision','comparisons','gate_checks','T_auxiliary','training_seconds']},indent=2))
