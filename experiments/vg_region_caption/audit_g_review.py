"""Saved-artifact G versus historical A audit. No model, metric rerun or training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'upt_hidden_pair'))
from audit_caption_reviews import paired


def sha(raw):return hashlib.sha256(raw).hexdigest()


def load(path,arm):
    archive_hash=sha(path.read_bytes())
    folder=HERE/'review';folder.mkdir(exist_ok=True)
    cache=folder/f'loaded_{arm}.json'
    if cache.is_file():
        old=json.loads(cache.read_text(encoding='utf-8'))
        if old['archive']['sha256']==archive_hash and 'root_hashes' in old:return old
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive members')
        roots={n:archive.read(n) for n in names if '/' not in n and n.endswith('.json')}
        result=dict(archive=dict(path=str(path),sha256=archive_hash),names=names,
            json={n:json.loads(raw) for n,raw in roots.items()},
            root_hashes={n:sha(raw) for n,raw in roots.items()},
            source_hashes={n:sha(archive.read(n)) for n in names if n.startswith('source/')},
            source_loop=archive.read('source/run_pilot.py').decode(),
            notebook_log=archive.read('notebook.log').decode(errors='replace'))
    cache.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
    return result


def index(rows,column):
    mapped={r['image_id']:r[column] for r in rows}
    if len(mapped)!=len(rows) or len(rows)!=1000:raise ValueError('Missing/duplicate development IDs')
    return mapped


def gates(metrics,comparison):
    a,g=metrics['A'],metrics['G']
    return dict(cider_gain_at_least_001=comparison['delta']>=.01,
        paired_ci_lower_positive=comparison['ci95'][0]>0,
        beats_original=g['CIDEr']>metrics['original']['CIDEr'],
        chair_s_not_increased=g['chair']['CHAIRs']<=a['chair']['CHAIRs'],
        chair_i_not_increased=g['chair']['CHAIRi']<=a['chair']['CHAIRi'],
        recall_drop_at_most_001=g['chair']['Recall']>=a['chair']['Recall']-.01)


def audit(a,g):
    artifacts={'A':a,'G':g};data={k:v['json'] for k,v in artifacts.items()}
    p={k:v['protocol.json'] for k,v in data.items()}
    allowed={'arm','implementation_sha256','auxiliary_payload_sha256','stage','promotion_requires',
        'phrase_weight','checkpoint_format','project_commit','auxiliary_schedule_sha256','output',
        'comparison_limits','aux_policy','locked_numeric_gate','evaluation_policy','loss'}
    differences=sorted(k for k in set(p['A'])|set(p['G']) if p['A'].get(k)!=p['G'].get(k))
    assert set(differences)==allowed,differences
    for name in ['groups.json','hidden_cache_receipt.json','human_review_cohort.json','train_object_targets.json',
                 'origin_predictions.json','origin_per_image_cider.json','origin_chair_output.json']:
        assert data['A'][name]==data['G'][name],name
    changed=[name for name in a['source_hashes'] if a['source_hashes'][name]!=g['source_hashes'].get(name)]
    assert changed==['source/run_pilot.py'],changed
    hook="        if hasattr(model, 'training_diagnostics'):\n            record.update(model.training_diagnostics())\n"
    assert g['source_loop'].replace(hook,'')==a['source_loop']
    aux=json.loads((HERE/'AUXILIARY_MANIFEST.json').read_text(encoding='utf-8'))
    assert data['G']['auxiliary_manifest.json']==aux
    assert p['G']['auxiliary_schedule_sha256']==aux['schedule_sha256']
    assert p['G']['auxiliary_payload_sha256']==aux['auxiliary_payload_sha256']
    histories={}
    for arm,files in data.items():
        summary=files['run_summary.json']
        assert artifacts[arm]['root_hashes']['protocol.json']==summary['protocol_sha256']
        for name,expected in p[arm]['implementation_sha256'].items():
            assert artifacts[arm]['source_hashes']['source/'+name]==expected,name
        assert not p[arm]['test_used'] and not summary['test_used']
        assert summary['initial_base_sha256']==data['A']['run_summary.json']['initial_base_sha256']
        assert summary['evaluation_reference_tokens_sha256']==data['A']['run_summary.json']['evaluation_reference_tokens_sha256']
        workers=[files[f'worker_complete_rank{rank}.json'] for rank in (0,1)]
        for k in ['initial_base_sha256','initial_full_sha256','final_full_sha256','final_base_sha256']:
            assert workers[0][k]==workers[1][k],(arm,k)
        for rank,worker in enumerate(workers):
            assert files[f'runtime_rank{rank}.json']==data['A'][f'runtime_rank{rank}.json']
            assert files[f'optimizer_rank{rank}.json']==data['A'][f'optimizer_rank{rank}.json']
            assert worker['status']=='COMPLETE' and worker['updates']==1250 and worker['source_arm']==arm
            assert worker['trainable_base_parameters']==33520400 and worker['trainable_branch_parameters']==0
            assert worker['final_full_sha256']==summary['final_full_sha256'] and not worker['test_used']
            history=files[f'history_rank{rank}.json'];histories[(arm,rank)]=history
            assert [r['step'] for r in history]==list(range(1,1251))
            for row in history:
                assert np.isfinite([v for v in row.values() if isinstance(v,(int,float))]).all()
                weight=0 if row['step']<=200 else .05
                assert row['scst_weight']==weight
                expected=row['ce_loss']+weight*row['scst_loss']
                if arm=='G':
                    assert row['phrase_weight']==.1 and row['aux_samples']==4
                    assert row['mean_aux_patches']==row['mean_original_region_patches']>=4
                    expected+=.1*row['phrase_ce_loss']
                assert abs(row['loss']-expected)<2e-6,(arm,row['step'])
    metrics={};scores={};captions={}
    for label,arm,condition in [('A','A','on'),('G','G','on'),('original','A','origin')]:
        files=data[arm];metrics[label]=files['run_summary.json']['metrics'][condition]
        scores[label]=index(files[f'{condition}_per_image_cider.json'],'CIDEr')
        captions[label]=index(files[f'{condition}_predictions.json'],'caption')
        assert set(scores[label])==set(scores['A'])==set(captions[label])
        assert abs(np.mean(list(scores[label].values()))-metrics[label]['CIDEr'])<1e-12
        chair=files[f'{condition}_chair_output.json']['sentences']
        expected_ids={r['coco_id'] for r in files['groups.json']['val']}
        assert {r['image_id'] for r in chair}==expected_ids and len(chair)==1000
        hallucinations=sum(len(r['mscoco_hallucinated_words']) for r in chair)
        mentions=sum(len(r['mscoco_generated_words']) for r in chair)
        recalled=sum(len(set(r['mscoco_generated_words'])&set(r['mscoco_gt_words'])) for r in chair)
        total=sum(len(set(r['mscoco_gt_words'])) for r in chair)
        counts=dict(images=1000,hallucinated_captions=sum(bool(r['mscoco_hallucinated_words']) for r in chair),
            hallucinated_mentions=hallucinations,object_mentions=mentions,recalled_unique_objects=recalled,reference_unique_objects=total)
        assert counts==metrics[label]['counts']
        expected=dict(CHAIRs=counts['hallucinated_captions']/1000,CHAIRi=hallucinations/mentions,Recall=recalled/total)
        assert all(abs(expected[k]-metrics[label]['chair'][k])<1e-12 for k in expected)
    ids=sorted(scores['A'])
    comparisons={l+'_minus_'+r:paired(scores[l],scores[r],ids) for l,r in [('G','A'),('G','original'),('A','original')]}
    checks=gates(metrics,comparisons['G_minus_A'])
    h=[row for rank in (0,1) for row in histories[('G',rank)]]
    return dict(status='SAVED_G_VS_A_ARTIFACT_AUDIT_PASS',
        decision='G_NUMERIC_GATE_PASS_WAIT_T_AND_BLIND_REVIEW' if all(checks.values()) else 'G_FAILS_LOCKED_GATE_WAIT_T_FOR_ATTRIBUTION',
        archives={arm:r['archive'] for arm,r in artifacts.items()},metrics=metrics,comparisons=comparisons,
        gate_checks=checks,protocol_difference_keys=differences,common_source_difference=changed,
        caption_training_loop_difference='Only optional two-line diagnostic hook; math unchanged',
        changed_G_A_captions=sum(captions['G'][i]!=captions['A'][i] for i in ids),
        auxiliary=dict(draws=sum(row['aux_samples'] for row in h),
            mean_patches=float(np.mean([row['mean_aux_patches'] for row in h])),
            first100_phrase_loss=float(np.mean([r['phrase_ce_loss'] for r in h if r['step']<=100])),
            last100_phrase_loss=float(np.mean([r['phrase_ce_loss'] for r in h if r['step']>1150])),
            schedule_sha256=aux['schedule_sha256'],payload_sha256=aux['auxiliary_payload_sha256']),
        training_seconds_by_rank=[histories[('G',rank)][-1]['seconds'] for rank in (0,1)],
        gpu_preflight_pass=all(s in g['notebook_log'] for s in ['TWO_GPU_REGION_CAPTION_SMOKE_PASS G','TWO_GPU_REGION_CAPTION_SMOKE_PASS T']),
        test_used=False,model_or_training_executed_locally=False,T_received=False,
        limits=['Reused dev1000; one seed, bootstrap does not cover selection bias or training-seed variance.',
                'T pending: no region-attribution conclusion, even if G beats original.',
                'CHAIR is object hallucination, not semantic relation correctness. No visual/human review in this audit.',
                'Checkpoint bytes not in review; checkpoint/state digests are worker-reported.'])


def markdown(r):
    lines=['# VG region-caption G — kết quả sơ bộ so với A','','**G hoàn tất đúng cấu hình, nhưng chưa đạt gate tăng chất lượng đã khóa. Chờ T để kiểm tra giá trị của vùng.**','',
        '| Mô hình | CIDEr ↑ | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Caption ảo giác |',
        '|---|---:|---:|---:|---:|---:|']
    for label in ['original','A','G']:
        m=r['metrics'][label];c=m['chair']
        lines.append(f"| {label} | {m['CIDEr']:.6f} | {100*c['CHAIRs']:.2f}% | {100*c['CHAIRi']:.3f}% | {100*c['Recall']:.3f}% | {m['counts']['hallucinated_captions']}/1000 |")
    lines+=['','CIDEr giữ thang evaluator, không nhân100. A là baseline huấn luyện tiếp cùng1250updates; original là checkpoint trước pilot.','',
        '| So sánh | Δ CIDEr | Paired bootstrap95% CI |','|---|---:|---|']
    for label,d in r['comparisons'].items():
        lines.append(f"| {label} | {d['delta']:+.6f} | [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] |")
    lines+=['','Bootstrap20000resamples,seed42,1000ảnh ghép theoID; không phải xác nhận trên test hoặc đo độ ổn định nhiều seed.','',
        '## Diễn giải','',
        'G−A chỉ khoảng+.002888, dưới ngưỡng+.01; CI chứa0. Không được kết luận G chắc chắn tốt hơn hoặc tương đương A. CHAIRs tăng52→53caption, unsupported mentions53→54; recall giảm1490→1484đối tượng duy nhất. CHAIRs/i không đạt điều kiện không tăng. Recall vẫn trong mức giảm tối đa.01 đã cho phép.','',
        'G tăng so original, nhưng A không có nhiệm vụ phụ cũng tăng; không gán toàn bộ G−original cho căn chỉnh vùng. Chưa có T nên chưa phân biệt tác dụng vùng với thêm văn bản/gradient. Không mở full train hoặc tune để cứu điểm. Giữ T đang chạy để hoàn tất đối chứng nếu ngân sách cho phép, không đổi cấu hình sau khi xem G.','',
        '## Kiểm tra','',
        'Cùng base initialization,cohort,stage hash,targets,legacy/cache,optimizer,runtime,evaluator reference và caption gốc với A. Common source chỉ khác run_pilot.py ở hook diagnostic2dòng; mỗi runtime source hash khớp protocol. Hai rank đủ1250updates, số tham số như A (33520400), không thêm branch; lịchCE/SCST/loss+.1phraseCE được đối chiếu từng step và hữu hạn. Synthetic model/mask/RNG tests vàDDP G/Tsmoke qua trong log Kaggle.','',
        f"Tổng{r['auxiliary']['draws']}lượt phrase; trung bình{r['auxiliary']['mean_patches']:.2f}patch/vùng. Phrase CE trung bình100bước đầu{r['auxiliary']['first100_phrase_loss']:.4f},100bước cuối{r['auxiliary']['last100_phrase_loss']:.4f}; đây là số mô tả trên minibatch khác nhau, không phải metric heldout. Giảm phrase loss không chứng minh grounding đúng.",
        f"Huấn luyện khoảng{max(r['training_seconds_by_rank'])/60:.2f}phút (không gồm staging,khởi tạoCIDEr hay đánh giá). G/A khác{r['changed_G_A_captions']}/1000caption, chưa chấm bằng ảnh.",'',
        'Không chạy model/training/inference/metric mới trên local. Checkpoint không nằm trongZIP; chỉ có worker-reported hashes. T chưa nhận, human review chưa thực hiện.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--a',type=Path,required=True);p.add_argument('--g',type=Path,required=True);args=p.parse_args()
    result=audit(load(args.a,'A'),load(args.g,'G'))
    (HERE/'G_VS_A_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'G_VS_A_RESULT.md').write_text(markdown(result),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','decision','comparisons','gate_checks','auxiliary','training_seconds_by_rank']},indent=2))
