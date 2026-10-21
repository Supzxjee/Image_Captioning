"""Audit saved A/H artifacts only; no model execution or new caption generation."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def paired(left, right, ids, seed=42):
    delta = np.array([left[i] - right[i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.concatenate([delta[rng.integers(0, len(delta), size=(500, len(delta)))].mean(axis=1)
                           for _ in range(40)])
    return dict(n=len(ids), delta=float(delta.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                positive=int((delta > 0).sum()), negative=int((delta < 0).sum()),
                equal=int((delta == 0).sum()), bootstrap_resamples=20000, bootstrap_seed=seed,
                scope='Paired development-image resampling; does not measure seed variance or selection bias.')


def audit(a_path, h_path):
    with zipfile.ZipFile(a_path) as a, zipfile.ZipFile(h_path) as h:
        archives = dict(A=a, H=h)
        read = lambda arm, name: json.loads(archives[arm].read(name))
        summaries = {arm: read(arm, 'run_summary.json') for arm in archives}
        protocols = {arm: read(arm, 'protocol.json') for arm in archives}
        differences = [k for k in protocols['A'] if protocols['A'][k] != protocols['H'].get(k)]
        assert set(differences) == {'arm', 'output', 'stage', 'conditions'}, differences
        for name in ['groups.json', 'hidden_cache_receipt.json', 'human_review_cohort.json',
                     'train_object_targets.json', 'origin_predictions.json', 'origin_per_image_cider.json']:
            assert a.read(name) == h.read(name), name
        for name in a.namelist():
            if name.startswith('source/'):
                assert a.read(name) == h.read(name), name
        for arm in archives:
            for rank in range(2):
                assert a.read(f'runtime_rank{rank}.json') == h.read(f'runtime_rank{rank}.json')
                worker = read(arm, f'worker_complete_rank{rank}.json')
                assert worker['status'] == 'COMPLETE' and worker['updates'] == 1250
                assert worker['source_arm'] == arm and not worker['test_used']
                assert worker['final_full_sha256'] == summaries[arm]['final_full_sha256']
                assert worker['initial_base_sha256'] == summaries['A']['initial_base_sha256']
                history = read(arm, f'history_rank{rank}.json')
                assert [r['step'] for r in history] == list(range(1, 1251))
                for row in history:
                    assert np.isfinite([v for v in row.values() if isinstance(v, (int, float))]).all()
                    assert row['scst_weight'] == (0 if row['step'] <= 200 else .05)
            assert not protocols[arm]['test_used']
        conditions = dict(A=('A', 'on'), H=('H', 'on'), H_off=('H', 'off'), original=('A', 'origin'))
        cider, captions = {}, {}
        metrics = {}
        for key, (arm, condition) in conditions.items():
            rows = read(arm, f'{condition}_per_image_cider.json')
            cider[key] = {r['image_id']: r['CIDEr'] for r in rows}
            predictions = read(arm, f'{condition}_predictions.json')
            captions[key] = {r['image_id']: r['caption'] for r in predictions}
            metrics[key] = summaries[arm]['metrics'][condition]
            assert len(cider[key]) == len(rows) == len(predictions) == 1000
            assert set(cider[key]) == set(cider['A']) == set(captions[key])
            assert abs(np.mean(list(cider[key].values())) - metrics[key]['CIDEr']) < 1e-12
            chair = read(arm, f'{condition}_chair_output.json')['sentences']
            assert sum(bool(r['mscoco_hallucinated_words']) for r in chair) == metrics[key]['counts']['hallucinated_captions']
            assert sum(len(r['mscoco_hallucinated_words']) for r in chair) == metrics[key]['counts']['hallucinated_mentions']
        ids = sorted(cider['A'])
        diag = [r for rank in range(2) for r in read('H', f'on_diagnostic_rank{rank}.json')]
        counts = {r['image_id']: r['valid_pairs'] for r in diag}
        assert set(counts) == set(ids)
        subsets = dict(all=ids, with_pairs=[i for i in ids if counts[i] > 0], no_pairs=[i for i in ids if counts[i] == 0])
        comparisons = {}
        for l, r in [('H', 'A'), ('H', 'original'), ('A', 'original'), ('H', 'H_off')]:
            comparisons[l + '_minus_' + r] = {subset: paired(cider[l], cider[r], members)
                                             for subset, members in subsets.items()}
        swap = {r['image_id']: r['caption'] for r in read('H', 'pair_swap_predictions.json')}
        sensitivity = {}
        for name, other in [('off', captions['H_off']), ('pair_swap', swap)]:
            sensitivity[name] = {subset: sum(captions['H'][i] != other[i] for i in members)
                                 for subset, members in subsets.items()}
            assert sensitivity[name]['all'] == summaries['H']['sensitivity'][name]['changed_captions']
            assert sensitivity[name]['no_pairs'] == 0
        branch = [r['branch_grad_norm'] for rank in range(2) for r in read('H', f'history_rank{rank}.json')]
        assert max(branch) > 0
        gate = protocols['H']['locked_numeric_gate']
        delta = comparisons['H_minus_A']['all']
        passed = dict(cider_gain_at_least_001=delta['delta'] >= .01,
                      paired_ci_lower_positive=delta['ci95'][0] > 0,
                      beats_original=metrics['H']['CIDEr'] > metrics['original']['CIDEr'],
                      chair_s_not_increased=metrics['H']['chair']['CHAIRs'] <= metrics['A']['chair']['CHAIRs'],
                      chair_i_not_increased=metrics['H']['chair']['CHAIRi'] <= metrics['A']['chair']['CHAIRi'],
                      recall_drop_at_most_001=metrics['H']['chair']['Recall'] >= metrics['A']['chair']['Recall'] - .01)
        return dict(status='MATCHED_ARTIFACT_AUDIT_PASS', decision='GO_NUMERIC_ONLY_PENDING_HUMAN' if all(passed.values()) else 'NO_GO_CURRENT_CONFIGURATION',
                    archives={arm: dict(path=str(path), sha256=sha(Path(path).read_bytes()))
                              for arm, path in [('A', a_path), ('H', h_path)]},
                    allowed_protocol_differences=differences, metrics=metrics,
                    comparisons=comparisons, sensitivity=sensitivity, pair_coverage={k: len(v) for k, v in subsets.items()},
                    changed_A_H=sum(captions['A'][i] != captions['H'][i] for i in ids),
                    H_branch_gradient=dict(positive_steps=sum(v > 0 for v in branch), total_rank_steps=len(branch), maximum=max(branch)),
                    locked_numeric_gate=gate, gate_checks=passed,
                    test_used=False, model_or_training_executed=False,
                    limits=['Reused dev1000, one seed; not confirmatory test evidence.',
                            'CHAIR evaluates object hallucination, not relation truth.',
                            'Sensitivity and nonzero gradients do not prove correct visual-semantic alignment.',
                            'H_off is adapted H without memory, not the matched A control.',
                            'No human factual labels read/generated in this audit; no visual correctness claims.',
                            'Review ZIP does not contain model checkpoint bytes; hashes are worker-reported.'])


def markdown(r):
    lines = ['# UPT hidden-pair A/H pilot — kết quả', '',
             '**Quyết định: NO-GO cho cấu hình H hiện tại; chưa chứng minh tăng chất lượng so với A.**', '',
             'Train 10.000 ảnh, development 1.000 ảnh, cùng base/seed/1250 updates. 200 bước CE rồi CE + .05 SCST; không dùng test.', '',
             '| Mô hình | CIDEr ↑ | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Caption ảo giác |',
             '|---|---:|---:|---:|---:|---:|']
    for name in ['original', 'A', 'H', 'H_off']:
        m = r['metrics'][name]; c = m['chair']
        lines.append(f"| {name} | {m['CIDEr']:.6f} | {100*c['CHAIRs']:.2f}% | {100*c['CHAIRi']:.3f}% | {100*c['Recall']:.3f}% | {m['counts']['hallucinated_captions']}/1000 |")
    lines += ['', 'CIDEr giữ thang của evaluator (không nhân 100); không có miền tối đa cố định.', '',
              '| So sánh CIDEr | Δ | Paired bootstrap 95% CI |', '|---|---:|---|']
    for name, subsets in r['comparisons'].items():
        d = subsets['all']; lines.append(f"| {name} | {d['delta']:+.6f} | [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] |")
    lines += ['', 'Bootstrap 20.000 lần, seed42, ghép theo image_id; không phản ánh phương sai seed hoặc việc đã xem dev nhiều lần.', '',
              f"H on/off đổi {r['sensitivity']['off']['all']}/1000 caption; đổi cặp UPT đổi {r['sensitivity']['pair_swap']['all']}/1000. Có {r['pair_coverage']['with_pairs']} ảnh có cặp, {r['pair_coverage']['no_pairs']} ảnh không có cặp; nhóm không cặp không đổi ở cả hai can thiệp.", '',
              'H có gradient nhánh và can thiệp gây đổi caption, nên không phải nhánh hoàn toàn bị ngắt. Tuy nhiên, bật memory không làm CIDEr tốt hơn tắt memory; độ nhạy không chứng minh quan hệ đúng.', '',
              'Phân nhóm chẩn đoán H−A: ảnh có cặp +.002564 (CI chứa0), không cặp −.017531 (CI chứa0). Không dùng nhóm thuận lợi để thay tiêu chí toàn bộ đã khóa. H và A có base weights cuối khác nhau; ảnh không cặp vẫn có thể đổi vì quá trình cập nhật base khác nhau. Đây là giải thích cơ chế có thể xảy ra, không phải bằng chứng rằng nhánh UPT gây lỗi ngữ nghĩa ở từng ảnh.', '',
              '## Kiểm tra và khuyến nghị', '',
              'Hai protocol chỉ khác arm/output/staging/conditions. Source, cohort, receipt, targets, caption gốc và evaluator reference khớp; hai rank hoàn tất, history hữu hạn và lịch CE/SCST đúng. Kaggle preflight trong log qua; pip có cảnh báo xung đột dependency nhưng không thấy lỗi kiểm tra hoặc thiếu bước huấn luyện.', '',
              'Không mở rộng full train hoặc thêm seed chỉ để cứu cấu hình này. Giữ A làm mốc; báo kết quả cho Astra để quyết định bước chẩn đoán hoặc thiết kế tiếp. Không tự kết luận UPT vô ích hay mọi hướng căn chỉnh không khả thi. Chưa cần yêu cầu người dùng chấm thêm nhiều ảnh để xác nhận một gain chưa xuất hiện.', '',
              'Chưa thực hiện human factual review; chưa xác minh ngữ nghĩa bằng xem ảnh. CHAIR không đánh giá đúng/sai quan hệ. Review ZIP không có checkpoint; chỉ kiểm tra hash worker báo cáo.']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=Path, required=True)
    parser.add_argument('--h', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    result = audit(args.a, args.h)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'CAPTION_A_H_RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.output / 'CAPTION_A_H_RESULT.md').write_text(markdown(result), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'decision', 'comparisons', 'sensitivity', 'H_branch_gradient']}, indent=2))
