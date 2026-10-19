"""Audit artifacts only: never execute archived Python or train/infer a model."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile
import numpy as np
from pilot_data import cohorts, identity


def audit(source, dataset_path, output):
    here = Path(__file__).parent
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None
        read = lambda n: json.loads(z.read(n))
        sha = lambda n: hashlib.sha256(z.read(n)).hexdigest()
        cfg, summary = read('protocol.json'), read('run_summary.json')
        arm = cfg['arm']
        assert summary['status'] == 'PILOT_COMPLETE' and arm == summary['arm'] and arm in 'ABCD'
        assert not cfg['test_used'] and not summary['test_used']
        assert summary['protocol_sha256'] == sha('protocol.json')
        assert cfg['groups'] == read('groups.json')
        assert identity(cfg['groups']) == summary['cohort_sha256'] == cfg['cohort_sha256']
        assert cfg['cohort_sha256'] == '6bbdf3b75869927334ec6c2f868edf9f52346c0e79401aa0e118ba5a02fb711b'
        dataset_bytes = Path(dataset_path).read_bytes()
        assert hashlib.sha256(dataset_bytes).hexdigest() == cfg['expected_hashes']['dataset_json']
        dataset = json.loads(dataset_bytes)
        locked = cohorts(dataset)
        for split in ('train','val'):
            assert [{k:v for k,v in r.items() if k!='mode'} for r in cfg['groups'][split]] == locked[split]
            assert dict(Counter(r['mode'] for r in cfg['groups'][split])) == cfg['counts_by_mode'][split]
        assert (cfg['global_batch_size'],cfg['warmup_updates'],cfg['base_lr'],cfg['scst_weight']) == (8,200,5e-6,.05)
        for name, expected in cfg['embedded_source_sha256'].items():
            assert sha('source/'+name) == expected
            assert hashlib.sha256((here/name).read_text(encoding='utf-8').encode()).hexdigest() == expected
        targets = read('train_object_targets.json')
        assert sha('train_object_targets.json') == cfg['object_targets_sha256']
        assert targets['metadata']['parser_version'] == 'aliases_v2'
        assert targets['metadata']['penalty_version'] == 'unique_count_v2'
        assert not targets['metadata']['heldout_reward_labels_exported']
        assert [{k:r[k] for k in ('image_id','coco_id','filename','split','mode')} for r in targets['data']] == cfg['groups']['train']
        histories, completions = [], []
        for rank in (0,1):
            done, runtime, hist = (read(f'{name}_rank{rank}.json') for name in ('worker_complete','runtime','history'))
            assert done['status']=='COMPLETE' and done['updates']==len(hist)==1250
            assert done['source_arm']==arm and done['rank']==rank
            assert (done['trainable_branch_parameters']==0) if arm=='A' else (done['trainable_branch_parameters']>0)
            assert runtime['gpu']['name']=='Tesla T4' and runtime['deterministic_algorithms'] and not runtime['warn_only']
            assert runtime['sdp_math'] and not runtime['sdp_flash'] and not runtime['sdp_mem_efficient']
            for step,r in enumerate(hist,1):
                assert r['step']==step and r['global_images_seen']==step*8
                assert all(math.isfinite(v) for v in r.values() if isinstance(v,(float,int)))
                weight=0. if step<=200 else .05
                assert r['scst_weight']==weight
                if arm=='A': assert r['branch_grad_norm']==0
                assert abs(r['loss']-r['ce_loss']-weight*r['scst_loss'])<1e-6
                if weight:
                    for p in ('sample','greedy'):
                        assert abs(r[p+'_reward']-r[p+'_cider']-.5*r[p+'_object'])<1e-6
            histories.append(hist); completions.append(done)
        assert completions[0]['initial_base_sha256']==completions[1]['initial_base_sha256']
        assert all(a['grad_norm']==b['grad_norm'] for a,b in zip(*histories))
        setup=z.read('setup.log').decode(errors='replace')
        for smoke_arm in 'ABCD': assert 'Two-GPU synchronized CE/SCST smoke PASS: '+smoke_arm in setup
        assert 'Ran 4 tests' in setup and 'CIDEr reward, official caption metrics and CHAIR smoke PASS' in setup
        expected={r['image_id']:r for r in cfg['groups']['val']}
        gt=read('val_gt.json')
        assert {r['id'] for r in gt['images']}==expected.keys()
        for row in cfg['groups']['val']:
            assert [r['caption'] for r in gt['annotations'] if r['image_id']==row['image_id']] == [s['raw'] for s in dataset['images'][row['image_id']]['sentences'][:5]]
        metrics, captions, chairs, per_image = {}, {}, {}, {}
        for condition in (('origin','on') if arm=='A' else ('origin','on','off')):
            rows=read(condition+'_predictions.json')
            assert len(rows)==1000 and {r['image_id'] for r in rows}==expected.keys()
            merged=sum((read(f'{condition}_rank{rank}.json') for rank in (0,1)),[])
            assert sorted(merged,key=lambda r:r['image_id'])==rows
            diagnostic=sum((read(f'{condition}_diagnostic_rank{rank}.json') for rank in (0,1)),[])
            assert len(diagnostic)==1000 and {r['image_id'] for r in diagnostic}==expected.keys()
            assert all(math.isfinite(r['reference_nll_per_token']) for r in diagnostic)
            stored=read(condition+'_metrics.json')
            assert stored==summary['metrics'][condition]['caption']
            per=read(condition+'_per_image_cider.json')
            assert len(per)==1000 and [r['image_id'] for r in per]==[r['image_id'] for r in rows]
            scores=np.asarray([r['CIDEr'] for r in per])
            assert np.isfinite(scores).all() and abs(scores.mean()-stored['CIDEr'])<1e-12
            for mode,m in summary['metrics'][condition]['by_mode'].items():
                indices=[i for i,r in enumerate(rows) if expected[r['image_id']]['mode']==mode]
                assert len(indices)==m['images'] and abs(scores[indices].mean()-m['CIDEr_full_val_IDF'])<1e-12
            c=read(condition+'_chair_output.json');sentences=c['sentences']
            by_coco={r['image_id']:r for r in sentences}
            assert len(sentences)==len(by_coco)==1000 and by_coco.keys()=={r['coco_id'] for r in expected.values()}
            assert all(by_coco[expected[r['image_id']]['coco_id']]['caption']==r['caption'] for r in rows)
            bad=sum(bool(r['mscoco_hallucinated_words']) for r in sentences)
            hallucinated=sum(len(r['mscoco_hallucinated_words']) for r in sentences)
            mentions=sum(len(r['mscoco_generated_words']) for r in sentences)
            correct=sum(len(set(r['mscoco_generated_words'])&set(r['mscoco_gt_words'])) for r in sentences)
            total=sum(len(set(r['mscoco_gt_words'])) for r in sentences)
            chair=dict(CHAIRs=bad/1000,CHAIRi=hallucinated/mentions,Recall=correct/total)
            assert all(abs(chair[k]-c['overall_metrics'][k])<1e-12 and abs(chair[k]-summary['metrics'][condition]['chair'][k])<1e-12 for k in chair)
            metrics[condition]=dict(caption=stored,chair=chair,hallucinated_captions=bad,hallucinated_mentions=hallucinated,
                                    object_mentions=mentions,correct_unique_objects=correct,reference_unique_objects=total,
                                    by_mode=summary['metrics'][condition]['by_mode'])
            captions[condition]={r['image_id']:r['caption'] for r in rows};chairs[condition]=by_coco;per_image[condition]=scores
        delta=per_image['on']-per_image['origin'];rng=np.random.default_rng(20261019)
        ci=np.percentile([delta[rng.integers(0,1000,1000)].mean() for _ in range(5000)],[2.5,97.5]).tolist()
        a,b=chairs['origin'],chairs['on']
        pair=dict(CIDEr_delta=float(delta.mean()),CIDEr_relative_percent=float(100*delta.mean()/per_image['origin'].mean()),
                  paired_fixed_IDF_bootstrap95=ci,bootstrap_source='Saved per-image CIDEr, not independent retokenization or seed uncertainty',
                  changed_captions=sum(captions['origin'][i]!=captions['on'][i] for i in expected),
                  hallucination_removed=sum(bool(a[i]['mscoco_hallucinated_words']) and not b[i]['mscoco_hallucinated_words'] for i in a),
                  hallucination_introduced=sum(bool(b[i]['mscoco_hallucinated_words']) and not a[i]['mscoco_hallucinated_words'] for i in a))
        result=dict(status='ARTIFACT_AUDIT_PASS',arm=arm,source_zip_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                    source_bytes=Path(source).stat().st_size,metrics=metrics,paired=pair,cohort_sha256=cfg['cohort_sha256'],
                    source_checkpoint_sha256=cfg['source_checkpoint_sha256'],initial_base_sha256=completions[0]['initial_base_sha256'],
                    end_checkpoint_sha256=summary['checkpoint_sha256'],training_updates=1250,global_images_seen=10000,
                    train_seconds_by_rank=[h[-1]['seconds'] for h in histories],test_used=False,
                    checks=['ZIP CRC, source/protocol/target/cohort hashes','Karpathy split and reference captions replayed from local pinned dataset',
                            'Both complete finite1250-update histories; loss/reward arithmetic; equal clipped-input grad norms across ranks',
                            'Recorded GPU/model/DDP/metric smoke tests passed; new run not launched by auditor',
                            '1000 unique val IDs and matching rank outputs/CHAIR captions; saved per-image CIDEr aggregation and CHAIR reaggregation'],
                    limits=['No checkpoint tensors or staged HDF5 in ZIP: cannot replay inference or verify end-checkpoint bytes/actual final cross-rank weights.',
                            'Caption metrics were stored by the official evaluator on Kaggle; no independent local tokenizer/CIDEr/BLEU/METEOR/ROUGE recomputation.',
                            'CHAIR parser outputs reaggregated, not independently retokenized; bootstrap uses saved fixed-IDF per-image scores.',
                            'One seed and reused val1000, raw greedy; not comparable directly to old beam/reranked test metrics.',
                            'Matched pilot only: compare other arms and origin/cohort/runtime before attribution; sensitivity is not verified grounding.'])
    Path(output).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(metrics=metrics,paired=pair),indent=2,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.source,a.dataset,a.output)
