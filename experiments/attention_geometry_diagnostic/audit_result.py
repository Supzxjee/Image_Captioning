"""Audit saved user-run tensors with NumPy only; never execute supplied source or model."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from diagnostic_data import A_SHA,BASE_SHA,POLICY,aggregate,evaluate_image,save,paired_ci

HERE=Path(__file__).resolve().parent
ZIP=Path('C:/Users/leduc/Downloads/attention_geometry_diagnostic_200_review (1).zip')


def main():
    loaded=json.loads((HERE/'review/loaded_second.json').read_text(encoding='utf-8'))
    cfg=json.loads(loaded['protocol.json']);receipt=json.loads(loaded['receipt.json'])
    assert receipt['status']=='DIAGNOSTIC_COMPLETE' and not receipt['optimizer_used'] and not receipt['test_used']
    assert cfg['policy']==POLICY and json.loads(loaded['policy.json'])==POLICY
    assert cfg['input_sha256']['checkpoint']==A_SHA
    for name,expected in cfg['implementation_sha256'].items():
        assert hashlib.sha256(loaded['source/'+name].encode()).hexdigest()==expected,name
    maps={}
    # Textual payloads were read once into loaded_second; only unread binary maps are accessed here.
    with zipfile.ZipFile(ZIP) as z:
        for name,expected in receipt['payload_sha256'].items():
            if name.endswith('.npz'):
                cached=HERE/'review'/name
                raw=cached.read_bytes() if cached.is_file() else z.read(name)
                if not cached.is_file():cached.write_bytes(raw)
                with np.load(io.BytesIO(raw),allow_pickle=False) as f:maps.update({k:f[k].copy() for k in f.files})
            else:raw=loaded[name].encode()
            assert hashlib.sha256(raw).hexdigest()==expected,name
    allrows=[];max_map_error=0.;bounds=[];cls=[];rank_info=[]
    for rank in (0,1):
        complete=json.loads(loaded[f'complete_rank{rank}.json'])
        rows=json.loads(loaded[f'rows_rank{rank}.json'])
        assert complete['status']=='FROZEN_FORWARD_COMPLETE' and complete['images']==len(rows)==69
        assert complete['base_tensor_sha256']==BASE_SHA and complete['checkpoint_sha256']==A_SHA
        assert complete['parity']==[dict(memory_max=0.,logit_max=0.)]
        rank_info.append(dict(rank=rank,seconds=complete['seconds'],text_binding_max_abs=complete['text_binding_max_abs']))
        for row in rows:
            key=str(row['coco_id']);q=maps['q_full_'+key].astype('f8');a=maps['p_to_q_'+key].astype('f8')
            assert q.shape==(32,197) and a.shape[1]==32
            np.testing.assert_allclose(q.sum(-1),1,atol=1e-6,rtol=0)
            np.testing.assert_allclose(a.sum(-1),1,atol=1e-6,rtol=0)
            paths=a@q;spatial=paths[:,1:];conditioned=spatial/spatial.sum(-1,keepdims=True)
            expected=np.stack([conditioned[p['positions']].mean(0) for p in row['phrases']])
            actual=maps[key]
            max_map_error=max(max_map_error,float(np.max(np.abs(expected-actual))))
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)
            metric=evaluate_image(actual,row['phrases'],row['pairs'],key)
            for control,values in metric.items():
                for k in ('mass','excess_mass','centroid_hit','axis_accuracy'):
                    assert abs(values[k]-row['metrics'][control][k])<1e-12,(key,k)
            qpatch=q[:,1:];qcond=qpatch/qpatch.sum(-1,keepdims=True)
            local=[]
            for index,p in enumerate(row['phrases']):
                mask=np.asarray(p['mask']);area=float(mask.mean())
                oracle=float((qcond@mask).max());true=float(actual[index]@mask)
                raw_oracle=float((qpatch@mask).max());raw_true=float((spatial[p['positions']]@mask).mean())
                assert true<=oracle+1e-6 and raw_true<=raw_oracle+1e-6
                cls.append(float(paths[p['positions'],0].mean()))
                local.append(dict(category=p['category'],area=area,actual_conditional_mass=true,
                    oracle_single_Q_conditional_mass=oracle,conditional_gap=oracle-true,
                    actual_raw_spatial_box_mass=raw_true,oracle_single_Q_raw_box_mass=raw_oracle))
            bounds.append(dict(coco_id=row['coco_id'],phrases=local,
                image_mean_conditional_gap=float(np.mean([p['conditional_gap'] for p in local]))))
            allrows.append(row)
    selected=cfg['selected'];assert len(allrows)==len(selected)==len({r['coco_id'] for r in allrows})==138
    assert sorted(r['coco_id'] for r in allrows)==sorted(r['coco_id'] for r in selected)
    result=aggregate(allrows);reported=json.loads(loaded['result.json'])
    def compare(actual,expected):
        if isinstance(actual,dict):
            assert actual.keys()==expected.keys()
            for k in actual:compare(actual[k],expected[k])
        elif isinstance(actual,list):
            assert len(actual)==len(expected)
            for a,b in zip(actual,expected):compare(a,b)
        elif isinstance(actual,float):np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
        else:assert actual==expected
    compare(result,reported)
    phrases=[p for row in bounds for p in row['phrases']]
    audit=dict(status='SAVED_ARTIFACT_AUDIT_PASS',zip_sha256=hashlib.sha256(ZIP.read_bytes()).hexdigest(),
        result=result,coverage=json.loads(loaded['coverage.json']),ranks=rank_info,phrase_count=len(phrases),
        classes=len({p['category'] for p in phrases}),axis_constraints=sum(len(r['pairs']) for r in allrows),
        max_reconstructed_map_abs_error=max_map_error,
        phrase_mean_CLS_mass=float(np.mean(cls)),
        image_bootstrap_conditional_oracle_gap=paired_ci([r['image_mean_conditional_gap'] for r in bounds]),
        phrase_mean_conditional_oracle_mass=float(np.mean([p['oracle_single_Q_conditional_mass'] for p in phrases])),
        phrase_mean_raw_oracle_mass=float(np.mean([p['oracle_single_Q_raw_box_mass'] for p in phrases])),
        phrase_mean_actual_raw_mass=float(np.mean([p['actual_raw_spatial_box_mass'] for p in phrases])),
        limitations=['Restricted weakly labelled TRAIN subset, not population grounding accuracy.',
            'Oracle maximizes 32 Q maps using GT, selection-biased diagnostic ceiling, not usable inference routing or proof of semantics.',
            'Conditional patch metrics exclude CLS; raw path masses reported separately.',
            'Exact parity is one same-shape image per rank; tensor identity/text checks supported by receipts, actual original checkpoint/cache not in ZIP.',
            'Text binding uses combined rtol=.002 and atol=.002; max absolute error alone can exceed .002.',
            'No caption generation, no training, no proof of predicate understanding or novelty.'])
    save(HERE/'RESULT.json',audit);save(HERE/'review/q_capacity_by_phrase.json',bounds)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('result','coverage')},indent=2))


if __name__=='__main__':main()
