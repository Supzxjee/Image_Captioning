"""Pure protocol checks, no torch or inference on import."""
import numpy as np
from phrase_data import identity
from crop_geometry import crop_rectangle,patch_centers


def experiment_manifest(data,original):
    assert data['status']=='LABELS_LOCKED_TRAINING_GATES_PENDING'
    fit,dev,probe=data['fit_groups'],data['exposed_development_groups'],data['new_probe_groups']
    assert len(fit)==14 and len(dev)==7 and len(probe)==8
    assert [g['group_id'] for g in fit]==original['partitions']['fit']
    groups=fit+dev+probe;assert len({g['group_id'] for g in groups})==29
    assert all(g['human_confirmed'] is True for g in probe)
    sets=[{e['image_id'] for g in part for e in g['events']} for part in (fit,dev,probe)]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    assert len({e['event_id'] for g in groups for e in g['events']})==58
    m=dict(original,groups=groups,partitions=dict(fit=[g['group_id'] for g in fit],exposed_development=[g['group_id'] for g in dev],new_probe=[g['group_id'] for g in probe]),
        cross_weight=0.,human_export_sha256=data['human_export_sha256'],phase='matched_fullframe_vs_paircrop_CE',
        scope='14fit/7exposeddev/8new-to-adaptation probe; one AI-assisted human reviewer; not an independent whole-system test')
    m.pop('manifest_content_sha256',None);m['manifest_content_sha256']=identity(m)
    return m


def compare_features(new,old):
    new,old=np.asarray(new,dtype=np.float64),np.asarray(old,dtype=np.float64)
    assert new.shape==old.shape==(197,768) and np.isfinite(new).all() and np.isfinite(old).all()
    error=np.abs(new-old);relative=float(np.linalg.norm(new-old)/max(np.linalg.norm(old),1e-12))
    passed=bool(np.all(error<=.01+.005*np.abs(old)) and relative<=1e-3)
    return dict(pass_gate=passed,mean_absolute_error=float(error.mean()),max_absolute_error=float(error.max()),relative_l2=relative)


def validate_stage(path,manifest):
    events={e['event_id']:e for g in manifest['groups'] for e in g['events']}
    ids=sorted({e['image_id'] for e in events.values()});eids=sorted(events)
    with np.load(path,allow_pickle=False) as z:
        assert z['image_ids'].tolist()==ids and z['event_ids'].tolist()==eids
        for key,shape,dtype in [('full',(len(ids),197,768),np.float16),('global_memory',(len(ids),32,512),np.float32),
                               ('crop',(len(eids),196,768),np.float16),('crop_xy',(len(eids),196,2),np.float32),('full_xy',(196,2),np.float32)]:
            a=z[key];assert a.shape==shape and a.dtype==dtype and np.isfinite(a).all(),key
        assert np.array_equal(z['full_xy'],np.asarray(patch_centers([0,0,1,1],1,1),dtype=np.float32))
        for i,key in enumerate(eids):
            e=events[key];rect=crop_rectangle(e['subject_box'],e['object_box'],e['width'],e['height'])
            expected=np.asarray(patch_centers(rect,e['width'],e['height']),dtype=np.float32)
            assert np.array_equal(z['crop_xy'][i],expected),key
