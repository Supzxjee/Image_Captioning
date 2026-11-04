"""Author-safe full1 coverage, selection and evaluation-resume tests."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from real_bank_data import sha
from scst_full_data import full_schedule,local_batch,image_events,selected_name,copy_evaluations,validate_state,UPDATES,MID
from scst_pilot_data import event_schedule,study_updates,seen_events


class Checks(unittest.TestCase):
    def test_full_exact_coverage_and_last_seven(self):
        pool,ids=full_schedule(range(113287));_,again=full_schedule(reversed(range(113287)))
        np.testing.assert_array_equal(ids,again)
        self.assertEqual(ids.shape,(14161,8));self.assertEqual(int((ids<0).sum()),1)
        self.assertEqual(len(set(ids[ids>=0])),113287);self.assertEqual(set(pool),set(range(113287)))
        data=dict(ids=np.arange(113288),group=np.array([0]*113287+[1]),caption_image=np.repeat(np.arange(113288),5))
        events=event_schedule(data,ids)
        self.assertEqual(events[-1,-1],-1)
        np.testing.assert_array_equal(data['ids'][data['caption_image'][events[events>=0]]],ids[ids>=0])
        left,lid=local_batch(events,ids,UPDATES-1,0);right,rid=local_batch(events,ids,UPDATES-1,1)
        self.assertEqual((len(left),len(right)),(4,3));self.assertFalse(set(lid)&set(rid))
        self.assertEqual(image_events(MID),56648);self.assertEqual(image_events(UPDATES),113287)
        self.assertEqual(seen_events(dict(study='full1'),UPDATES),113287)
        for wrong in ([],[1]*113287):
            with self.assertRaises(ValueError):full_schedule(wrong)

    def test_horizon_and_unequal_weighting(self):
        self.assertEqual(study_updates(dict(study='full1',updates=14161)),14161)
        with self.assertRaises(ValueError):study_updates(dict(updates=14161))
        with self.assertRaises(ValueError):study_updates(dict(study='full1',updates=2500))
        # DDP averages rank gradients: world*n_local/n_global restores the true image mean.
        values=np.arange(7.)
        actual=(values[:4].mean()*(2*4/7)+values[4:].mean()*(2*3/7))/2
        self.assertAlmostEqual(actual,values.mean())

    def test_selection_fallback_ties_and_recall(self):
        base=dict(CIDEr=1.,CHAIRs=.06,CHAIRi=.04,Recall=.4)
        metrics=dict(original=base,mid={**base,'CIDEr':1.1},end={**base,'CIDEr':1.1})
        self.assertEqual(selected_name(metrics)[0],'mid')
        metrics['end']['CIDEr']=1.2;metrics['end']['Recall']=.39
        self.assertEqual(selected_name(metrics)[0],'mid')
        metrics['mid']['CHAIRs']=.07
        self.assertEqual(selected_name(metrics),('original',['original']))
        metrics['mid']={**base,'CIDEr':float('nan')}
        with self.assertRaises(ValueError):selected_name(metrics)

    def test_phase_state_and_resume_hashes(self):
        empty=dict(updates=0,image_events=0,validated=[],validation_files={});validate_state(empty)
        with self.assertRaises(ValueError):validate_state({**empty,'updates':1,'image_events':8})
        original={f'evaluations/original/{n}':'x' for n in ('val_rank0.json','val_rank1.json','worker0.json','worker1.json')}
        middle=dict(updates=MID,image_events=56648,validated=['original'],validation_files=original)
        validate_state(middle)
        with self.assertRaises(ValueError):validate_state({**middle,'updates':MID+1,'image_events':56656})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'old';out=Path(d)/'new';f=root/'evaluations/original/val_rank0.json'
            f.parent.mkdir(parents=True);f.write_text('[]',encoding='utf-8')
            name='evaluations/original/val_rank0.json';files={name:sha(f)}
            copy_evaluations(root,out,files);self.assertEqual((out/name).read_bytes(),f.read_bytes())
            f.write_text('changed',encoding='utf-8')
            with self.assertRaises(ValueError):copy_evaluations(root,out,files)
            with self.assertRaises(ValueError):copy_evaluations(root,out,{'../escape':'x'})

    def test_full_scoring_selection_orchestration_without_scorers(self):
        import json
        from scst_full_data import score,PHASE_STEPS
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);original=out/'original.bin';original.write_bytes(b'original')
            files={}
            for phase in PHASE_STEPS:
                folder=out/'evaluations'/phase;folder.mkdir(parents=True)
                for n in ('val_rank0.json','val_rank1.json','worker0.json','worker1.json'):
                    p=folder/n;p.write_text('[]',encoding='utf-8');files[str(p.relative_to(out)).replace('\\','/')]=sha(p)
            state=dict(updates=UPDATES,image_events=113287,validated=list(PHASE_STEPS),validation_files=files)
            (out/'progress.json').write_text(json.dumps(state),encoding='utf-8')
            for n in ('validation_references.json','vocabulary.json'):(out/n).write_text('[]',encoding='utf-8')
            for n,step in [('mid',MID),('checkpoint',UPDATES)]:
                p=out/(n+'.bin');p.write_bytes(n.encode())
                (out/(n+'_receipt.json')).write_text(json.dumps(dict(sha256=sha(p),updates=step,image_events=image_events(step))),encoding='utf-8')
            def fake_score(cfg):
                path=Path(cfg['output']);phase=path.name
                value={'original':1.,'mid':1.2,'end':1.1}[phase]
                (path/'result.json').write_text(json.dumps(dict(metrics=dict(CIDEr=value,CHAIRs=.05,CHAIRi=.03,Recall=.4))),encoding='utf-8')
            with patch('run_scst_pilot.score',side_effect=fake_score),patch('scst_pilot_data.exact_checkpoint',return_value=original),patch('scst_pilot_data.CHECKPOINT_SHA',sha(original)),patch('builtins.print'):
                score(dict(output=str(out),checkpoint_root=str(out),study='full1',updates=UPDATES,arm='SCST_CE'))
            self.assertEqual((out/'best.bin').read_bytes(),b'mid')
            result=json.loads((out/'result.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'],'CSA_SCST_FULL1_COMPLETE');self.assertEqual(result['selection']['selected_name'],'mid')

    def test_packaged_sources_parse(self):
        here=Path(__file__).parent
        for n in ('scst_full_data.py','test_scst_full_model.py','run_scst_pilot.py','make_scst_pilot_notebooks.py'):
            ast.parse((here/n).read_text(encoding='utf-8'),filename=n)


if __name__=='__main__':unittest.main()
