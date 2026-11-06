"""Author-safe NumPy/metadata tests; no Torch/model/scorer/API execution."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from prompt_scst_data import SOURCES,PHASES,EVAL_NAMES,source_checkpoint,check_source,validate_state,copy_evaluations,validate_candidates
from scst_full_data import full_schedule,local_batch,image_events,UPDATES
from real_bank_data import sha,save
from prompt_scst_rerank import fixed_rank,validation_image_paths,validation_detections,prepare_images


class Checks(unittest.TestCase):
    def test_full_schedule_unique_no_heldout_final_4_3(self):
        _,ids=full_schedule(range(113287));_,again=full_schedule(list(reversed(range(113287))))
        np.testing.assert_array_equal(ids,again);self.assertEqual(ids.shape,(14161,8))
        self.assertEqual(len(np.unique(ids[ids>=0])),113287);self.assertEqual(int((ids<0).sum()),1)
        for rank,n in ((0,4),(1,3)):
            ev,co=local_batch(ids,ids,UPDATES-1,rank);self.assertEqual(len(ev),n);np.testing.assert_array_equal(ev,co)
        self.assertEqual(image_events(UPDATES),113287)

    def fixture(self,arm):
        epoch=SOURCES[arm]['epoch']
        return dict(identity=dict(schema='csa_prompt_scratch_xe10_v1',arm=arm,epochs=10,stage_sha256={'a':'b'},
            prompt_files_sha256={'c':'d'},feature_sharing=True,common_init_sha256='d6dad13dc69bf5fe9355e525e5240b518082d897ca82a53db4f71e37966c9791'),
            state=dict(updates=epoch*17711,caption_events=epoch*566747,best_epoch=epoch),model={'model.base.nouns':None})

    def test_selected_source_identity_and_cross_arm_rejection(self):
        for arm in SOURCES:
            ck=self.fixture(arm);check_source(ck,arm,{'a':'b'},{'c':'d'})
            for key in ('updates','caption_events','best_epoch'):
                wrong=copy.deepcopy(ck);wrong['state'][key]+=1
                with self.assertRaises(ValueError):check_source(wrong,arm,{'a':'b'},{'c':'d'})
            with self.assertRaises(ValueError):check_source(ck,'PROMPT' if arm=='CONTROL' else 'CONTROL',{'a':'b'},{'c':'d'})
            with self.assertRaises(ValueError):check_source(ck,arm,{'a':'changed'},{'c':'d'})

    def test_exact_best_sha_accepts_alias_and_rejects_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'best.bin').write_bytes(b'selected')
            expected=sha(root/'best.bin')
            with patch.dict(SOURCES,{'CONTROL':dict(SOURCES['CONTROL'],sha256=expected)}):
                self.assertEqual(source_checkpoint(root,'CONTROL'),root/'best.bin')
                (root/'best.bin').write_bytes(b'changed')
                with self.assertRaises(ValueError):source_checkpoint(root,'CONTROL')

    def test_resume_phases_cannot_skip_original_or_end_early(self):
        state=dict(updates=0,image_events=0,validated=[],validation_files={});validate_state(state)
        state.update(updates=1,image_events=8)
        with self.assertRaises(ValueError):validate_state(state)
        state.update(validated=['original'],validation_files={f'evaluations/original/{n}':'h' for n in EVAL_NAMES});validate_state(state)
        state['validated'].append('end')
        with self.assertRaises(ValueError):validate_state(state)
        state.update(updates=UPDATES,image_events=113287,validation_files={f'evaluations/{p}/{n}':'h' for p in PHASES for n in EVAL_NAMES});validate_state(state)

    def test_resume_evaluation_copies_replay_and_detects_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'in';out=Path(folder)/'out';(root/'evaluations/original').mkdir(parents=True)
            files={}
            for n in EVAL_NAMES:
                p=root/'evaluations/original'/n;save(p,{'sealed':True});files['evaluations/original/'+n]=sha(p)
            copy_evaluations(root,out,files);self.assertTrue((out/'evaluations/original/cider_replay.json').is_file())
            (root/'evaluations/original/cider_replay.json').write_text('{}')
            with self.assertRaises(ValueError):copy_evaluations(root,out,files)
            with self.assertRaises(ValueError):copy_evaluations(root,out,{'../oops':'h'})

    def test_five_beams_finite_order_and_fixed_selector_ties(self):
        row=dict(coco_id=1,candidates=[dict(caption='a dog',logprob=-float(i)) for i in range(5)])
        validate_candidates([row],{1});self.assertEqual(fixed_rank(row,[1.]*5,{'dog':.9}),0)
        row['candidates'][0]['logprob']=float('nan')
        with self.assertRaises(ValueError):validate_candidates([row],{1})
        with self.assertRaises(ValueError):fixed_rank(row,[float('nan')]*5,{})

    def test_jpeg_single_scan_nested_missing_and_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'images/val2014').mkdir(parents=True);(root/'images/val2014/a.jpg').write_bytes(b'fixture')
            records=[dict(filename='a.jpg')];found=validation_image_paths(root,records)
            self.assertEqual(found['a.jpg'],(root/'images/val2014/a.jpg').resolve())
            with self.assertRaises(FileNotFoundError):validation_image_paths(root,[dict(filename='missing.jpg')])
            (root/'a.jpg').write_bytes(b'duplicate')
            with self.assertRaisesRegex(ValueError,'Ambiguous'):validation_image_paths(root,records)

    def test_filtered_detector_preserves_selected_confidences(self):
        from captioning.object_consistency import index_detections
        source={'nested/a.jpg':[dict(label='dog',conf=.9),dict(label='cat',conf=.3)],'other.jpg':[dict(label='person',conf=.8)]}
        selected=validation_detections(source,{'a.jpg'})
        self.assertEqual(selected,{'a.jpg':index_detections(source)['a.jpg']})
        with self.assertRaises(ValueError):validation_detections(source,{'missing.jpg'})
        with self.assertRaises(ValueError):validation_detections({**source,'other/a.jpg':[]},{'a.jpg'})

    def test_parallel_jpeg_hash_order_and_corruption_guard(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths={}
            for name in ('a.jpg','b.jpg'):
                path=root/name;Image.new('RGB',(2,2)).save(path);paths[name]=path
            val={2:dict(filename='b.jpg'),1:dict(filename='a.jpg')};detected={'a.jpg':{'dog':.9},'b.jpg':{}}
            rows=prepare_images(val,paths,detected,workers=2)
            self.assertEqual([r['coco_id'] for r in rows],[1,2]);self.assertEqual(rows[0]['image_sha256'],sha(paths['a.jpg']))
            paths['b.jpg'].write_bytes(b'not an image')
            with self.assertRaises(OSError):prepare_images(val,paths,detected,workers=2)

    def test_sources_parse_and_scope_guards(self):
        here=Path(__file__).parent
        for name in ('prompt_scst_data.py','run_prompt_scst.py','prompt_scst_rerank.py','make_prompt_scst_notebooks.py'):
            ast.parse((here/name).read_text(encoding='utf-8'))
        text=(here/'run_prompt_scst.py').read_text(encoding='utf-8')
        for required in ('check_source(','strict=True','worker_seconds','resume_root','checkpoint()','ORIGINAL CIDEr replay differs','2*len(ev)/images'):
            self.assertIn(required,text)
        self.assertNotIn('nn.init.zeros_',text)
        self.assertNotIn('TrainingObjectReward',text)
        rerank=(here/'prompt_scst_rerank.py').read_text(encoding='utf-8');self.assertNotIn('Blip',rerank)
        self.assertIn('truncation=False',rerank);self.assertIn('same', (here/'PROMPT_SCST_PROTOCOL.md').read_text(encoding='utf-8').lower())
        generator=(here/'make_prompt_scst_notebooks.py').read_text(encoding='utf-8')
        self.assertIn("'prepare','--config',CONFIG],timeout=1800)",generator)
        self.assertIn("'_full1_v2'",generator);self.assertIn('No SCST checkpoint saved yet',generator)


if __name__=='__main__':unittest.main(verbosity=2)
