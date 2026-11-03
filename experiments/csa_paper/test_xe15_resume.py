"""Pure metadata/NumPy/static checks; no Torch/model/scorer execution."""
import ast
import copy
import json
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from full_xe_data import learning_rate,validate_state,STEPS,TRAIN_EVENTS
from xe15_resume_data import check_resume_identity,resolve_resume_root
from real_bank_data import HERE,read


class Checks(unittest.TestCase):
    def fixture(self,root):
        source=root/'input';source.mkdir();receipts=root/'pinned';receipts.mkdir()
        package={}
        for n,raw in (('checkpoint.bin',b'latest139-fixture'),('best.bin',b'best9-fixture')):
            (source/n).write_bytes(raw);package[n]=hashlib.sha256(raw).hexdigest()
        for n,key in (('checkpoint_receipt.json','checkpoint.bin'),('best_receipt.json','best.bin')):
            (receipts/n).write_text(json.dumps(dict(sha256=package[key],updates=177110)),encoding='utf-8')
        return source,receipts,package

    def test_missing_receipts_recover_only_exact139_binary_fixtures(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root=Path(directory);source,receipts,package=self.fixture(root);destination=root/'verified'
            # Windows fixture substitutes copying tiny bytes for Linux read-only symlinks.
            with patch.object(Path,'symlink_to',lambda self,target:shutil.copyfile(target,self)):
                actual=resolve_resume_root(source,destination,package,receipts)
            self.assertEqual(actual,destination)
            self.assertEqual((destination/'checkpoint.bin').read_bytes(),(source/'checkpoint.bin').read_bytes())
            self.assertFalse((source/'checkpoint_receipt.json').exists())
            self.assertEqual(read(destination/'checkpoint_receipt.json')['sha256'],package['checkpoint.bin'])

    def test_duplicate_identical_receipts_and_directory_are_safe(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root=Path(directory);source,receipts,package=self.fixture(root)
            for n in ('checkpoint_receipt.json','best_receipt.json'):shutil.copy2(receipts/n,source/n)
            shutil.copytree(source,root/'duplicate');(root/'checkpoint_receipt.json').mkdir()
            actual=resolve_resume_root(root,root/'verified',package,receipts)
            self.assertIn(actual,(source,root/'duplicate'))

    def test_no_rollback_when_non139_latest_present_without_valid_receipts(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root=Path(directory);source,receipts,package=self.fixture(root)
            (source/'later').mkdir();(source/'later/checkpoint.bin').write_bytes(b'later-progress')
            with self.assertRaisesRegex(ValueError,'non139 latest'):resolve_resume_root(source,root/'verified',package,receipts)

    def state(self,epoch=11,cursor=0,phase='train'):
        return dict(epoch=epoch,batch_cursor=cursor,phase=phase,updates=(epoch-1)*STEPS+cursor,
            caption_events=(epoch-1)*TRAIN_EVENTS+min(cursor*32,TRAIN_EVENTS),history=[dict(epoch=i) for i in range(1,epoch)])

    def identities(self):
        old=read(HERE/'full_xe10_review_text/input_identity.json');new=copy.deepcopy(old)
        new.update(epochs=15,schema='csa_full_xe15_v1');new['source']['run_full_xe.py']='approved-new-runner';new['source']['full_xe_data.py']='approved-horizon'
        return old,new,read(HERE/'SAVE_XE139_PACKAGE.json')['files']['checkpoint.bin']

    def test_horizon_counts_and_no_warmup_restart(self):
        self.assertEqual((5*STEPS,15*STEPS,5*TRAIN_EVENTS,15*TRAIN_EVENTS),(88555,265665,2833735,8501205))
        self.assertAlmostEqual(learning_rate(177111,total_epochs=15),1e-4)
        self.assertAlmostEqual(learning_rate(265665,total_epochs=15),1e-4)
        with self.assertRaises(ValueError):learning_rate(177111)
        with self.assertRaises(ValueError):learning_rate(265666,total_epochs=15)

    def test_exact_latest139_migration_and_current_resume(self):
        old,new,digest=self.identities()
        self.assertEqual(check_resume_identity(old,new,digest,self.state()),'exact139-latest10-to15')
        self.assertEqual(check_resume_identity(new,new,'current-receipt-hash',self.state(13,1000)),'identical-current-run')
        validate_state(self.state(16),total_epochs=15)
        with self.assertRaises(ValueError):validate_state(self.state(16,1),total_epochs=15)

    def test_reject_wrong_weights_or_changed_inputs(self):
        old,new,digest=self.identities()
        with self.assertRaises(ValueError):check_resume_identity(old,new,'wrong-sha',self.state())
        for key in ('software','stage_sha256','global_batch','bank_package','cuda'):
            changed=copy.deepcopy(new);changed[key]='changed'
            with self.assertRaises(ValueError):check_resume_identity(old,changed,digest,self.state())
        changed=copy.deepcopy(new);changed['source']['model.py']='changed'
        with self.assertRaises(ValueError):check_resume_identity(old,changed,digest,self.state())
        with self.assertRaises(ValueError):check_resume_identity(old,new,digest,self.state(10))

    def test_static_runtime_contracts(self):
        for n in ('run_full_xe.py','full_xe_data.py','xe15_resume_data.py','score_xe15_quality.py','make_xe15_notebook.py'):
            ast.parse((HERE/n).read_text(encoding='utf-8'))
        s=(HERE/'run_full_xe.py').read_text(encoding='utf-8')
        self.assertIn("args.target_epochs==15 and not args.resume_root",s)
        self.assertIn("optimizer.load_state_dict(ck['optimizer'])",s)
        self.assertIn("torch.cuda.set_rng_state(rng['cuda'],device)",s)
        self.assertIn("while state['epoch']<=args.target_epochs",s)
        self.assertIn('beam_search(base,vo,vr,v,beams=5,maximum=20)',s)

    def test_notebook_unexecuted(self):
        p=HERE/'CSA_FULL_XE_11_TO_15_2GPU.ipynb'
        if not p.exists():self.skipTest('Generate after core commit')
        nb=json.loads(p.read_text(encoding='utf-8'))
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(c['source']);self.assertIsNone(c['execution_count']);self.assertEqual(c['outputs'],[])


if __name__=='__main__':unittest.main(verbosity=2)
