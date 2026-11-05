"""Safe data/NumPy checks; no Torch or scorer imports."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from candidate_data import baseline,check_record,image_batch,choose_oracle,prepare
from real_bank_data import read,save,sha


class Checks(unittest.TestCase):
    def fixture(self):
        vocab=['PAD','BOS','EOS','UNK','a','cat']
        c=dict(caption='a cat',ids=[1,4,5,2,0],logprob=-2.)
        row=dict(coco_id=10,candidates=[dict(c,logprob=-2.-k) for k in range(5)])
        return vocab,row,dict(coco_id=10,caption='a cat',ids=[1,4,5,2],score=-2.)

    def test_replay_content_and_errors(self):
        vocab,row,old=self.fixture();self.assertEqual(check_record(row,old,vocab),0)
        for key,value in [('caption','a'),('ids',[1,4,2]),('logprob',-1.)]:
            changed=copy.deepcopy(row);changed['candidates'][0][key]=value
            with self.assertRaises(ValueError):check_record(changed,old,vocab)
        bad=copy.deepcopy(row);bad['candidates'][2]['ids']=[1,4,5,2,4]
        with self.assertRaises(ValueError):check_record(bad,old,vocab)

    def test_oracle_selection_alignment_ties_and_finite_guard(self):
        _,row,_=self.fixture();other=copy.deepcopy(row);other['coco_id']=20
        other['candidates'][1]['caption']='a different caption'
        selected,ranks,score=choose_oracle([row,other],[[1,0],[1,3],[0,2],[0,0],[0,0]])
        self.assertEqual(ranks,[0,1]);self.assertEqual(score,2)
        self.assertEqual([r['coco_id'] for r in selected],[10,20]);self.assertEqual(selected[1]['caption'],'a different caption')
        with self.assertRaises(ValueError):choose_oracle([row],[[float('nan')]]*5)

    def test_batch_same_fp32_padding_and_order(self):
        d=dict(features=np.arange(3*768).reshape(3,768).astype(np.float16),offsets=np.array([0,1,3]))
        x,v=image_batch(d,np.array([1,0]));self.assertEqual(x.dtype,np.float32)
        self.assertEqual(v.tolist(),[[True,True],[True,False]])
        np.testing.assert_array_equal(x[0],d['features'][1:].astype(np.float32))
        self.assertTrue((x[1,1]==0).all())

    def test_historical_snapshot_is_complete_and_replayable(self):
        b=baseline();self.assertEqual(len(b['order']),5000);self.assertEqual(len(set(b['order'])),5000)
        self.assertEqual([r['coco_id'] for r in b['predictions']],b['order'][::2]+b['order'][1::2])
        self.assertEqual([r['coco_id'] for r in b['references']],b['order'])
        self.assertEqual(sum(len(r['references']) for r in b['references']),25010)
        self.assertEqual(b['metrics']['counts']['hallucinated_captions'],260)
        self.assertEqual(b['metrics']['short_le2'],407)
        for old in b['predictions']:
            row=dict(coco_id=old['coco_id'],candidates=[dict(caption=old['caption'],ids=old['ids'],logprob=old['score']-k) for k in range(5)])
            check_record(row,old,b['vocabulary'])

    def test_stage_only_valid_and_verify_payloads_and_refs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';out.mkdir();stage=root/'stage';vocab=['PAD','BOS','EOS','UNK']
            save(root/'receipt.json',{});save(root/'vocabulary.json',vocab)
            entries=[dict(group='train',files={'TRAIN_MUST_NOT_BE_READ.npz':'not-present'})];refs=[]
            for k in range(5):
                ids=np.arange(k*1000,(k+1)*1000,dtype=np.int64)
                rows=[dict(coco_id=int(i),width=1,height=1,region_kinds=['predicted_roi'],sequences=[[1,3,2]+[0]*51],captions=['x']*5) for i in ids]
                j=f'val_{k:04}.json';n=f'val_{k:04}.npz';save(root/j,dict(images=rows))
                np.savez_compressed(root/n,features=np.ones((1000,768),np.float16),boxes=np.tile(np.array([0,0,1,1],np.float32),(1000,1)),
                    scores=np.ones(1000,np.float32),offsets=np.arange(1001,dtype=np.int64),ids=ids)
                entries.append(dict(group='val',files={j:sha(root/j),n:sha(root/n)}))
                refs.extend(dict(coco_id=int(i),references=['x']*5) for i in ids)
            save(root/'manifest.json',dict(shards=entries))
            package={n.name:sha(n) for n in root.iterdir() if n.is_file()}
            cfg=dict(full_root=str(root),output=str(out),stage=str(stage))
            def checked_read(path):
                return dict(files=package) if Path(path).name=='SAVE_FULL_ROI_V134_PACKAGE.json' else read(path)
            with patch('candidate_data.read',side_effect=checked_read),patch('candidate_data.baseline',return_value=dict(order=list(range(5000)),vocabulary=vocab,references=refs)):
                prepare(cfg)
                self.assertEqual(np.load(stage/'offsets.npy').tolist(),list(range(5001)))
                self.assertEqual(np.load(stage/'features.npy').shape,(5000,768))
                (root/'val_0000.json').write_text('changed',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'Changed VALID shard'):prepare(cfg)

    def test_sources_parse_and_no_training_in_worker(self):
        here=Path(__file__).parent
        for n in ('candidate_data.py','run_candidates.py','make_candidate_notebook.py'):
            tree=ast.parse((here/n).read_text(encoding='utf-8'))
            if n=='run_candidates.py':
                self.assertFalse(any(isinstance(x,ast.Attribute) and x.attr in ('backward','step','train') for x in ast.walk(tree)))

    def test_transport_pins_package_supported_entry_and_version_routing(self):
        tree=ast.parse(Path(__file__).with_name('make_candidate_notebook.py').read_text(encoding='utf-8'))
        execution=next(ast.literal_eval(x.value) for x in ast.walk(tree) if isinstance(x,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='execute' for t in x.targets))
        code=ast.parse(execution)
        command=next(x.value for x in ast.walk(code) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='command' for t in x.targets))
        self.assertEqual(ast.literal_eval(command.elts[1]),'-c')
        self.assertEqual(ast.literal_eval(command.elts[2]),'from kaggle.cli import main; main()')
        self.assertIn("'kaggle==2.2.5'",execution)
        self.assertIn("request.version_label)==('ducanh2403','2-vi-cai-kia-qua-nhieu','v161')",execution)
        self.assertIn('download.mkdir(exist_ok=True)',execution)
        self.assertLess(execution.index("'kaggle==2.2.5'"),execution.index('token=UserSecretsClient'))


if __name__=='__main__':unittest.main(verbosity=2)
