"""Safe byte/metadata/NumPy tests for locked TEST; no model/scorer execution."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from real_bank_data import read,save,sha
from rerank_test_data import resolve_inputs,fixed_rank,read_beams,load_features,checked_lock


class Checks(unittest.TestCase):
    def test_lock_rejects_test_time_policy_changes(self):
        expected=read(Path(__file__).with_name('RERANK_TEST_INPUTS.json'))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);lock=dict(checkpoint_sha256=expected['checkpoint_sha256'],selector=expected['frozen_selector'],clip=expected['clip'],
                images=5000,beam=5,max_new_tokens=20,cache_sha256=expected['test_receipt']['files'])
            save(root/'test_lock.json',lock);self.assertEqual(checked_lock(root),lock)
            lock['selector']['clip_weight']=.5;save(root/'test_lock.json',lock)
            with self.assertRaises(ValueError):checked_lock(root)

    def test_exact_checkpoint_and_cache_hash_no_version_substitution(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'best.bin').write_bytes(b'checkpoint161fixture')
            for n in ('test_regions.npz','test_metadata.json','vocabulary.json'):(root/n).write_bytes(n.encode())
            locked=dict(checkpoint_sha256=sha(root/'best.bin'),test_receipt=dict(files={n:sha(root/n) for n in ('test_regions.npz','test_metadata.json','vocabulary.json')}))
            cfg=dict(checkpoint_root=str(root),test_root=str(root));p,files=resolve_inputs(cfg,locked)
            self.assertEqual(p,root/'best.bin');self.assertEqual(len(files),3)
            (root/'best.bin').write_bytes(b'wrong')
            with self.assertRaises(ValueError):resolve_inputs(cfg,locked)

    def test_feature_loader_hash_and_no_reference_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);np.savez(root/'test_regions.npz',features=np.ones((2,768),np.float16),offsets=np.array([0,1,2],np.int64),ids=np.array([5,6],np.int64))
            save(root/'vocabulary.json',['PAD','BOS','EOS','UNK','dog']);save(root/'test_metadata.json',dict(captions='not consumed by loader'))
            lock=dict(cache_files={n:str(root/n) for n in ('test_regions.npz','test_metadata.json','vocabulary.json')})
            lock['cache_sha256']={n:sha(p) for n,p in lock['cache_files'].items()}
            a,v=load_features(lock);self.assertEqual(set(a),{'features','offsets','ids'});self.assertEqual(v[4],'dog')
            (root/'test_metadata.json').write_text('changed')
            with self.assertRaises(ValueError):load_features(lock)

    def test_fixed_policy_reference_independent_and_earlier_ties(self):
        r=dict(candidates=[dict(caption=c,logprob=-1.-i) for i,c in enumerate(['a dog','a cat','a bird','a bear','a horse'])])
        scores=[.1,2.,.3,.2,.1];self.assertEqual(fixed_rank(r,scores,{'cat':.95}),1)
        r.update(references=['a dog'],oracle_rank=4,CIDEr=[100,0,0,0,0])
        self.assertEqual(fixed_rank(r,scores,{'cat':.95}),1)
        r['candidates']=[dict(caption='a dog',logprob=-1.) for _ in range(5)]
        self.assertEqual(fixed_rank(r,[1.]*5,{'dog':.9}),0)
        with self.assertRaises(ValueError):fixed_rank(r,[float('nan')]*5,{})

    def test_beam_full_coverage_receipt_and_malformed_padding_guard(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);save(out/'inference_inputs.json',[dict(coco_id=i) for i in range(5000)])
            save(out/'vocabulary.json',['PAD','BOS','EOS','UNK','dog'])
            lock=dict(cache_files={'vocabulary.json':str(out/'vocabulary.json')},checkpoint_sha256='fixture')
            for rank in (0,1):
                path=out/f'beams_rank{rank}.jsonl'
                with path.open('w') as f:
                    for i in range(rank,5000,2):f.write(json.dumps(dict(coco_id=i,candidates=[dict(ids=[1,4,2],caption='dog',logprob=-1.-k) for k in range(5)]))+'\n')
                save(out/f'decode_worker{rank}.json',dict(complete=True,images=2500,checkpoint_sha256='fixture',beams_sha256=sha(path)))
            rows=read_beams(out,lock);self.assertEqual([r['coco_id'] for r in rows],list(range(5000)))
            path=out/'beams_rank0.jsonl';lines=path.read_text().splitlines();r=json.loads(lines[0]);r['candidates'][1]['ids']=[1,4,2,4];lines[0]=json.dumps(r);path.write_text('\n'.join(lines)+'\n')
            w=read(out/'decode_worker0.json');w['beams_sha256']=sha(path);save(out/'decode_worker0.json',w)
            with self.assertRaises(ValueError):read_beams(out,lock)

    def test_locked_protocol_metadata_and_static_no_training(self):
        here=Path(__file__).parent;locked=read(here/'RERANK_TEST_INPUTS.json')
        self.assertTrue(locked['checkpoint_version'].endswith('/161'));self.assertTrue(locked['test_cache_version'].endswith('/143'))
        self.assertEqual(locked['frozen_selector']['clip_weight'],.4);self.assertEqual(locked['frozen_selector']['object_weight'],.3)
        self.assertEqual(locked['test_receipt']['images'],5000)
        for n in ('rerank_test_data.py','run_rerank_test.py','make_rerank_test_notebook.py'):
            tree=ast.parse((here/n).read_text(encoding='utf-8'))
            if n=='run_rerank_test.py':
                for function in [x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('decode','worker')]:
                    self.assertFalse(any(isinstance(x,ast.Attribute) and x.attr in ('backward','step','train') for x in ast.walk(function)))
                    for x in ast.walk(function):
                        if isinstance(x,ast.Constant) and isinstance(x.value,str):self.assertFalse(any(k in x.value.lower() for k in ('reference','annotation','cider','oracle')))


if __name__=='__main__':unittest.main(verbosity=2)
