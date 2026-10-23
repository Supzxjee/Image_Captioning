"""Model-free regression tests. GPU integration is deliberately run only on Kaggle."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import unittest
import zipfile
import numpy as np
from evidence_data import labels,lexicon,validate_package,normalize,A_SHA

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy=json.loads((ROOT/'experiments/coco_grounding_feasibility/MAPPING_POLICY_V11.json').read_text())
        objects=runpy.run_path(str(ROOT/'captioning/object_consistency.py'));cls.aliases=objects['ALIASES'];cls.classes=list(objects['COCO_LABELS'])

    def test_compound_annotation_override_and_weak_negative(self):
        y,o=labels(['book'],['a hot dog and a toy horse next to a book'],self.aliases,self.policy,self.classes)
        for c in ('hot dog','book'):self.assertEqual((y[self.classes.index(c)],o[self.classes.index(c)]),(1,1))
        self.assertEqual((y[self.classes.index('horse')],o[self.classes.index('horse')]),(0,0))
        self.assertEqual((y[self.classes.index('dog')],o[self.classes.index('dog')]),(0,1))
        self.assertEqual((y[self.classes.index('elephant')],o[self.classes.index('elephant')]),(0,1))

    def test_raw_vocabulary_punctuation_and_compound_guard(self):
        vocab={'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3,'dog':4,'dog.':5,'hot':6,'book':7}
        table=lexicon(vocab,self.aliases,self.policy,self.classes)
        self.assertEqual(table['candidate_ids'],[4,5]);self.assertEqual(table['candidate_classes'],[16,16])
        guard=next(g for g in table['guards'] if g['compound']==['hot','dog'])
        self.assertEqual(guard['columns'],[0,1]);self.assertEqual(guard['prefix_ids_reversed'],[[6]])

    def test_sealed_data_and_prior(self):
        manifest=json.loads((HERE/'DATA_PACKAGE.json').read_text());archive=HERE/'data/noun_evidence_train10k_v1_dataset.zip'
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),manifest['archive_sha256'])
        with zipfile.ZipFile(archive) as z:
            self.assertEqual(len(z.namelist()),5)
            payload={Path(n).stem:json.loads(z.read(n)) for n in z.namelist()}
            for name,sha in payload['receipt']['payload_sha256'].items():self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),sha)
            targets=payload['targets'];groups=payload['groups'];self.assertEqual(len(groups['val']),5000)
            self.assertEqual(set(targets),{str(r['image_id']) for r in groups['train']})
            keys=[(v['coco_id'],r['annotation_id'],r['class_id']) for v in targets.values() for r in v['regions']]
            self.assertEqual(len(keys),5602);self.assertEqual(len(set(keys)),5602)
            y=np.array([r['y'] for r in targets.values()]);o=np.array([r['observed'] for r in targets.values()])
            np.testing.assert_array_equal((y.sum(0)+.5)/(o.sum(0)+1),payload['prior']['probability'])
            self.assertEqual(int(o.sum())*2,1597744)
            self.assertEqual(len(A_SHA),64)

    def test_syntax_and_notebook_integrity(self):
        for path in HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
        for path in HERE.glob('NOUN_EVIDENCE_*_2GPU.ipynb'):
            self.assertLess(path.stat().st_size,1000000)
            notebook=json.loads(path.read_text(encoding='utf-8'))
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[]);ast.parse(''.join(cell['source']))
            self.assertIn('/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin',path.read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main(verbosity=2)
