"""Local data/packaging checks only. GPU checks remain user-run in the notebooks."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest
import zipfile
import zlib
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'experiments/upt_alignment_pilot'))
from routing_data import tensors_for


class Checks(unittest.TestCase):
    def test_multitoken_and_empty_image_mapping(self):
        masks={'a':dict(patch_area_fraction=np.ones((14,14)).tolist())}
        rows=[dict(decoder_logit_positions=[3,4],caption_index=4,category='person',mask_key='a')]
        value=tensors_for(rows,masks,3,2)
        self.assertEqual(value['span_caption'].tolist(),[4,0,0])
        self.assertEqual(value['span_token_valid'].tolist(),[[True,True],[False,False],[False,False]])
        np.testing.assert_array_equal(value['span_positions'][0],[3,4])
        self.assertEqual(value['span_masks'].shape,(3,196))
        self.assertFalse(tensors_for([],masks,3,2)['span_valid'].any())

    def test_sealed_dataset(self):
        ready=json.loads((HERE/'FULL_DATA_READY.json').read_text(encoding='utf-8'))
        archive=HERE/'review/coco_noun_box_train10k_v11_dataset.zip'
        if not archive.exists():self.skipTest('Raw local dataset intentionally not in Git.')
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),ready['dataset_archive_sha256'])
        with zipfile.ZipFile(archive) as z:
            self.assertIsNone(z.testzip());self.assertEqual(len(z.namelist()),7)
            self.assertTrue(all(Path(n).name==n for n in z.namelist()))
            receipt=json.loads(z.read('manifest_receipt.json'))
            for name,expected in receipt['payload_sha256'].items():
                self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),expected)
            self.assertEqual(receipt['retained_spans'],17904)
            self.assertEqual(receipt['person_spans'],4476)

    def test_notebook_sources_and_locked_recipe(self):
        bundles=[]
        for arm in ('B','R'):
            path=HERE/f'COCO_ROUTING_{arm}_2GPU.ipynb'
            self.assertLess(path.stat().st_size,1_000_000)
            notebook=json.loads(path.read_text(encoding='utf-8'))
            texts=[]
            for cell in notebook['cells']:
                if cell['cell_type']!='code':continue
                self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])
                text=''.join(cell['source']);ast.parse(text);texts.append(text)
            combined='\n'.join(texts)
            packed=re.search(r"b64decode\('([^']+)'\)",combined).group(1)
            bundle=json.loads(zlib.decompress(base64.b64decode(packed)))
            for name,source in bundle.items():
                ast.parse(source,filename=name)
                if (HERE/name).exists():self.assertEqual(source,(HERE/name).read_text(encoding='utf-8'))
            for item in ("warmup_updates=200","base_lr=5e-6","aux_weight=.05","updates=1250",
                         "--nproc_per_node=2","('preflight','worker')","nltk==3.9.2"):
                self.assertIn(item,combined)
            self.assertNotIn('HIDDEN_CACHE=',combined)
            bundles.append(bundle)
        self.assertEqual(*bundles)


if __name__=='__main__':unittest.main(verbosity=2)
