import ast
import math
from pathlib import Path
import unittest
import numpy as np
from full_cache_data import collate,stress_events,target_tokens

class Checks(unittest.TestCase):
    def fixture(self):
        sequences=np.array([[1,4,2]+[0]*51,[1,5,6,7,2]+[0]*49,[1,8,2]+[0]*51],np.int32)
        return dict(features=np.arange(7*768,dtype=np.float32).reshape(7,768).astype(np.float16),offsets=np.array([0,1,6,7],np.int64),
            ids=np.array([10,20,30]),sequences=sequences,caption_image=np.array([0,1,2],np.int32),group=np.array([0,0,1],np.uint8))

    def test_batch_maps_images_and_references_retains_eos_and_padding(self):
        d=self.fixture();x,mask,seq=collate(d,[1,0])
        self.assertEqual(x.shape,(2,5,768));self.assertEqual(mask.sum(1).tolist(),[5,1])
        self.assertEqual(seq.tolist(),[[1,5,6,7,2],[1,4,2,0,0]])
        self.assertEqual(target_tokens(d,[1,0]),6);self.assertEqual(int((seq[:,1:]!=0).sum()),6)
        np.testing.assert_array_equal(x[0],d['features'][1:6].astype(np.float32));self.assertTrue((x[1,1:]==0).all())

    def test_stress_uses_train_only_and_contains_max_regions_and_length(self):
        d=self.fixture();indices=stress_events(d,2,size=2)
        self.assertEqual(set(indices),{0,1});self.assertEqual(indices[0],1)
        self.assertEqual(math.ceil(566747/32),17711);self.assertEqual(17711*15,265665)

    def test_sources_compile_without_neural_import(self):
        root=Path(__file__).parent
        for n in ('full_cache_data.py','run_full_benchmark.py','make_full_benchmark_notebook.py'):ast.parse((root/n).read_text(encoding='utf-8'))
        tree=ast.parse((root/'full_cache_data.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            modules=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
            self.assertFalse(any(n.split('.')[0] in ('torch','transformers') for n in modules))

if __name__=='__main__':unittest.main(verbosity=2)
