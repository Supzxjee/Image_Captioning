"""Model-free geometry/sampling/resource checks. GPU checks run only in Kaggle."""
import ast
import base64
import hashlib
import json
import contextlib
import io
import tempfile
import zipfile
import zlib
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from tile_data import make_tiles,tile_centers,select_sample,estimates,PAYLOAD_PER_IMAGE


class Checks(unittest.TestCase):
    def test_224_identity_control(self):
        pixels=np.random.default_rng(42).integers(0,256,(224,224,3),dtype=np.uint8);image=Image.fromarray(pixels)
        for h,l in zip(make_tiles(image,'H'),make_tiles(image,'L')):np.testing.assert_array_equal(np.asarray(h),np.asarray(l))

    def test_odd_size_and_condition_validation(self):
        pixels=np.zeros((257,513,3),dtype=np.uint8)
        pixels[:128,:256]=[255,0,0];pixels[:128,256:]=[0,255,0];pixels[128:,:256]=[0,0,255];pixels[128:,256:]=[255,255,0]
        for condition in ('H','L'):
            tiles=make_tiles(Image.fromarray(pixels),condition)
            self.assertEqual([t.size for t in tiles],[(224,224)]*4)
            np.testing.assert_array_equal([np.asarray(t)[112,112] for t in tiles],[[255,0,0],[0,255,0],[0,0,255],[255,255,0]])
        with self.assertRaises(ValueError):make_tiles(Image.fromarray(pixels),'unknown')

    def test_coordinates_follow_quadrant_and_patch_order(self):
        c=tile_centers();self.assertEqual(c.shape,(196,2));self.assertEqual(len(np.unique(c,axis=0)),196)
        np.testing.assert_allclose(c[0],[1/28,1/28]);np.testing.assert_allclose(c[49],[.5+1/28,1/28])
        np.testing.assert_allclose(c[98],[1/28,.5+1/28]);np.testing.assert_allclose(c[-1],[1-1/28,1-1/28])
        self.assertTrue(((c>0)&(c<1)).all())

    def test_sampling_disjoint_and_order_independent(self):
        rows=[dict(coco_id=i,image_id=i) for i in range(10000)];a=select_sample(rows);b=select_sample(list(reversed(rows)))
        self.assertEqual(a,b);self.assertEqual(len(a['warmup']),16);self.assertEqual(len(a['measure']),256)
        self.assertFalse({r['coco_id'] for r in a['warmup']}&{r['coco_id'] for r in a['measure']})

    def test_estimates_use_concurrent_wall_and_include_both_conditions(self):
        w=[dict(status='COMPLETE',measured_images=128,measure_start_ns=1000000000,measure_end_ns=11000000000),
           dict(status='COMPLETE',measured_images=128,measure_start_ns=2000000000,measure_end_ns=14000000000)]
        r=estimates(w);self.assertEqual(r['measured_wall_seconds'],13);self.assertEqual(r['full_tensor_payload_bytes'],6021120000)
        self.assertEqual(PAYLOAD_PER_IMAGE,401408)
        with self.assertRaises(ValueError):estimates(w[:1])

    def test_locked_archive_and_portable_notebook(self):
        root=Path(__file__).resolve().parent
        archive=root.parent/'noun_evidence_proposal/data/noun_evidence_train10k_v1_dataset.zip'
        if archive.exists():
            from tile_data import ARCHIVE_SHA
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),ARCHIVE_SHA)
            with zipfile.ZipFile(archive) as z:groups=json.loads(z.read('groups.json'))
            sample=select_sample(groups['train']);self.assertEqual(len(groups['val']),5000)
            self.assertFalse({r['coco_id'] for r in sample['measure']}&{r['coco_id'] for r in groups['val']})
        notebook=root/'NATIVE_DETAIL_BENCHMARK_256_2GPU.ipynb'
        if notebook.exists():
            self.assertLess(notebook.stat().st_size,1000000)
            cells=json.loads(notebook.read_text(encoding='utf-8'))['cells']
            for cell in cells:
                if cell['cell_type']!='code':continue
                self.assertEqual(cell['outputs'],[]);self.assertIsNone(cell['execution_count'])
                tree=ast.parse(''.join(cell['source']))
                for node in tree.body:
                    if not isinstance(node,ast.Assign):continue
                    if any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in node.targets):
                        payload=next(n.value for n in ast.walk(node.value) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>1000)
                        embedded=json.loads(zlib.decompress(base64.b64decode(payload)))
                        for name,source in embedded.items():
                            path=root/name if '/' not in name else root.parents[1]/name
                            self.assertEqual(source,path.read_text(encoding='utf-8'),name)

    def test_failed_input_archives_diagnostics_and_preserves_existing_output(self):
        from run_benchmark import launch
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root=Path(temporary);bad=root/'bad.bin';bad.write_bytes(b'invalid checkpoint')
            out=root/'benchmark'
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):launch(dict(output=str(out),checkpoint=str(bad)))
            result=json.loads((out/'benchmark_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'],'FAILED');self.assertFalse(result['trained'])
            with zipfile.ZipFile(str(out)+'_review.zip') as z:self.assertIn('benchmark_summary.json',z.namelist())
            original=(out/'benchmark_summary.json').read_bytes()
            with self.assertRaises(FileExistsError):launch(dict(output=str(out)))
            self.assertEqual((out/'benchmark_summary.json').read_bytes(),original)

    def test_source_and_notebook_syntax(self):
        root=Path(__file__).resolve().parent
        for f in root.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'),filename=str(f))


if __name__=='__main__':unittest.main(verbosity=2)
