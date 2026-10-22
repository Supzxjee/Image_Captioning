"""Packaging/data checks only; these tests never import torch or execute notebook cells."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
import zlib
import numpy as np
from audit import geometry

HERE=Path(__file__).resolve().parent


class Checks(unittest.TestCase):
    def test_models_pin_matches_git_blob(self):
        root=HERE.parents[1]
        blob=subprocess.check_output(['git','show','6acd3c9f0553d4d12d46684bbe7dcb8c38033d6b:captioning/models.py'],cwd=root)
        local=(root/'captioning/models.py').read_text(encoding='utf-8').encode()
        self.assertEqual(hashlib.sha256(blob).hexdigest(),hashlib.sha256(local).hexdigest())

    def test_embedded_sources_and_unexecuted_cells(self):
        raw=(HERE/'COCO_ROUTING_INTERFACE_SMOKE_2GPU.ipynb').read_bytes()
        self.assertLess(len(raw),1_000_000)
        nb=json.loads(raw);bundles=[]
        for cell in nb['cells']:
            if cell['cell_type']!='code':continue
            self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])
            tree=ast.parse(''.join(cell['source']))
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode':
                    bundles.append(json.loads(zlib.decompress(base64.b64decode(ast.literal_eval(node.args[0])))))
        self.assertEqual(len(bundles),1)
        bundle=bundles[0]
        for name in ('route_probe.py','run_route_smoke.py'):
            self.assertEqual(bundle[name],(HERE/name).read_text(encoding='utf-8'))
            ast.parse(bundle[name])
        samples=json.loads(bundle['train4_masks.json'])
        self.assertEqual([r['coco_id'] for r in samples],[64627,529017,468966,580507])
        for row in samples:
            g=geometry(row['bbox_xywh'],*row['image_size'],include_mask=True)
            np.testing.assert_allclose(row['patch_area_fraction'],g['patch_area_fraction'],rtol=0,atol=0)
            self.assertEqual(row['target_token_positions'],[p+1 for p in row['decoder_logit_positions']])
        self.assertTrue(all(Path(name).name==name for name in bundle))

    def test_no_optimizer_or_model_execution_in_builder(self):
        for name in ('route_probe.py','run_route_smoke.py'):
            tree=ast.parse((HERE/name).read_text(encoding='utf-8'))
            calls=[n.func for n in ast.walk(tree) if isinstance(n,ast.Call)]
            self.assertFalse(any(isinstance(f,ast.Attribute) and f.attr in ('step','Adam','AdamW','SGD') for f in calls))
        text=(HERE/'make_smoke_notebook.py').read_text(encoding='utf-8')
        self.assertNotIn('import torch',text)


if __name__=='__main__':unittest.main(verbosity=2)
