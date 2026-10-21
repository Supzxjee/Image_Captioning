"""Offline notebook/policy equivalence checks. No torch or notebook execution."""
import ast
import json
from pathlib import Path
import unittest
from make_visual_only_notebook import embedded_sources

HERE=Path(__file__).resolve().parent


class Checks(unittest.TestCase):
    def test_same_G_data_runtime_and_caption_path_with_only_new_GV_sources(self):
        g=json.loads((HERE/'VG_REGION_CAPTION_G_2GPU.ipynb').read_text(encoding='utf-8'))
        path=HERE/'VG_REGION_CAPTION_GV_2GPU.ipynb'
        self.assertLess(path.stat().st_size,1_000_000)
        gv=json.loads(path.read_text(encoding='utf-8'))
        original=embedded_sources(g);current=embedded_sources(gv)
        self.assertEqual({k:v for k,v in current.items() if k in original},original)
        self.assertEqual(set(current)-set(original),{'visual_only_model.py','run_visual_only.py',
            'test_visual_only_model.py','test_visual_only_ddp.py'})
        for name,source in current.items():
            if name.endswith('.py'):ast.parse(source)
        for cell in gv['cells']:
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']))
                self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])
        first=''.join(gv['cells'][1]['source']);smoke=''.join(gv['cells'][3]['source'])
        self.assertIn("ARM='GV'",first)
        self.assertIn('/vg_region_caption_GV',first)
        self.assertIn('test_visual_only_model.py',smoke);self.assertIn('test_visual_only_ddp.py',smoke)
        self.assertIn('--nproc_per_node=2',smoke)
        self.assertIn('run_visual_only.py',''.join(gv['cells'][4]['source']))

if __name__=='__main__':unittest.main(verbosity=2)
