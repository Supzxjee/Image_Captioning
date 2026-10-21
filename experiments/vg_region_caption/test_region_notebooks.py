"""Offline checks only: embedded source/data integrity and matched G/T artifacts."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import unittest
import zlib
from region_data import identity, patch_mask

HERE=Path(__file__).resolve().parent


def assignment(tree,name):
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))


def embedded(notebook):
    source=''.join(notebook['cells'][2]['source'])
    tree=ast.parse(source)
    hashes=assignment(tree,'SOURCE_HASHES')
    packed=next(n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
    bundle=json.loads(zlib.decompress(base64.b64decode(packed)))
    return tree,hashes,bundle


class Checks(unittest.TestCase):
    def test_portable_unexecuted_matched_notebooks_and_source_hashes(self):
        bundles=[]
        for arm in ('G','T'):
            file=HERE/f'VG_REGION_CAPTION_{arm}_2GPU.ipynb'
            self.assertLess(file.stat().st_size,1_000_000)
            notebook=json.loads(file.read_text(encoding='utf-8'))
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    ast.parse(''.join(cell['source']))
                    self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])
            tree,hashes,bundle=embedded(notebook);bundles.append(bundle)
            self.assertEqual(len(assignment(tree,'PROJECT_COMMIT')),40)
            for name,source in bundle.items():
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),hashes[name])
                if name.endswith('.py'):ast.parse(source)
            config=''.join(notebook['cells'][1]['source'])
            self.assertIn(f"ARM='{arm}'",config)
            self.assertNotIn('/upt-hidden-pairs-caption11k/upt_hidden_pairs_11k',config)
            smoke=''.join(notebook['cells'][3]['source'])
            self.assertIn('test_region_ddp.py',smoke);self.assertIn('--nproc_per_node=2',smoke)
            self.assertIn('run_region.py',''.join(notebook['cells'][4]['source']))
        self.assertEqual(bundles[0],bundles[1])

    def test_payload_train_only_eos_no_small_boxes_no_review_label_selection(self):
        raw=zlib.decompress((HERE/'auxiliary_payload.bin').read_bytes())
        payload=json.loads(raw)
        report=json.loads((HERE/'AUXILIARY_MANIFEST.json').read_text())
        self.assertEqual(hashlib.sha256(raw).hexdigest(),report['auxiliary_payload_sha256'])
        self.assertEqual(identity(payload['schedule']),report['schedule_sha256'])
        self.assertEqual(len(payload['schedule']),10000)
        allowed=set(payload['train_image_ids'])-set(payload['exclusions'])
        self.assertTrue(set(payload['viewed_audit_image_ids'])<=set(payload['exclusions']))
        for row in payload['bank']:
            self.assertIn(row['coco_id'],allowed)
            self.assertGreaterEqual(sum(patch_mask(row['region_normalized'])),4)
            self.assertLessEqual(len(row['encoded']),30)
            self.assertEqual(row['encoded'][0],1);self.assertEqual(row['encoded'][-1],2)
            self.assertNotIn(3,row['encoded'])
        self.assertEqual({r['coco_id'] for r in payload['bank']}.__len__(),report['scheduled_unique_images'])

    def test_caption_loop_math_matches_pinned_A_source(self):
        import subprocess
        root=HERE.parents[1]
        original=subprocess.check_output(['git','show','dfa1d4e:experiments/upt_alignment_pilot/run_pilot.py'],cwd=root,text=True,encoding='utf-8')
        current=(root/'experiments/upt_alignment_pilot/run_pilot.py').read_text(encoding='utf-8')
        addition="        if hasattr(model, 'training_diagnostics'):\n            record.update(model.training_diagnostics())\n"
        self.assertEqual(current.replace(addition,''),original)


if __name__=='__main__':unittest.main(verbosity=2)
