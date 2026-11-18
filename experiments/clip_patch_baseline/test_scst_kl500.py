"""Offline source, metadata and scalar-math checks. Never imports Torch."""
import ast
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from audit_seed43_scst_valid import compare
from make_scst_kl500_notebook import build
from make_seed_rerank_notebooks import HERE, code, embedded
from make_seed_scst1500_notebooks import pilot_source
from scst_kl_valid_compare import gates
from test_seed_test_notebook_routing import FakeApi, notebook_assignments, routing_source


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = build()
        cls.main = code(cls.nb, 3)
        cls.trainer = pilot_source(cls.main)

    def test_reproducible_notebook_and_all_embedded_sources_parse(self):
        saved = json.loads((HERE/'CSA_SEED43_SCST_KL500_VALID5000_2GPU.ipynb').read_text(encoding='utf8'))
        self.assertEqual(saved, self.nb)
        count = 0
        for cell in self.nb['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), 'cell', 'exec')
        for node in ast.walk(ast.parse(self.main)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'write_text' and node.args:
                value = node.args[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, str) and '\n' in value.value:
                    compile(value.value, 'embedded source', 'exec')
                    count += 1
        self.assertGreaterEqual(count, 8)
        self.assertEqual(embedded(self.main, 'scst_kl.py'), (HERE/'scst_kl.py').read_text(encoding='utf8'))
        self.assertEqual(embedded(self.main, 'scst_kl_valid_compare.py'), (HERE/'scst_kl_valid_compare.py').read_text(encoding='utf8'))

    def test_exact_179_and_185_routing_and_download_filters(self):
        names = ('kaggle', 'kaggle.api', 'kaggle.api.kaggle_api_extended')
        modules = {name: ModuleType(name) for name in names}
        modules[names[-1]].KaggleApi = FakeApi
        for source in ('ducanh3/notebook7bc83c4c76/179', 'ducanh3/notebook7bc83c4c76/185'):
            with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
                exec(routing_source(self.nb), {'SOURCE_VERSION': source})
        tree = ast.parse(self.main)
        self.assertFalse(any(isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'CHECKPOINT_ROOT' for n in ast.walk(tree)))
        self.assertIn('api_env = os.environ.copy()', self.main)
        pattern = r'(^|/)update0500/(pilot_checkpoint[.]bin|pilot_checkpoint_receipt[.]json)$'
        self.assertTrue(re.search(pattern, 'out/update0500/pilot_checkpoint.bin'))
        self.assertFalse(re.search(pattern, 'out/update1500/pilot_checkpoint.bin'))
        self.assertIn(pattern, self.main)
        self.assertIn('8f1a1fc181b335543d8be117da1315260cdc01b4d4fd4b464d8367a82003d500', self.main)
        self.assertIn("if control_receipt!=", self.main)

    def test_fixed_training_and_teacher_isolation_guards(self):
        for text in ('UPDATES = 500', 'BATCH = 8', 'torch.manual_seed(4342+rank)',
                     'reference = copy.deepcopy(model).requires_grad_(False).eval()',
                     "loss = rl + cfg['kl_beta']*kl", 'selected_logps(logits.detach()',
                     'reference_versions != tuple(p._version', 'id(p) in optimizer_ids',
                     'torch.equal(before_teacher, torch.cuda.get_rng_state(device))'):
            self.assertIn(text, self.trainer)
        self.assertNotIn('cross_entropy', self.trainer)
        self.assertNotIn('load_state_dict(ck[\'optimizer\'])', self.trainer)
        self.assertEqual(self.trainer.count('optimizer ='), 1)
        self.assertIn('torch.optim.Adam(wrapped.parameters()', self.trainer)
        kl = ast.parse(embedded(self.main, 'scst_kl.py'))
        function = next(n for n in kl.body if isinstance(n, ast.FunctionDef) and n.name == 'reference_kl')
        source = ast.unparse(function)
        self.assertIn('reference_logits.detach()[..., 2:].float()', source)
        self.assertIn('logits[..., 2:].float()', source)
        self.assertIn('logp.exp() * (logp - logq)', source)
        self.assertIn('masked_fill(~mask, 0).sum(1).mean()', source)
        self.assertLess(self.main.index("SOURCE/'scst_kl.py','--output'"), self.main.index("[*launch,pilot,'train'"))

    def test_checkpoint_binary_and_receipt_share_beta_and_reference(self):
        tree = ast.parse(self.trainer)
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'checkpoint_step')
        saved = []
        class Destination:
            def __truediv__(self, name): return self
            def mkdir(self, **kw): pass
            def with_suffix(self, suffix): return self
        env = dict(rank=0, out=Destination(), src='own-XE', cfg={'kl_beta': 1.0},
                   model=SimpleNamespace(state_dict=lambda: {'w': 'same architecture'}),
                   optimizer=SimpleNamespace(state_dict=lambda: {'updates': 500}),
                   torch=SimpleNamespace(save=lambda data, path: saved.append(data)),
                   os=SimpleNamespace(replace=lambda *args: None),
                   dist=SimpleNamespace(barrier=lambda **kw: None),
                   save=lambda path, data: saved.append(data), sha=lambda path: 'own-XE')
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'checkpoint metadata', 'exec'), env)
        env['checkpoint_step'](500)
        self.assertEqual(len(saved), 2)
        for payload in saved:
            self.assertEqual(payload['updates'], 500)
            self.assertEqual(payload['kl_beta'], 1)
            self.assertEqual(payload['reference_sha256'], 'own-XE')
            self.assertEqual(payload['source_sha256'], 'own-XE')

    def test_valid_only_three_arms_and_cached_data(self):
        setup = code(self.nb, 1)
        self.assertIn('for p in (COCO_JSON,):', setup)
        self.assertNotIn('TEST_OUTPUT_VERSION=', setup)
        self.assertEqual(ast.literal_eval(notebook_assignments(self.nb, 'WALL_HOURS')[0].value), 3)
        self.assertIn('materialize_part0(', self.main)
        self.assertIn('linked_patches=links', self.main)
        for name in ('xe', 'scst500', 'kl500'):
            self.assertIn("evaluate_arm('"+name+"'", self.main)
        self.assertEqual(self.main.count("[*launch,pilot,'train'"), 1)
        data = embedded(self.main, 'csa_per_caption_test_data.py')
        self.assertIn('module.configure_valid', embedded(self.main, 'run_csa_per_caption_test.py'))
        self.assertIn("r['split'] == 'val'", data)
        self.assertIn("r['split'] == 'val'", embedded(self.main, 'csa_per_caption_rerank_data.py'))
        self.assertNotIn('receipt[\'updates\'] != 1500', embedded(self.main, 'csa_per_caption_rerank_data.py'))
        self.assertNotIn('run_scst_resume1500.py', self.main)
        self.assertIn('no_automatic_extension=True', embedded(self.main, 'scst_kl_valid_compare.py'))

    def test_gate_requires_both_caption_gain_and_lower_hallucination(self):
        xe = dict(CIDEr=1.2, CHAIRs=.04, CHAIRi=.03)
        control = dict(CIDEr=1.204, CHAIRs=.045, CHAIRi=.032)
        good = dict(CIDEr=1.203, CHAIRs=.039, CHAIRi=.029)
        self.assertTrue(all(gates(xe, control, good).values()))
        self.assertFalse(all(gates(xe, control, dict(good, CHAIRs=.041)).values()))
        self.assertFalse(all(gates(xe, control, dict(good, CHAIRi=.031)).values()))
        self.assertFalse(all(gates(xe, control, dict(good, CIDEr=1.201)).values()))

    def test_scalar_forward_kl_derivative_and_mask_scaling(self):
        # Independent scalar oracle for the equation, not an autograd/CUDA test.
        def softmax(z):
            ex = [math.exp(v-max(z)) for v in z]
            return [x/sum(ex) for x in ex]
        def kl(z, q):
            return sum(p*math.log(p/r) for p, r in zip(softmax(z), q))
        z, q = [.2, -.7, 1.1], softmax([-.1, .4, .3])
        p = softmax(z)
        self.assertAlmostEqual(kl(z, p), 0)
        self.assertGreater(kl(z, q), 0)
        for j in range(3):
            plus, minus = z.copy(), z.copy()
            plus[j] += 1e-5; minus[j] -= 1e-5
            numerical = (kl(plus,q)-kl(minus,q))/2e-5
            analytic = p[j]*(math.log(p[j]/q[j])-kl(z,q))
            self.assertAlmostEqual(numerical, analytic, places=8)

    def test_audit_transitions_and_reference_mismatch_rejection(self):
        def row(i, flag):
            return dict(image_id=i, caption=str(i), mscoco_gt_words=['person'],
                        mscoco_hallucinated_words=[['dog','dog']] if flag else [])
        xe = dict(sentences=[row(i, i in (1, 2)) for i in range(5000)])
        scst = dict(sentences=[row(i, i in (2, 3)) for i in range(5000)])
        cider = {i: .5 for i in range(5000)}
        result = compare(xe, scst, cider, cider)
        self.assertEqual(result['transition_counts'], dict(new_flags=1, repairs=1, shared_flags=1, clean_both=4997))
        self.assertEqual(result, compare(xe, scst, cider, cider))
        wrong = copy.deepcopy(scst); wrong['sentences'][4]['mscoco_gt_words'] = ['car']
        with self.assertRaises(ValueError): compare(xe, wrong, cider, cider)
        wrong = copy.deepcopy(scst); wrong['sentences'].append(wrong['sentences'][0])
        with self.assertRaises(ValueError): compare(xe, wrong, cider, cider)


if __name__ == '__main__':
    unittest.main(verbosity=2)
