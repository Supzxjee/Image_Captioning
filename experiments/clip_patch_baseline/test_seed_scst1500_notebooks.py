"""Offline metadata, source and phase-isolation checks; no Torch or NumPy."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from make_seed_scst1500_notebooks import HERE, build, code, load, pilot_source
from test_seed_test_notebook_routing import FakeApi, notebook_assignments, routing_source


def sources(nb, name):
    matches = []
    for n in ast.walk(ast.parse(code(nb, 3))):
        if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Attribute) or n.func.attr != 'write_text' or not n.args:
            continue
        value = n.func.value
        if isinstance(value, ast.BinOp) and isinstance(value.right, ast.Constant) and value.right.value == name:
            matches.append(ast.literal_eval(n.args[0]))
    return matches


def metadata_env(source, read, save):
    namespace = dict(Path=Path, json=json, __file__='/code/csa_per_caption_test_data.py', read=read, save=save,
                     sha=lambda path: 'metadata', shared=SimpleNamespace(DATA_SHA='metadata', lock=lambda: dict(
                         beam_width=5, max_new_tokens=20, length_normalization=False, reranking=False,
                         expected_test_ids=[999999], test_id_sha256='test cohort')))
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.Assign, ast.FunctionDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'metadata contracts', 'exec'), namespace)
    return namespace


class Checks(TestCase):
    def test_compile_routing_and_dataset_configuration(self):
        names = ('kaggle', 'kaggle.api', 'kaggle.api.kaggle_api_extended')
        modules = {name: ModuleType(name) for name in names}
        modules[names[-1]].KaggleApi = FakeApi
        for seed, source in ((43, 'ducanh3/notebook7bc83c4c76/179'),
                             (44, 'ducanh2403/2-vi-cai-kia-qua-nhieu/236')):
            with self.subTest(seed=seed):
                nb = load(f'CSA_PER_CAPTION_SEED{seed}_SCST1500_TEST_RERANK_2GPU.ipynb')
                self.assertEqual(nb, build(seed))
                for cell in nb['cells']:
                    if cell['cell_type'] == 'code':
                        compile(''.join(cell['source']), 'cell', 'exec')
                for node in ast.walk(ast.parse(code(nb, 3))):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'write_text' and node.args:
                        value = node.args[0]
                        if isinstance(value, ast.Constant) and isinstance(value.value, str) and '\n' in value.value:
                            compile(value.value, 'embedded Python', 'exec')
                self.assertEqual(ast.literal_eval(notebook_assignments(nb, 'SOURCE_VERSION')[0].value), source)
                self.assertEqual(ast.literal_eval(notebook_assignments(nb, 'WALL_HOURS')[0].value), 4)
                self.assertEqual(ast.literal_eval(notebook_assignments(nb, 'CACHE_ROOT')[0].value),
                                 '/kaggle/input/datasets/ducanh3/csa-scst-full-karpathy-cache-v1')
                with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
                    exec(routing_source(nb), {'SOURCE_VERSION': source})
                self.assertIn(f'csa_per_caption_seed{seed}_scst1500_test_rerank', code(nb, 1))

    def test_training_recipe_and_optimizer_resume_unchanged_except_rng(self):
        fresh = pilot_source(code(load('CLIP_PATCH_CSA_SCST_ONLY1000_MERGED_DATASET_2GPU.ipynb'), 3))
        resume = pilot_source(code(load('CLIP_PATCH_CSA_SCST_RESUME1500_RERANK_VALID5000_2GPU.ipynb'), 3))
        for seed in (43, 44):
            nb = load(f'CSA_PER_CAPTION_SEED{seed}_SCST1500_TEST_RERANK_2GPU.ipynb')
            first = pilot_source(code(nb, 3))
            second, = sources(nb, 'run_scst_resume1500.py')
            self.assertEqual(first, fresh.replace('4242+rank', f'{seed*100+42}+rank'))
            expected = resume.replace('4242+rank', f'{seed*100+42}+rank').replace('5242+rank', f'{seed*100+1042}+rank')
            self.assertEqual(second, expected)
            self.assertIn('loss = rl', first)
            self.assertNotIn('cross_entropy', first)
            self.assertIn("optimizer.load_state_dict(ck['optimizer'])", second)
            self.assertIn('range(START_UPDATE, UPDATES)', second)
            self.assertIn('checkpoint_step(step+1)', first)
            for filename in ('scst_cache.py', 'scst_dataset_mount.py'):
                original = sources(load('CLIP_PATCH_CSA_SCST_ONLY1000_MERGED_DATASET_2GPU.ipynb'), filename)
                self.assertEqual(sources(nb, filename), original)

    def test_cache_binding_preserves_data_and_rejects_foreign_cohort(self):
        for seed in (43, 44):
            valid, _ = sources(load(f'CSA_PER_CAPTION_SEED{seed}_SCST1500_TEST_RERANK_2GPU.ipynb'), 'csa_per_caption_test_data.py')
            data = {'images': [dict(cocoid=i, split='val') for i in range(5000)] + [dict(cocoid=999999, split='test')]}
            store = {'dataset': data}
            read = lambda path: copy.deepcopy(store[str(path)])
            save = lambda path, value: store.__setitem__(str(path), copy.deepcopy(value))
            env = metadata_env(valid, read, save)
            cfg = dict(dataset_json='dataset', output='/out', checkpoint_root='/weights/best.bin')
            env['configure_valid'](cfg)
            expected = env['lock']()
            self.assertEqual(expected['source_version'], 'ducanh3/notebook7bc83c4c76/179' if seed == 43 else 'ducanh2403/2-vi-cai-kia-qua-nhieu/236')
            self.assertEqual(expected['epoch'], 8 if seed == 43 else 7)
            cached = dict(expected, checkpoint_sha256='old42', stage_receipt_sha256='same-data-hash')
            store[str(Path('/out')/'test_lock.json')] = cached
            env['bind_verified_cache'](cfg, 'old42')
            rebound = store[str(Path('/out')/'test_lock.json')]
            self.assertEqual(rebound['checkpoint_sha256'], env['CHECKPOINT_SHA'])
            self.assertEqual(rebound['stage_receipt_sha256'], 'same-data-hash')
            self.assertEqual(rebound['expected_test_ids'], list(range(5000)))
            for key, value in [('checkpoint_sha256', 'foreign'), ('expected_test_ids', [1, 2])]:
                wrong = copy.deepcopy(cached)
                wrong[key] = value
                store[str(Path('/out')/'test_lock.json')] = wrong
                with self.assertRaises(ValueError):
                    env['bind_verified_cache'](cfg, 'old42')

    def test_test_checkpoint_lineage_and_phase_isolation(self):
        for seed in (43, 44):
            nb = load(f'CSA_PER_CAPTION_SEED{seed}_SCST1500_TEST_RERANK_2GPU.ipynb')
            valid, test = sources(nb, 'csa_per_caption_test_data.py')
            lineage = dict(sha256='new1500', parent_sha256='own1000', updates=1500)
            env = metadata_env(test, lambda path: lineage, lambda *args: None)
            ck = dict(updates=1500, source_sha256=env['XE_SOURCE_SHA'], parent_sha256='own1000', model={'core.nouns': [], 'core.verbs': []})
            self.assertEqual(env['validate_checkpoint_state'](ck)['checkpoint_sha256'], 'new1500')
            self.assertEqual(env['lock']()['expected_test_ids'], [999999])
            policy = env['runtime_policy']('2.11.0+cu130', '2.11.0+cu130')
            self.assertEqual(policy['training_torch'], '2.11.0+cu130')
            self.assertFalse(policy['bitwise_cross_build_equivalence_established'])
            with self.assertRaises(RuntimeError):
                env['runtime_policy']('2.11.0+cpu', '2.11.0+cu128')
            self.assertIn('configure_valid', valid)
            self.assertNotIn('configure_valid', test)
            _, test_runner = sources(nb, 'run_csa_per_caption_test.py')
            contracts = next(n for n in ast.parse(test_runner).body if isinstance(n, ast.FunctionDef) and n.name == 'contracts')
            self.assertNotIn('configure_valid', ast.unparse(contracts))
            self.assertIn("d.LINEAGE['training_torch']", test_runner)
            for key, bad in [('updates', 1499), ('source_sha256', 'other XE'), ('parent_sha256', 'other1000')]:
                wrong = copy.deepcopy(ck)
                wrong[key] = bad
                with self.assertRaises(ValueError):
                    env['validate_checkpoint_state'](wrong)
            main_source = code(nb, 3)
            self.assertIn("SOURCE=TRAIN_SOURCE/'test_code'", main_source)
            self.assertIn("OUTPUT=ROOT_OUTPUT/'test'", main_source)
            self.assertIn("continuation=ROOT_OUTPUT/'continuation'", main_source)
            self.assertIn("cfg=dict(semantic=True,prompt_test_root=str(PROMPT_TEST_ROOT)", main_source)
            self.assertIn("receipt['parent_sha256']!=first_receipt['sha256']", main_source)
            self.assertIn("ROOT_OUTPUT/'comparison.json'", main_source)
            self.assertEqual(main_source.count("'--max_restarts=0'"), 1)


if __name__ == '__main__':
    main(verbosity=2)
