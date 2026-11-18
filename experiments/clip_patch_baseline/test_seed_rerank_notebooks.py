"""Offline AST and replay-contract tests; no Torch, GPU, credentials or network."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest import TestCase, main
from unittest.mock import patch

from make_seed_rerank_notebooks import HERE, build, code, embedded, embedded_node, function_source
from test_seed_test_notebook_routing import FakeApi, notebook_assignments, routing_source


def notebook(seed):
    return json.loads((HERE / f'CSA_PER_CAPTION_SEED{seed}_RERANK_TEST5000_API_2GPU.ipynb').read_text(encoding='utf-8'))


class Checks(TestCase):
    def test_sources_pins_and_effective_api_routing(self):
        names = ('kaggle', 'kaggle.api', 'kaggle.api.kaggle_api_extended')
        modules = {n: ModuleType(n) for n in names}
        modules[names[-1]].KaggleApi = FakeApi
        historical = json.loads((HERE / 'CLIP_PATCH_CSA_PER_CAPTION_BEST9_RERANK_TEST5000_2GPU.ipynb').read_text(encoding='utf-8'))
        for seed, checkpoint, output in ((43, 'ducanh3/notebook7bc83c4c76/179', 'ducanh3/notebook7bc83c4c76/181'),
                                         (44, 'ducanh2403/2-vi-cai-kia-qua-nhieu/236', 'ducanh2403/2-vi-cai-kia-qua-nhieu/238')):
            with self.subTest(seed=seed):
                nb = notebook(seed)
                self.assertEqual(nb, build(seed, historical))
                for cell in nb['cells']:
                    if cell['cell_type'] == 'code':
                        compile(''.join(cell['source']), 'cell', 'exec')
                for name, expected in [('SOURCE_VERSION', checkpoint), ('TEST_OUTPUT_VERSION', output), ('SEED', seed), ('WALL_HOURS', 3)]:
                    self.assertEqual(ast.literal_eval(notebook_assignments(nb, name)[0].value), expected)
                with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
                    exec(routing_source(nb), {'SOURCE_VERSION': checkpoint})
                original = json.loads((HERE / f'CSA_PER_CAPTION_SEED{seed}_TEST5000_API_2GPU.ipynb').read_text(encoding='utf-8'))
                for name in ('REVISION', 'MAPPING', 'HASHES', 'TRAIN_REVISION', 'TRAIN_PINS', 'SECRET_NAMES'):
                    self.assertEqual(ast.literal_eval(notebook_assignments(nb, name)[0].value),
                                     ast.literal_eval(notebook_assignments(original, name)[0].value))
                source = code(nb, 3)
                for filename in ('csa_per_caption_test_data.py', 'run_csa_per_caption_test.py', 'csa_per_caption_rerank_data.py', 'run_csa_per_caption_rerank.py'):
                    compile(embedded(source, filename), filename, 'exec')
                self.assertEqual(embedded(source, 'csa_per_caption_test_data.py'), embedded(code(original, 3), 'csa_per_caption_test_data.py'))
                self.assertEqual(embedded(source, 'run_csa_per_caption_rerank.py'), embedded(code(historical, 3), 'run_csa_per_caption_rerank.py'))
                worker = function_source(embedded(source, 'run_csa_per_caption_test.py'), 'worker')
                self.assertEqual(worker, function_source(embedded(code(historical, 3), 'run_csa_per_caption_test.py'), 'worker'))
                self.assertIn('beams_sha256', worker)
                self.assertNotIn('.backward(', source)
                self.assertEqual(source.count("'--max_restarts=0'"), 1)
                self.assertIn("'--file-pattern', '(^|/)best[.]bin$'", source)
                setup = code(nb, 1)
                self.assertIn(f'/kaggle/working/csa_per_caption_seed{seed}_rerank_test5000', setup)
                self.assertIn('objectdetectionecache/objectdetectioncache.json', setup)

    def test_replay_receipts_and_original_metric_guard(self):
        for seed in (43, 44):
            source = code(notebook(seed), 3)
            node = embedded_node(ast.parse(source), 'REPLAY_RECEIPTS.json')
            receipts = ast.literal_eval(node.args[0].args[0])
            checked = json.loads((HERE / f'CSA_PER_CAPTION_SEED{seed}_TEST5000_RESULT.json').read_text(encoding='utf-8'))
            self.assertEqual(receipts['checkpoint_sha256'], checked['result']['checkpoint_sha256'])
            self.assertEqual(receipts['rank_caption_sha256'], [checked[f'worker{i}']['caption_sha256'] for i in (0, 1)])
            self.assertEqual(receipts['reference_sha256'], checked['scoring_lock']['files_sha256']['test_references.json'])
            body = next(n for n in ast.parse(source).body if isinstance(n, ast.Try)).body
            i = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'expected_top1')
            block = compile(ast.Module(body=body[i:i+2], type_ignores=[]), 'metric-guard', 'exec')
            metrics = checked['result']['metrics'].copy()
            env = {'result': {'metrics': {'decoder_rank0': metrics}}}
            exec(block, env)
            metrics['CIDEr'] += .001
            with self.assertRaisesRegex(ValueError, 'metric replay differs: CIDEr'):
                exec(block, env)

    def test_replay_stops_before_detector_and_clip_on_wrong_payload(self):
        source = embedded(code(notebook(43), 3), 'csa_per_caption_rerank_data.py')
        fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
        replay = {'checkpoint_sha256': 'checkpoint', 'rank_caption_sha256': ['ok', 'ok'], 'reference_sha256': 'ok'}
        lock = {'checkpoint_sha256': 'checkpoint'}
        values = {'test_lock.json': lock, 'REPLAY_RECEIPTS.json': replay, 'RERANK_TEST_INPUTS.json': {}}

        def sha(path):
            if str(path) == 'dataset':
                raise RuntimeError('passed replay guards')
            return 'ok'

        env = {'Path': Path, 'HERE': Path('/source'), 'read': lambda path: values[Path(path).name], 'sha': sha}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'prepare', 'exec'), env)
        cfg = dict(output='/out', test_output='/test', dataset_json='dataset')
        with self.assertRaisesRegex(RuntimeError, 'passed replay guards'):
            env['prepare'](cfg)
        for target, key, bad, message in [(lock, 'checkpoint_sha256', 'wrong', 'checkpoint'),
                                         (replay, 'reference_sha256', 'wrong', 'reference bytes'),
                                         (replay, 'rank_caption_sha256', ['wrong', 'ok'], 'replay differs')]:
            previous = target[key]
            target[key] = bad
            with self.assertRaisesRegex(ValueError, message):
                env['prepare'](cfg)
            target[key] = previous

    def test_full_beam_coverage_and_corruption_guards(self):
        source = embedded(code(notebook(43), 3), 'csa_per_caption_rerank_data.py')
        fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'read_beams')
        inputs = [dict(coco_id=i) for i in range(5000)]
        values = {'inference_inputs.json': inputs, 'vocabulary.json': ['pad', 'a', 'dog']}
        for rank in (0, 1):
            ids = range(rank, 5000, 2)
            values[f'worker{rank}.json'] = dict(complete=True, images=2500, checkpoint_sha256='checkpoint', beams_sha256=f'beam{rank}')
            values[f'captions_rank{rank}.json'] = [dict(coco_id=i, ids=[1, 2], caption='a dog', score=-2.) for i in ids]
            values[f'beams_rank{rank}.json'] = [dict(coco_id=i, candidates=[dict(ids=[1, 2], caption='a dog', logprob=-2.-k) for k in range(5)]) for i in ids]
        env = {'Path': Path, 'read': lambda path: values[Path(path).name],
               'sha': lambda path: 'beam' + Path(path).stem[-1], 'text_from_ids': lambda ids, words: ' '.join(words[i] for i in ids)}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'beams', 'exec'), env)
        run = lambda: env['read_beams'](Path('/test/rerank'), {'checkpoint_sha256': 'checkpoint'})
        self.assertEqual([r['coco_id'] for r in run()], list(range(5000)))
        candidate = values['beams_rank0.json'][0]['candidates'][0]
        candidate['caption'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'Top-1'):
            run()
        candidate['caption'] = 'a dog'
        values['worker1.json']['beams_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            run()


if __name__ == '__main__':
    main(verbosity=2)
