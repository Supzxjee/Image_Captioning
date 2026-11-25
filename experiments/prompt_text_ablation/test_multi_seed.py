"""Reuse zero-text compile/closure/mock guards; never import Torch/models."""
import ast
import hashlib
import json
import unittest

import test_notebook as prior


class Seed43Checks(prior.Checks):
    seed = 43

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.nb = json.loads((prior.HERE/f'FULL_ZERO_PROMPT_TEXT_SEED{cls.seed}_XE10_2GPU.ipynb').read_text(encoding='utf-8'))
        cls.setup = ''.join(cls.nb['cells'][1]['source'])
        # Assignments are AST-rewritten with spaces; avoid textual splitting on '='.
        cls.header = cls.setup[:cls.setup.index('SOURCE_PAYLOADS')]
        cls.values = prior.literals(cls.setup)
        cls.payloads = cls.values['SOURCE_PAYLOADS']

    def test_train_recipe_selection_and_resume_isolation(self):
        self.assertEqual((self.values['SEED'], self.values['RESUME_ROOT'], self.values['WALL_HOURS']), (self.seed, '', 11.5))
        self.assertIn("prompt_text_mode='zero'", self.header)
        self.assertIn("prior.get('method')!=cfg['method']", self.header)
        self.assertIn("prior.get('prompt_text_mode')!='zero'", self.header)
        self.assertIn("prior['seed']!=SEED", self.header)
        source = self.payloads['csa_full_runtime.py']
        tree = ast.parse(source)
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_model')
        # Execute only the metadata predicate; omit imports and model construction.
        body = ast.get_source_segment(source, factory)
        self.assertIn(f"cfg.get('seed')!={self.seed}", body)
        predicate = next(n for n in factory.body if isinstance(n, ast.If))
        check = compile(ast.Module(body=[predicate], type_ignores=[]), '<metadata only>', 'exec')
        cfg = dict(csa_per_caption_full=True, semantic=True, prompt_text_mode='zero',
                   method='full_zero_prompt_text_xe10_v1', seed=self.seed)
        exec(check, {'cfg': cfg})
        for other in (42, 43 if self.seed == 44 else 44):
            with self.assertRaises(ValueError): exec(check, {'cfg': dict(cfg, seed=other)})
        for changes in ({'prompt_text_mode': 'correct'}, {'method': 'full_prompt'}, {'semantic': False}):
            with self.assertRaises(ValueError): exec(check, {'cfg': dict(cfg, **changes)})
        original = prior.literals(''.join(json.loads((prior.HERE/'FULL_ZERO_PROMPT_TEXT_SEED42_XE10_2GPU.ipynb').read_text(encoding='utf-8'))['cells'][1]['source']))
        old = original['SOURCE_PAYLOADS']
        changed = [n for n in old if self.payloads[n] != old[n]]
        self.assertEqual(set(changed), {'csa_full_runtime.py', 'score_csa_full.py'})
        self.assertEqual(self.payloads['run_csa_full.py'], old['run_csa_full.py'])
        self.assertEqual(''.join(self.nb['cells'][3]['source']), ''.join(json.loads((prior.HERE/'FULL_ZERO_PROMPT_TEXT_SEED42_XE10_2GPU.ipynb').read_text(encoding='utf-8'))['cells'][3]['source']))
        self.assertEqual(source.replace(f"cfg.get('seed')!={self.seed}", "cfg.get('seed')!=42"), old['csa_full_runtime.py'])
        manifest = json.loads((prior.HERE/f'SEED{self.seed}_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
        target = prior.HERE/f'FULL_ZERO_PROMPT_TEXT_SEED{self.seed}_XE10_2GPU.ipynb'
        self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), manifest['notebook_sha256'])
        self.assertEqual(manifest['status'], 'READY_UNRUN_QUOTA_INSUFFICIENT')
        self.assertEqual(manifest['sha256'], self.values['SOURCE_SHA256'])


class Seed44Checks(Seed43Checks):
    seed = 44


if __name__ == '__main__':
    unittest.main(verbosity=2)
