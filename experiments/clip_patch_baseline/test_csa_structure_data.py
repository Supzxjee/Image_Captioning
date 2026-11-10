"""Author-safe NumPy/mock/AST checks; never import a neural module."""
import ast
from pathlib import Path
import tempfile
from unittest import TestCase, main
from unittest.mock import patch
import numpy as np
import csa_structure_data as d
from test_semantic_data import record

HERE = Path(__file__).resolve().parent


class Checks(TestCase):
    def test_probe_uses_distinct_train_images_and_includes_relation(self):
        captions=np.repeat(np.arange(256),5); valid=np.zeros((256,3),bool);valid[200,0]=True
        events=d.probe_events(captions,valid)
        self.assertEqual(len(events),32);self.assertEqual(len(set(captions[events])),32)
        self.assertEqual(captions[events[0]],200)
        with self.assertRaises(ValueError):d.probe_events(captions,np.zeros_like(valid))

    def test_prompt_order_train_scope_and_instance_mapping(self):
        texts = ['a person', 'a bicycle', 'person riding bicycle']
        result = d.selected_prompts([record(99, 'val'), record(1), record(2)], [2, 1], texts)
        self.assertEqual(result['object_weights'].shape, (2, 8, 196))
        np.testing.assert_array_equal(result['subject_weights'][:, 0], result['object_weights'][:, 0])
        np.testing.assert_array_equal(result['target_weights'][:, 0], result['object_weights'][:, 1])
        for rows, ids in [([record(1)], [1, 2]), ([record(1), record(1)], [1]),
                          ([record(1, 'val')], [1]), ([record(1)], [1, 1])]:
            with self.assertRaises(ValueError): d.selected_prompts(rows, ids, texts)

    def test_empty_and_broken_prompts(self):
        row = record(); row.update(objects=[], relations=[])
        result = d.selected_prompts([row], [1], [])
        self.assertTrue(all(not v.any() for v in result.values()))
        row = record(); row['relations'][0]['subject_instance'] = 'wrong'
        with self.assertRaises(ValueError): d.selected_prompts([row], [1], ['a person', 'a bicycle', 'person riding bicycle'])

    def test_small_stage_seals_and_reject_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); stage = root/'stage'; stage.mkdir(); out = root/'out'; out.mkdir(); cache = root/'cache'; cache.mkdir()
            ids = np.arange(1, 257, dtype=np.int64); np.save(stage/'coco_ids.npy', ids)
            rows = [record(int(i)) for i in ids]
            (cache/'prompt_links.jsonl').write_text(''.join(d.json.dumps(r)+'\n' for r in rows), encoding='utf-8')
            texts = ['a person', 'a bicycle', 'person riding bicycle']; vectors = np.eye(3, 512, dtype=np.float32)
            fixture = ({'cache_files': {}}, cache, {'train': ids.tolist()}, texts, vectors, {}, {'noun': vectors, 'verb': vectors}, {'files': {}})
            with patch.object(d, 'checked_inputs', return_value=fixture):
                d.stage_prompts(dict(stage=str(stage), output=str(out)))
            receipt = d.checked_semantics(stage)
            self.assertEqual((receipt['objects'], receipt['relations']), (512, 256))
            self.assertFalse(receipt['test_used']); self.assertFalse(receipt['validation_used'])
            with (stage/'sem_object_weights.npy').open('ab') as f: f.write(b'changed')
            with self.assertRaises(ValueError): d.checked_semantics(stage)

    def test_model_packages_then_splits_without_early_mixing(self):
        tree = ast.parse((HERE/'csa_patch_model.py').read_text(encoding='utf-8'))
        methods = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        decode = ast.unparse(methods['decode'])
        self.assertIn('memory.split(196, dim=1)', decode)
        self.assertIn('self.core.decode(prefix, vo, vr, valid)', decode)
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertNotIn('strength_gate', names); self.assertNotIn('base', names)
        source = HERE.parent/'csa_paper/model.py'
        if not source.is_file(): source = HERE/'bank_layers.py'
        core = ast.parse(source.read_text(encoding='utf-8'))
        encode = next(n for c in core.body if isinstance(c, ast.ClassDef) and c.name == 'CSACaptioner' for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'encode')
        self.assertEqual([n.func.attr for n in ast.walk(encode) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {'object1', 'object2', 'relation'}], ['object1', 'object2', 'relation'])
        decoder = next(c for c in core.body if isinstance(c, ast.ClassDef) and c.name == 'DecoderLayer')
        source = ast.unparse(decoder)
        for name in ('self.object_attn', 'self.relation_attn', 'self.gate_o', 'self.gate_r', 'previous_word'):
            self.assertIn(name, source)

    def test_runner_sources_parse_without_importing_torch(self):
        for name in ('csa_patch_model.py', 'csa_structure_runtime.py', 'csa_structure_data.py', 'run_csa_structure_smoke.py'):
            ast.parse((HERE/name).read_text(encoding='utf-8'))
        runner = ast.parse((HERE/'run_csa_structure_smoke.py').read_text(encoding='utf-8'))
        imports = [n for n in runner.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any('torch' in ast.unparse(n) for n in imports))


if __name__ == '__main__': main(verbosity=2)
