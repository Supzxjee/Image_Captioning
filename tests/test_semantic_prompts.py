import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from captioning.semantic_prompts import align_prompt_cache, parse_scene, scene_prompt, generate_valid_scene
from build_vlm_prompts import embed, load_scene_rows, records_from_json


class SemanticPromptTests(unittest.TestCase):
    def setUp(self):
        self.scene = {'objects': [{'name': 'Person', 'attributes': ['Red shirt']},
                                  {'name': 'bicycle', 'attributes': []}],
                      'relations': [{'subject': 'person', 'predicate': 'next to', 'object': 'bicycle'}]}

    def test_variants_share_objects_and_spatial_adds_explicit_relation(self):
        scene = parse_scene('```json\n' + json.dumps(self.scene) + '\n```')
        self.assertEqual(scene_prompt(scene, 'objects'), 'red shirt person; bicycle')
        self.assertEqual(scene_prompt(scene, 'spatial'),
                         'person next to bicycle; red shirt person; bicycle')

    def test_rejects_hallucinated_endpoint_or_unsupported_predicate(self):
        for field, value in [('object', 'car'), ('predicate', 'riding')]:
            scene = json.loads(json.dumps(self.scene))
            scene['relations'][0][field] = value
            with self.assertRaises(ValueError):
                parse_scene(json.dumps(scene))

    def test_schema_retry_receives_error_and_preserves_successful_relation(self):
        bad = json.loads(json.dumps(self.scene))
        bad['relations'][0]['object'] = 'car'
        calls, errors = [], []
        def generate_raw(previous_raw, previous_error):
            calls.append((previous_raw, previous_error))
            return json.dumps(bad if len(calls) == 1 else self.scene)
        scene, raw, retries = generate_valid_scene(generate_raw, 2,
            lambda *error: errors.append(error))
        self.assertEqual(retries, 1)
        self.assertEqual(len(scene['relations']), 1)
        self.assertIn('endpoints', calls[1][1])
        self.assertEqual(len(errors), 1)

    def test_retry_exhaustion_does_not_return_empty_fallback(self):
        calls = []
        def generate_raw(*feedback):
            calls.append(feedback)
            return 'not JSON'
        with self.assertRaisesRegex(ValueError, 'after 3 attempts'):
            generate_valid_scene(generate_raw, retries=2)
        self.assertEqual(len(calls), 3)

    def test_filename_alignment_handles_new_kaggle_mount_and_rejects_collisions(self):
        value = {'tokens': 'sentinel'}
        path = '/kaggle/input/new/images/train2014/COCO_42.jpg'
        for oldkey in ('COCO_42.jpg', '/kaggle/input/old/images/train2014/COCO_42.jpg'):
            self.assertEqual(align_prompt_cache({oldkey: value}, [path]), {path: value})
        with self.assertRaises(ValueError):
            align_prompt_cache({'/a/x.jpg': value, '/b/x.jpg': value}, [])

    def test_no_caption_fields_in_vlm_records_and_duplicate_scene_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp)/'dataset.json'
            path.write_text(json.dumps({'images': [dict(filename='x.jpg', filepath='train2014',
                split='train', sentences=[{'raw': 'NEVER PASS THIS TO THE VLM'}])]}))
            self.assertEqual(records_from_json(path),
                             [dict(filename='x.jpg', filepath='train2014', split='train')])
            rows = Path(tmp)/'scenes.jsonl'
            row = json.dumps(dict(filename='x.jpg', scene=self.scene))
            rows.write_text(row+'\n'+row+'\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate scene'):
                load_scene_rows([rows])

    def test_embedding_writes_two_compatible_caches_and_rejects_missing_images(self):
        import hashlib
        import torch

        class Inputs(dict):
            def to(self, device):
                return self

        class Tokenizer:
            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

            def __call__(self, texts, **kw):
                if kw.get('return_tensors'):
                    count = len(texts)
                    return Inputs(input_ids=torch.ones(count, 20, dtype=torch.long),
                                  attention_mask=torch.ones(count, 20, dtype=torch.long))
                if isinstance(texts, str):
                    return {'input_ids': [1, 2, 3]}
                return {'input_ids': [[1, 2, 3] for _ in texts]}

        class Model(Tokenizer):
            def float(self): return self
            def to(self, device): return self
            def eval(self): return self
            def __call__(self, **inputs):
                return SimpleNamespace(last_hidden_state=torch.ones(len(inputs['input_ids']), 20, 512))

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            dataset = root/'dataset.json'
            record = dict(filename='x.jpg', filepath='train2014', split='train')
            dataset.write_text(json.dumps({'images': [record]}))
            scenes = root/'scenes.jsonl'
            scenes.write_text(json.dumps(dict(record, scene=self.scene))+'\n')
            Path(str(scenes)+'.meta.json').write_text(json.dumps({
                'complete': True, 'source': 'images_only', 'limit': 0,
                'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest()}))
            args = SimpleNamespace(dataset_json_path=dataset, scenes=[scenes],
                                   output_dir=root/'cache', batch_size=2, allow_partial=False)
            with patch.dict('sys.modules', {'transformers': SimpleNamespace(
                    CLIPTextModel=Model, CLIPTokenizer=Tokenizer)}), patch.object(torch.cuda, 'is_available', return_value=False):
                embed(args)
                for variant in ('objects', 'spatial'):
                    bundle = torch.load(root/'cache'/f'prompt_vlm_{variant}.pt', weights_only=False)
                    self.assertTrue(bundle['metadata']['complete'])
                    self.assertEqual(bundle['data']['x.jpg']['tokens'].shape, (20, 512))
                    self.assertEqual(bundle['data']['x.jpg']['tokens'].dtype, torch.float16)
                    self.assertEqual(bundle['data']['x.jpg']['mask'].sum(), 20)
                dataset.write_text(json.dumps({'images': [record, dict(record, filename='y.jpg')]}))
                # Keep provenance current so this specifically exercises coverage rejection.
                Path(str(scenes)+'.meta.json').write_text(json.dumps({
                    'complete': True, 'source': 'images_only', 'limit': 0,
                    'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest()}))
                with self.assertRaisesRegex(ValueError, 'Missing 1 scenes'):
                    embed(args)


if __name__ == '__main__':
    unittest.main()
