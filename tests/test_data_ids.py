"""Exercise production split parsing without importing GPU/image dependencies."""
import ast
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from captioning.config import Config
from captioning.visual_cache import coco_image_id


class FakeFrame(list):
    pass


class FakeTokenizer:
    def __init__(self, captions):
        self.word2idx = {'<pad>': 0}


class DataIdsTests(unittest.TestCase):
    def test_load_data_keeps_eval_id_separate_from_coco_id(self):
        source = Path(__file__).resolve().parents[1] / 'captioning' / 'data.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_data')
        namespace = dict(json=json, os=SimpleNamespace(path=SimpleNamespace(isfile=lambda p: True,
                                join=lambda *p: '/'.join(p))),
                         pd=SimpleNamespace(DataFrame=FakeFrame),
                         torch=SimpleNamespace(load=lambda *a, **k: {'data': {}}),
                         ImageFile=SimpleNamespace(LOAD_TRUNCATED_IMAGES=False),
                         CaptionTokenizer=FakeTokenizer, coco_image_id=coco_image_id,
                         SimpleNamespace=SimpleNamespace, build_image_transform=lambda p: None)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        images = []
        for split, coco_id in [('train', 42), ('restval', 7), ('val', 100), ('test', 99)]:
            images.append(dict(split=split, imgid=coco_id + 1000, filepath='val2014',
                               filename=f'COCO_val2014_{coco_id:012d}.jpg',
                               sentences=[{'raw': 'a bicycle'}]))
        with patch('builtins.open', return_value=io.StringIO(json.dumps({'images': images}))):
            data = namespace['load_data'](Config())
        self.assertEqual([r['coco_id'] for r in data.train_df], [42, 7])
        self.assertEqual(data.val_df[0]['coco_id'], 100)
        self.assertEqual(data.test_df[0]['coco_id'], 99)
        self.assertEqual(data.test_df[0]['eval_id'], 3)
        self.assertEqual(data.test_df[0]['karpathy_id'], 1099)


if __name__ == '__main__':
    unittest.main()
