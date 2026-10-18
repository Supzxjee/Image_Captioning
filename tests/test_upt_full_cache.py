import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from captioning.upt_cache import image_plan, pack_prediction, finish_shard, open_partial, read_shard
from build_upt_full_cache import restore, scan, save


def fixture():
    dataset = {'images': [{'cocoid': i, 'filename': f'COCO_train2014_{i:012d}.jpg',
                          'filepath': 'train2014', 'split': split, 'sentences': ['must never enter cache']}
                         for i, split in enumerate(('train', 'restval', 'train', 'val', 'test'), 1)]}
    vocabulary = {'objects': ['person', 'horse'], 'verbs': ['hold', 'ride', 'other'],
                  'verbs_by_object': [[0], [0, 1]]}
    p = {'boxes': [[0., 0., 20., 30.], [10., 20., 50., 60.]], 'labels': [0, 1], 'scores': [.9, .8],
         'size': [100, 100], 'subject_indices': [0], 'object_indices': [1], 'logits': [[-1., 2., 3.]],
         'official_output': {'pairing': [[0, 0], [1, 1]], 'labels': [0, 1], 'scores': [.1, .7]}}
    return dataset, vocabulary, p


class FullCacheTests(unittest.TestCase):
    def test_plan_all_splits_and_no_caption_content(self):
        tasks = image_plan(fixture()[0], 2)
        self.assertEqual([(t['split'], len(t['images'])) for t in tasks], [('train', 2), ('train', 1), ('val', 1), ('test', 1)])
        rows = [r for t in tasks for r in t['images']]
        self.assertEqual(len({r['image_id'] for r in rows}), 5)
        self.assertNotIn('must never enter cache', json.dumps(rows))
        self.assertEqual(rows[1]['source_split'], 'restval')

    def test_duplicate_identity_rejected(self):
        data, _, _ = fixture()
        data['images'].append(copy.deepcopy(data['images'][0]))
        with self.assertRaises(ValueError):
            image_plan(data)

    def test_sparse_all_supported_logits_and_native_scores(self):
        dataset, vocabulary, prediction = fixture()
        row = image_plan(dataset)[0]['images'][0]
        packed = pack_prediction(row, prediction, vocabulary, 'hash', [640, 480])
        self.assertEqual(packed['pair_logits'], [[-1., 2.]])
        self.assertEqual(packed['native_scores'], [[.1, .7]])
        self.assertEqual(packed['pairs'], [[0, 1]])
        self.assertNotIn('claims', packed)
        self.assertEqual(packed['original_size_wh'], [640, 480])

    def test_no_pair_preserves_objects_and_empty_detection_supported(self):
        dataset, vocabulary, p = fixture()
        p.update(subject_indices=[], object_indices=[], logits=[], official_output={'pairing': [[], []], 'labels': [], 'scores': []})
        packed = pack_prediction(image_plan(dataset)[0]['images'][0], p, vocabulary, 'hash', [10, 10])
        self.assertEqual(len(packed['labels']), 2)
        self.assertEqual(packed['status'], 'no_human_object_pairs')
        p.update(boxes=[], labels=[], scores=[])
        self.assertEqual(pack_prediction({}, p, vocabulary, 'hash', [10, 10])['boxes_resized_xyxy'], [])

    def test_invalid_pair_nonfinite_logits_and_wrong_dimension_fail(self):
        dataset, v, p = fixture(); row = image_plan(dataset)[0]['images'][0]
        for edit in ('self', 'nan', 'dimension'):
            value = copy.deepcopy(p)
            if edit == 'self': value['object_indices'] = [0]
            elif edit == 'nan': value['logits'][0][0] = float('nan')
            else: value['logits'] = [[1.]]
            with self.assertRaises(ValueError):
                pack_prediction(row, value, v, 'hash', [10, 10])

    def test_journal_partial_recovery_finish_and_signature(self):
        task = image_plan(fixture()[0], 2)[0]
        with tempfile.TemporaryDirectory() as tmp:
            journal = open_partial(tmp, task, 'same')
            journal.append({**task['images'][0], 'data': 1.234567890123})
            with journal.path.open('ab') as stream: stream.write(b'{"image_id":')
            resumed = open_partial(tmp, task, 'same')
            self.assertEqual(len(resumed.records), 1)
            resumed.append({**task['images'][1], 'data': 2.})
            path = finish_shard(tmp, task, 'same', resumed.records)
            self.assertEqual(read_shard(path, task, 'same'), resumed.records)
            self.assertFalse(journal.path.exists())
            with self.assertRaises(ValueError): read_shard(path, task, 'different')
            with self.assertRaises(FileExistsError): finish_shard(tmp, task, 'same', resumed.records)

    def test_incomplete_shard_and_gzip_corruption_rejected(self):
        task = image_plan(fixture()[0], 2)[0]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): finish_shard(tmp, task, 's', task['images'][:1])
            path = finish_shard(tmp, task, 's', task['images'])
            path.write_bytes(path.read_bytes()[:-5])
            with self.assertRaises((EOFError, OSError)): read_shard(path, task, 's')

    def test_restore_completed_and_partial_after_kill_without_manifest(self):
        tasks = image_plan(fixture()[0], 2)
        with tempfile.TemporaryDirectory() as tmp:
            previous, output = Path(tmp) / 'previous', Path(tmp) / 'output'
            previous.mkdir(); output.mkdir()
            save(previous / 'cache_identity.json', {'signature': 's'})
            finish_shard(previous, tasks[0], 's', tasks[0]['images'])
            journal = open_partial(previous, tasks[1], 's')
            journal.append(tasks[1]['images'][0])
            restore(previous, output, tasks, 's')
            result = scan(output, tasks, 's')
            self.assertEqual(result['complete_shard_images']['train'], 2)
            self.assertEqual(result['partial_images']['train'], 1)
            self.assertFalse(result['complete'])
            self.assertTrue(journal.path.is_file())  # Input unchanged.
            with self.assertRaises(ValueError): restore(previous, output, tasks, 'different')


if __name__ == '__main__':
    unittest.main()
