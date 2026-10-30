"""Metadata/NumPy tests only; never imports Torch, model, HDF5 or metric scorers."""
import gzip
import json
import math
import unittest
import numpy as np
from kv_pilot_data import *


class Checks(unittest.TestCase):
    def test_partner_involution_and_no_cross_instance_orbit(self):
        self.assertEqual(len(PARTNER),90)
        for slot,partner in enumerate(PARTNER):
            self.assertEqual(PARTNER[partner],slot)
            i,j = PAIRS[slot//3]
            self.assertEqual(PAIRS[partner//3],(j,i))
            self.assertEqual(PREDICATES[partner%3],INVERSE[PREDICATES[slot%3]])
            self.assertNotEqual(slot,partner)

    def test_exact_schedules_no_dropped_tail_or_group_subsampling(self):
        a,b = schedules(),schedules()
        np.testing.assert_array_equal(a['bce'],b['bce'])
        self.assertEqual(a['bce'].shape,(500,8))
        flat = a['bce'].flatten().tolist()
        self.assertEqual(sorted(flat[:401]),list(range(401)))
        self.assertEqual(sorted(a['mp'][:58]),list(range(58)))
        local0 = a['bce'][:,:4].flatten().tolist()
        local1 = a['bce'][:,4:].flatten().tolist()
        self.assertEqual(Counter(local0+local1),Counter(flat))
        counts = Counter(flat)
        self.assertLessEqual(max(counts.values())-min(counts.values()),1)

    def test_native_word_ids_preserve_unknown_in_field(self):
        vocab = json.loads(gzip.decompress((HERE/'data/gain_metadata.json.gz').read_bytes()))['word2idx']
        audit = json.loads((HERE/'KV_NATIVE_TEXT_AUDIT.json').read_bytes())
        facts = [dict(subject=r['text'],predicate='contains',object='person') for r in audit['fields']]
        report = tokenize_fields(dict(partitions=dict(fit=facts,held=[])),vocab)
        mapped = {r['text']:r for r in report['fields']}
        self.assertEqual(mapped['hair drier']['ids'],[vocab['hair'],3])
        self.assertEqual(mapped['hair drier']['unk_mask'],[False,True])
        self.assertEqual(report['unknown_words'],['drier'])
        self.assertEqual(len(report['fields']),67)
        bad = dict(vocab);bad['contains'],bad['<unk>'] = bad['<unk>'],bad['contains']
        with self.assertRaisesRegex(ValueError,'vocabulary'):tokenize_fields(dict(partitions=dict(fit=facts)),bad)

    def test_mp_known_denominator_and_nonzero_positive_entropy_floor(self):
        self.assertAlmostEqual(log_mp_numpy([0,0,-1000],2),math.log(2))
        self.assertAlmostEqual(log_mp_numpy([2,1,-1],2),log_mp_numpy([12,11,9],2))
        self.assertAlmostEqual(log_mp_numpy([0,0],1),math.log(2))
        for p in (0,2):
            with self.assertRaises(ValueError):log_mp_numpy([1,0],p)

    def test_same_query_concordance_and_sparse_bootstrap(self):
        records = [dict(fact_id='a',family='contact',label=1,image_id=1,z=2),
                   dict(fact_id='b',family='contact',label=0,image_id=2,z=0),
                   dict(fact_id='c',family='containment',label=1,image_id=3,z=1),
                   dict(fact_id='d',family='containment',label=0,image_id=4,z=1)]
        groups = [dict(positive=['a'],negative=['b']),dict(positive=['c'],negative=['d'])]
        score = semantic_metrics(records,groups,'z')
        self.assertEqual(score['macro_family_auc'],.75)
        self.assertEqual(score['query_concordance'],.75)
        sparse = semantic_metrics(records,groups,'z',{1:1,3:1})
        self.assertIsNone(sparse['macro_family_auc'])
        self.assertIsNone(sparse['query_concordance'])
        self.assertEqual(sparse['invalid_query_groups'],2)


if __name__ == '__main__':unittest.main(verbosity=2)
