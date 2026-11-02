"""Mock token objects only; never load spaCy or rerun actual POS."""
from types import SimpleNamespace
import unittest

from lemma_trace import inspect_doc, summarize


def token(surface, lemma, pos='VERB'):
    return SimpleNamespace(text=surface, lemma_=lemma, pos_=pos, tag_='VBG', morph='VerbForm=Part')


class Checks(unittest.TestCase):
    def test_wrong_ride_lemma_is_flagged_not_corrected(self):
        original = token('riding', 'rid')
        result = inspect_doc([original])
        self.assertEqual(result['focus'][0]['status'], 'MISMATCH')
        self.assertEqual(result['focus'][0]['lemma'], 'rid')
        self.assertEqual(original.lemma_, 'rid')

    def test_legitimate_rid_and_ride_and_nonverb(self):
        result = inspect_doc([token('rid', 'rid'), token('ridding', 'rid'), token('Riding', 'ride'), token('rides', 'ride', 'NOUN')])
        self.assertEqual([x['status'] for x in result['focus']], ['MATCH', 'MATCH', 'MATCH', 'NOT_VERB_REVIEW'])

    def test_summary_separates_controls_from_actual_train(self):
        rows = [dict(kind='train_caption', **inspect_doc([token('riding', 'rid')])),
                dict(kind='synthetic_control', **inspect_doc([token('rid', 'rid')]))]
        result = summarize(rows)
        self.assertEqual(result['case_count'], 2)
        self.assertFalse(result['bank_modified'])
        self.assertFalse(result['dataset_error_rate_estimated'])
        self.assertEqual(len(result['transitions']), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
