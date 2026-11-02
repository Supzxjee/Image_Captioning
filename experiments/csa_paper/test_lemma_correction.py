"""Pure metadata rule/count tests; no POS, model or pip execution."""
import unittest

from lemma_correction import corrected_lemma, corrected_bank


class Checks(unittest.TestCase):
    def test_narrow_rule_preserves_real_rid_and_nonverbs(self):
        self.assertEqual(corrected_lemma('riding', 'VERB', 'rid'), 'ride')
        self.assertEqual(corrected_lemma('Rides', 'VERB', 'rid'), 'ride')
        for surface, pos, lemma in [('rid', 'VERB', 'rid'), ('ridding', 'VERB', 'rid'),
                ('riding', 'NOUN', 'rid'), ('riding', 'VERB', 'ride'), ('rode', 'NOUN', 'rode')]:
            self.assertEqual(corrected_lemma(surface, pos, lemma), lemma)

    def test_count_conservation_and_original_preservation(self):
        original = dict(minimum_count=5, noun=[dict(word='horse', count=9)],
            verb=[dict(word='rid', count=30), dict(word='ride', count=8), dict(word='hold', count=6)])
        result, delta = corrected_bank(original, 20)
        self.assertEqual(result['noun'], original['noun'])
        self.assertEqual(result['verb'], [dict(word='ride', count=28), dict(word='rid', count=10), dict(word='hold', count=6)])
        self.assertEqual(original['verb'][0]['count'], 30)
        self.assertEqual(delta['after_rid']+delta['after_ride'], 38)
        with self.assertRaises(ValueError):
            corrected_bank(original, 31)

    def test_existing_minimum_rule_applies_only_after_count_delta(self):
        original = dict(minimum_count=5, noun=[], verb=[dict(word='rid', count=10), dict(word='ride', count=5)])
        result, delta = corrected_bank(original, 8)
        self.assertEqual(result['verb'], [dict(word='ride', count=13)])
        self.assertEqual(delta['below_minimum_omitted_mass'], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
