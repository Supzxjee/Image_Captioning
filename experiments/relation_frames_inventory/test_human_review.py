"""Test provenance/confirmation guards using copies, not real review edits."""
import copy
import json
from pathlib import Path
import unittest
from import_human_review import validate

HERE=Path(__file__).resolve().parent


class Checks(unittest.TestCase):
    def setUp(self):
        read=lambda p:json.loads(p.read_text(encoding='utf-8'))
        self.selection=read(HERE/'AUDIT32_SELECTION.json');self.draft=read(HERE/'ASSISTANT_AUDIT32.json')
        self.report=read(HERE/'HUMAN_AUDIT32_EFFECTIVE.json');self.supplement=read(HERE/'HUMAN_AUDIT32_CHAT_SUPPLEMENT.json')
        self.report.update(schema_version=1,reviewer_type='assistant_proposed_pending_human_verification')
        # Reconstruct the partial-export fixture without depending on a private ignored file.
        self.report['annotations'][self.supplement['confirmations'][0]['review_id']]['confirmed_by_human']=False
        for item in self.supplement['confirmations'][1:]:
            self.report['annotations'][item['review_id']]['assertion']='uncertain'

    def test_whole_pending_export_still_has31_valid_confirmations(self):
        result,_=validate(self.selection,self.draft,self.report,dict(confirmations=[]))
        self.assertEqual(result['effective_confirmations'],31)
        self.assertEqual([r['index'] for r in result['pending']],[12])

    def test_explicit_revisions_and_override_of_ai32(self):
        result,_=validate(self.selection,self.draft,self.report,self.supplement)
        self.assertEqual(result['effective_confirmations'],32)
        self.assertEqual(result['uncertain'],[])
        self.assertEqual(result['definite_source_disagreements'],[])
        self.assertEqual([r['index'] for r in result['changes_from_assistant']],[16,22,32])
        self.assertEqual([r['index'] for r in result['explicit_assertion_revisions']],[16,22])
        self.assertFalse(result['training_approved'])

    def test_wrong_audit_or_unknown_record_fails(self):
        bad=copy.deepcopy(self.report);bad['audit_sha256']='wrong'
        with self.assertRaises(ValueError):validate(self.selection,self.draft,bad,self.supplement)
        bad=copy.deepcopy(self.report);bad['annotations']['unknown']={}
        with self.assertRaises(ValueError):validate(self.selection,self.draft,bad,self.supplement)

    def test_chat_cannot_silently_change_an_answer(self):
        bad=copy.deepcopy(self.supplement);bad['confirmations'][0]['assertion']='supported'
        with self.assertRaises(ValueError):validate(self.selection,self.draft,self.report,bad)

    def test_explicit_revision_requires_correct_old_value_and_origin(self):
        bad=copy.deepcopy(self.supplement);bad['confirmations'][1]['origin']='untrusted_document'
        with self.assertRaises(ValueError):validate(self.selection,self.draft,self.report,bad)
        bad=copy.deepcopy(self.supplement);bad['confirmations'][1]['previous_assertion']='supported'
        with self.assertRaises(ValueError):validate(self.selection,self.draft,self.report,bad)


if __name__=='__main__':unittest.main(verbosity=2)
