import unittest
import torch
from captioning.hoi_acceptance import average_precision, acceptance_metrics, score_inventory
from captioning.evidence_routing import fit_selective_policy, selective_decisions


class HOIAcceptanceTests(unittest.TestCase):
    def test_unavailable_positive_stays_in_recall_denominator_unknown_not_false(self):
        p = torch.tensor([[.9]*6, [.8]*6, [.7]*6, [.6]*6])
        y = torch.tensor([[1]*6, [1]*6, [0]*6, [-1]*6])
        valid = torch.ones_like(y, dtype=torch.bool); valid[1] = False
        decisions = torch.ones_like(y); decisions[1] = -1
        metrics = acceptance_metrics(p, y, valid, decisions)
        self.assertEqual(metrics['accepted_true'], 6)
        self.assertEqual(metrics['accepted_false'], 6)
        self.assertEqual(metrics['accepted_unknown'], 6)
        self.assertEqual(metrics['full_pool_positive_recall'], .5)
        self.assertEqual(metrics['known_precision'], .5)
        self.assertEqual(metrics['unavailable_known'], 6)

    def test_tied_ap_is_grouped_and_order_invariant(self):
        self.assertAlmostEqual(average_precision(torch.tensor([.5, .5]), torch.tensor([1, 0])), .5)
        self.assertAlmostEqual(average_precision(torch.tensor([.5, .5]), torch.tensor([0, 1])), .5)
        self.assertIsNone(average_precision(torch.tensor([.9]), torch.tensor([0])))

    def test_missing_ontology_and_pair_do_not_become_predictions(self):
        row = {'subject':'person', 'subject_box':[0.,0.,.2,.9],
               'object':'bicycle', 'object_box':[.2,.2,.9,.9]}
        pred = {'subject_indices':torch.tensor([0]), 'object_indices':torch.tensor([1]),
                'labels':torch.tensor([0,1]), 'logits':torch.ones((1,6)),
                'official_output':{'pairing':torch.tensor([[0],[1]]), 'labels':torch.tensor([1]), 'scores':torch.tensor([.7])}}
        from captioning.hoi_transfer_eval import node_key, VERB_NAMES
        mapping = {node_key(row['subject'],row['subject_box']):{'index':0},
                   node_key(row['object'],row['object_box']):{'index':1}}
        raw, native, valid, reasons = score_inventory([row],pred,mapping,VERB_NAMES,[[],[1]])
        self.assertEqual(valid.tolist(),[[False,True,False,False,False,False]])
        self.assertAlmostEqual(float(native[0,1]),.7,places=6)
        self.assertEqual(reasons[0][0],'unsupported_object_verb')
        _, _, missing, _ = score_inventory([row],pred,{},VERB_NAMES,[[],[1]])
        self.assertFalse(missing.any())

    def test_unavailable_calibration_cannot_create_supported_tail(self):
        p = torch.tensor([[.99],[.98],[.97],[.96],[.95],[.1]])
        y = torch.tensor([[-1],[-1],[-1],[-1],[1],[0]])
        policy = fit_selective_policy(p,y,precision=.9,min_known=5)
        self.assertGreater(policy['positive'][0],1)
        decisions = selective_decisions(p,torch.ones_like(y,dtype=torch.bool),policy)
        self.assertFalse((decisions==1).any())


if __name__ == '__main__': unittest.main()
