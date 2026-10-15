import json
import tempfile
import unittest
from pathlib import Path
from captioning.hoi_caption_reranking import extract_claims,candidate_evidence,rerank_with_hoi,METHOD
from captioning.multiscore_reranking import rerank_multiscore
from captioning.inference_journal import InferenceJournal
from run_hoi_caption_validation import merge_records


class CaptionHOITests(unittest.TestCase):
    def test_conservative_subject_verb_object_and_negation(self):
        self.assertEqual(extract_claims('A man is riding a bicycle next to a bus.'),
            [{'subject':'person','verb':'ride','object':'bicycle'}])
        self.assertEqual(extract_claims('A woman holding an umbrella.'),
            [{'subject':'person','verb':'hold','object':'umbrella'}])
        for caption in ['A dog holding a ball.','A man is not riding a horse.',
                        'No man holding an umbrella.','A horse next to a man.',
                        'A man holding something and a woman with an umbrella.']:
            self.assertEqual(extract_claims(caption),[])
        self.assertEqual(extract_claims('A person speaking on a phone.')[0]['object'],'cell phone')

    def test_unknown_neutral_and_exact_zero_weight_baseline(self):
        empty=candidate_evidence('A man riding a bike.',{})
        self.assertEqual(empty['feature'],0.)
        self.assertEqual(empty['claims'][0]['status'],'unknown')
        record={'image_id':1,'candidates':[
            {'caption':'A man riding a bike.','logprob':-2.,'clipscore':.7,'hoi_evidence':empty},
            {'caption':'A man near a bicycle.','logprob':-3.,'clipscore':.8,
             'hoi_evidence':candidate_evidence('A man near a bicycle.',{})}]}
        detected={'person':.9,'bicycle':.9}
        old,_=rerank_multiscore(record,detected,.4,.3)
        new,_=rerank_with_hoi(record,detected,0.)
        self.assertEqual(old['original_rank'],new['original_rank'])
        self.assertEqual(old['combined_score'],new['combined_score'])
        new,_=rerank_with_hoi(record,detected,.2)
        self.assertEqual(old['original_rank'],new['original_rank'])

    def test_image_predicate_contrast_does_not_use_half_probability_cutoff(self):
        import torch
        from captioning.hoi_caption_reranking import evidence_inventory,VERBS
        pred={'logits':torch.tensor([[-8.,-8.,-4.,-8.,-8.,-8.],[-8.,-8.,-8.,-8.,-8.,-8.]]),
            'subject_indices':torch.tensor([0,0]),'object_indices':torch.tensor([1,2]),
            'labels':torch.tensor([0,1,2]),'size':torch.tensor([100,100]),
            'boxes':torch.tensor([[0.,0.,20.,90.],[10.,10.,50.,90.],[50.,10.,90.,90.]])}
        inv=evidence_inventory(pred,['person','chair','bench'],list(VERBS),[[],[2],[2]])
        good=candidate_evidence('A man sitting on a chair.',inv)
        bad=candidate_evidence('A man sitting on a bench.',inv)
        self.assertLess(inv['sit_on','chair']['probability'],.5)
        self.assertGreater(good['feature'],0.)
        self.assertLess(bad['feature'],0.)
        inv=evidence_inventory(pred,['person','chair','bench'],list(VERBS),[[],[2],[]])
        self.assertEqual(candidate_evidence('A man sitting on a chair.',inv)['feature'],0.)

    def test_merge_never_silently_loses_or_duplicates_images(self):
        rows=[{'image_id':i,'coco_id':i+100,'filename':str(i)} for i in range(4)]
        self.assertEqual(merge_records(rows[::2]+rows[1::2],rows),rows)
        for broken in [rows[:-1],rows+rows[:1],[{**rows[0],'coco_id':999}]+rows[1:]]:
            with self.assertRaises(ValueError):merge_records(broken,rows)

    def test_resume_input_identity_and_interrupted_final_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'part.jsonl'
            log=InferenceJournal(path,{'checkpoint':'abc'},[1,2])
            log.append({'image_id':1,'caption':'safe'})
            with path.open('ab') as stream:stream.write(b'{"image_id":')
            restored=InferenceJournal(path,{'checkpoint':'abc'},[1,2])
            self.assertEqual(len(restored.records),1)
            restored.append({'image_id':2,'caption':'safe'})
            self.assertEqual(len(InferenceJournal(path,{'checkpoint':'abc'},[1,2]).records),2)
            with self.assertRaises(ValueError):InferenceJournal(path,{'checkpoint':'changed'},[1,2])
            with self.assertRaises(ValueError):restored.append({'image_id':2})


if __name__=='__main__':unittest.main()
