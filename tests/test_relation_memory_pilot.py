import importlib
import types
import unittest
from unittest.mock import patch
import torch
from captioning.relation_memory_pilot import action_entry, choose_training, adapt_parameters


class RelationMemoryPilotTests(unittest.TestCase):
    def test_disjoint_deterministic_train_and_mask_source_unchanged(self):
        source = dict(mask=torch.tensor([1,1]), relations=[dict(relation='riding'),dict(relation='above')])
        torch.testing.assert_close(action_entry(source)['mask'],torch.tensor([1,0]))
        torch.testing.assert_close(source['mask'],torch.tensor([1,1]))
        records=[dict(filename=f'{i}.jpg',split='train',sentences=[dict(raw='a person')]) for i in range(6)]
        records.append(dict(filename='val.jpg',split='val',sentences=[dict(raw='a person')]))
        prompts={r['filename']:{} for r in records}; relations={r['filename']:source for r in records}
        a=choose_training(records,prompts,relations,{'0.jpg'},count=4)
        b=choose_training(list(reversed(records)),prompts,relations,{'0.jpg'},count=4)
        self.assertEqual(a,b)
        self.assertFalse({r['filename'] for r in a}&{'0.jpg','val.jpg'})
        with self.assertRaises(ValueError): choose_training(records,prompts,relations,{'0.jpg'},count=7)

    def test_only_attention_and_norm_adapt_with_finite_optimizer_step(self):
        with patch.dict('sys.modules',{'transformers':types.SimpleNamespace(CLIPModel=object)}):
            models=importlib.import_module('captioning.models')
        encoder=torch.nn.Linear(3,3)
        decoder=models.CaptionDecoder(10,d_model=16,nhead=4,num_layers=1)
        model=models.ImageCaptioningModel(encoder,decoder)
        parameters,names=adapt_parameters(model)
        self.assertTrue(all('.multihead_attn.' in name or '.norm2.' in name for name in names))
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))
        self.assertFalse(decoder.embedding.weight.requires_grad)
        self.assertFalse(decoder.fc_out.weight.requires_grad)
        before=parameters[0].detach().clone()
        optimizer=torch.optim.AdamW(parameters,lr=1e-5)
        logits=decoder(torch.tensor([[1,2,3]]),torch.randn(1,8,16),memory_key_padding_mask=torch.zeros(1,8).bool())
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,10),torch.tensor([2,3,4]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward();optimizer.step()
        self.assertFalse(torch.equal(before,parameters[0]))


if __name__=='__main__':unittest.main()
