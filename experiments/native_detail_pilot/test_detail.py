"""Metadata locally; torch/HDF5 interface tests run on Kaggle before expensive staging."""
import ast
import base64
import hashlib
import json
import zlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from detail_data import schedule,verify_cache


class Metadata(unittest.TestCase):
    def test_balanced_caption_independent_schedule(self):
        rows=[dict(coco_id=i) for i in range(10000)];s=schedule(rows)
        self.assertEqual([sum(x) for x in s],[5000]*4)
        self.assertTrue(all(sum(s[e][i] for e in range(4))==2 for i in range(10000)))
        reversed_s=schedule(list(reversed(rows)))
        self.assertEqual(s[0],list(reversed(reversed_s[0])))
        with self.assertRaises(ValueError):schedule(rows[:-1])

    def test_global_ce_normalization_counterexample(self):
        sums=[40.,9.];counts=[20,1];ddp_mean=sum(2*s/sum(counts) for s in sums)/2
        self.assertAlmostEqual(ddp_mean,sum(sums)/sum(counts),places=12);self.assertNotEqual(ddp_mean,sum(s/c for s,c in zip(sums,counts))/2)

    def test_wrong_cache_fails_before_model_load(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root=Path(tmp);(root/'cache_receipt.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):verify_cache(root,dict(files={}))

    def test_runtime_syntax(self):
        root=Path(__file__).resolve().parent
        for p in root.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))

    def test_portable_notebook_hashes_and_exact_input_path(self):
        root=Path(__file__).resolve().parent;nb=root/'NATIVE_DETAIL_SHARED_HL_2GPU.ipynb'
        if not nb.exists():self.skipTest('Build notebook first.')
        self.assertLess(nb.stat().st_size,1000000)
        cells=json.loads(nb.read_text(encoding='utf-8'))['cells'];self.assertIn('native-detail-hl-15k-datase/native_detail_HL_15k_dataset',''.join(cells[1]['source']))
        for c in cells:
            if c['cell_type']!='code':continue
            tree=ast.parse(''.join(c['source']));self.assertEqual(c['outputs'],[]);self.assertIsNone(c['execution_count'])
            assignments={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
            if 'SOURCES' not in assignments:continue
            packed=next(n.value for n in ast.walk(assignments['SOURCES']) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>1000)
            embedded=json.loads(zlib.decompress(base64.b64decode(packed)));hashes=ast.literal_eval(assignments['SOURCE_HASHES'])
            for name,source in embedded.items():
                p=root/name
                if name=='tile_data.py':p=root.parent/'native_detail_benchmark'/name
                elif name=='hidden_metric_smoke.py':p=root.parent/'upt_hidden_pair'/name
                elif name=='hl_upload_inventory.json':p=root.parent/'native_detail_cache/upload_inventory.json'
                elif name.startswith('captioning/'):p=root.parents[1]/name
                self.assertEqual(source,p.read_text(encoding='utf-8'),name)
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),hashes[name])


@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch interface checks execute in Kaggle; no local model execution.')
class Model(unittest.TestCase):
    def setup_model(self):
        import torch
        from torch import nn
        from detail_model import DetailModel
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__();self.visual=nn.Linear(768,512);self.qformer=nn.Identity()
                self.prompt_conditioned_qformer=False;self.object_alignment=self.cascade_alignment=self.relation_alignment=None
        class Decoder(nn.Module):
            def __init__(self):super().__init__();self.out=nn.Linear(512,13)
            def forward(self,tgt,memory,**kwargs):return self.out(memory.mean(1))[:,None].expand(-1,tgt.shape[1],-1)
        class Base(nn.Module):
            def __init__(self):super().__init__();self.encoder=Encoder();self.decoder=Decoder()
            def build_memory(self,visual,legacy,mask):
                q=self.encoder.qformer(self.encoder.visual(visual)[:,:32]);grounded=legacy+q.mean(1,keepdim=True)
                return torch.cat([grounded,q],1),torch.zeros(len(q),legacy.shape[1]+32,dtype=torch.bool)
        torch.manual_seed(42);model=DetailModel(Base());targets=torch.randint(1,13,(2,5,8));targets[:,:,-2:]=0;targets[:,:,0]=1
        b=dict(visual=torch.randn(2,197,768),legacy=torch.randn(2,3,512),legacy_mask=torch.ones(2,3),tiles=torch.randn(2,196,512),xy=torch.rand(2,196,2),captions=targets,caption_mask=targets.eq(0))
        return torch,model,b

    def test_zero_identity_and_recomputed_legacy(self):
        torch,model,b=self.setup_model();a,p=model.memory(b,'A');zero,_=model.memory(b)
        torch.testing.assert_close(a,zero,rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in model.head.parameters()),262656)
        with torch.no_grad():model.head.o.weight.normal_(std=.01)
        changed,_=model.memory(b)
        self.assertFalse(torch.equal(a[:,:3],changed[:,:3]));self.assertFalse(torch.equal(a[:,3:],changed[:,3:]))
        self.assertEqual(len(model.base.encoder.qformer._forward_hooks),0)
        model.train();self.assertFalse(model.base.training)

    def test_gradient_through_frozen_decoder_and_all_head_projections(self):
        torch,model,b=self.setup_model();original={k:v.clone() for k,v in model.base.state_dict().items()};optimizer=torch.optim.AdamW(model.head.parameters(),lr=1e-4)
        for _ in range(2):
            optimizer.zero_grad();loss,stats=model(b,0);loss.backward();optimizer.step()
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum()>0 for p in model.head.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.base.parameters()))
        for k,v in model.base.state_dict().items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('h5py'),'HDF5/Torch loader check executes in Kaggle.')
class Loader(unittest.TestCase):
    def test_same_image_condition_and_truncated_eos(self):
        import h5py
        import numpy as np
        import torch
        from detail_stage import loader
        class Tokenizer:
            eos_idx=2
            def encode(self,text):return torch.tensor([1,4,5,6]),torch.zeros(4,dtype=torch.bool)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root=Path(tmp);stage=root/'stage.h5'
            with h5py.File(stage,'w') as f:
                f.attrs['complete']=True;f.create_dataset('coco_ids',data=np.arange(2))
                f.create_dataset('visual',data=np.zeros((2,197,768),dtype='f2'));f.create_dataset('legacy',data=np.zeros((2,3,512),dtype='f4'));f.create_dataset('legacy_mask',data=np.ones((2,3),dtype='u1'))
            with h5py.File(root/'train_000.h5','w') as f:
                f.create_dataset('coco_ids',data=np.arange(2));f.create_dataset('H',data=np.full((2,196,512),3.,dtype='f2'));f.create_dataset('L',data=np.full((2,196,512),-4.,dtype='f2'));f.create_dataset('xy',data=np.zeros((196,2),dtype='f4'))
            rows=[dict(coco_id=i,image_id=i) for i in range(2)];cfg=dict(groups=dict(train=rows),stage=str(stage),tile_cache=str(root),hl_manifest=[dict(name='train_000',rows=rows)],schedule=[[1,0]]*4,eval_batch_size=16)
            data=dict(images=[dict(sentences=[dict(raw='test')]*5)]*2)
            batches=[]
            for condition in ('H','L'):
                it=loader(cfg,data,Tokenizer(),'train',0,condition=condition);b=next(iter(it));batches.append(b)
                self.assertTrue(torch.all(b['captions'][:,:,-1]==2));self.assertFalse(b['caption_mask'][:,:,-1].any())
                self.assertTrue(torch.all(b['tiles']==(3 if condition=='H' else -4)))
            self.assertTrue(torch.equal(batches[0]['image_id'],batches[1]['image_id']))
            del it


if __name__=='__main__':unittest.main(verbosity=2)
