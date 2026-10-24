"""One shared262656-parameter adapter; hook Q before existing legacy grounding."""
import torch
from torch import nn
from torch.nn import functional as F


class DetailAdapter(nn.Module):
    def __init__(self):
        super().__init__();self.q=nn.Linear(512,128,bias=False);self.k=nn.Linear(514,128,bias=False);self.v=nn.Linear(514,128,bias=False);self.o=nn.Linear(128,512,bias=False)
        nn.init.zeros_(self.o.weight)
        assert sum(p.numel() for p in self.parameters())==262656

    def forward(self,queries,tiles,xy):
        b,n,_=tiles.shape;coords=xy.expand(b,-1,-1)
        semantic=torch.cat([F.layer_norm(tiles,(512,)),coords],-1)
        def heads(t):return t.reshape(b,-1,4,32).transpose(1,2)
        q=heads(self.q(F.layer_norm(queries,(512,))));k=heads(self.k(semantic));v=heads(self.v(semantic))
        attended=(torch.softmax(q@k.transpose(-1,-2)/(32**.5),-1)@v).transpose(1,2).reshape(b,queries.shape[1],128)
        return self.o(attended)


class DetailModel(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.head=DetailAdapter();self.base.requires_grad_(False);self.base.eval()
        e=base.encoder
        if e.qformer is None or e.prompt_conditioned_qformer or e.object_alignment is not None or e.cascade_alignment is not None or e.relation_alignment is not None:raise ValueError('Expected unchanged A QFormer encoder.')

    def train(self,mode=True):
        super().train(mode);self.base.eval();return self

    @property
    def decoder(self):return self.base.decoder

    def memory(self,batch,condition='on'):
        if condition=='A':return self.base.build_memory(batch['visual'],batch['legacy'],batch['legacy_mask'])
        if condition!='on':raise ValueError(condition)
        handle=self.base.encoder.qformer.register_forward_hook(lambda module,args,q:q+self.head(q,batch['tiles'],batch['xy']))
        try:
            # No no_grad here: Q' must retain its gradient through frozen legacy attention and decoder.
            return self.base.build_memory(batch['visual'],batch['legacy'],batch['legacy_mask'])
        finally:handle.remove()

    def teacher(self,memory,padding,targets,masks):
        b,n,d=memory.shape
        if len(targets)!=b*5:raise ValueError('Expected image-major five-reference expansion.')
        memory=memory[:,None].expand(b,5,n,d).reshape(b*5,n,d);padding=padding[:,None].expand(b,5,n).reshape(b*5,n)
        return self.decoder(targets,memory,tgt_key_padding_mask=masks,memory_key_padding_mask=padding)

    def forward(self,batch,pad_idx):
        import torch.distributed as dist
        memory,padding=self.memory(batch);targets=batch['captions'].flatten(0,1);masks=batch['caption_mask'].flatten(0,1)
        logits=self.teacher(memory,padding,targets[:,:-1],masks[:,:-1])
        total=F.cross_entropy(logits.flatten(0,1),targets[:,1:].reshape(-1),ignore_index=pad_idx,reduction='sum')
        count=targets[:,1:].ne(pad_idx).sum().to(total.dtype);stats=torch.stack([total.detach(),count.detach()])
        world=dist.get_world_size() if dist.is_initialized() else 1
        if world>1:dist.all_reduce(stats)
        if stats[1]<=0:raise ValueError('No valid target tokens.')
        return total*world/stats[1],stats
