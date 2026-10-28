"""Real A caption decoder with a shared I/J reader; user-run Torch only."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


class PairReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.entity=nn.Sequential(nn.Linear(517,512),nn.ReLU())
        self.pair=nn.Sequential(nn.Linear(1544,512),nn.ReLU())
        self.qs=nn.Linear(512,128,bias=False);self.qo=nn.Linear(512,128,bias=False);self.qr=nn.Linear(512,128,bias=False)
        self.ks=nn.Linear(512,128,bias=False);self.ko=nn.Linear(512,128,bias=False);self.kr=nn.Linear(512,128,bias=False)
        self.null_keys=nn.Parameter(torch.zeros(2,128));self.neutral_pair=nn.Parameter(torch.zeros(128))
        self.output=nn.Linear(1536,512,bias=False);nn.init.zeros_(self.output.weight)

    def bank(self,x):
        roi=x['roi'];union=x['union'];v=x['patch']
        obj=torch.bmm(roi/roi.sum(-1,keepdim=True).clamp_min(1e-12),v)
        u=torch.bmm(union/union.sum(-1,keepdim=True).clamp_min(1e-12),v)
        e=self.entity(torch.cat([obj,x['boxes'],x['confidence']],-1))
        e=e.masked_fill(~x['valid'][...,None],0)
        e=e*torch.tensor([1.]*12+[0.],device=e.device)[None,:,None]
        left=e[:,:,None,:].expand(-1,13,13,-1).reshape(-1,169,512)
        right=e[:,None,:,:].expand(-1,13,13,-1).reshape(-1,169,512)
        r=self.pair(torch.cat([left,right,u,x['geometry']],-1))*x['interaction'][...,None]
        ks=self.ks(e);ko=self.ko(e)
        ks=torch.cat([ks[:,:12],self.null_keys[0][None,None].expand(len(e),1,-1)],1)
        ko=torch.cat([ko[:,:12],self.null_keys[1][None,None].expand(len(e),1,-1)],1)
        kr=torch.where(x['interaction'][...,None],self.kr(r),self.neutral_pair[None,None,:])
        pairvalid=(x['valid'][:,:,None]&x['valid'][:,None,:]).reshape(-1,169)
        return dict(e=e,r=r,ks=ks,ko=ko,kr=kr,valid=x['valid'],pairvalid=pairvalid)

    def read(self,h,bank,arm,off=False):
        e,r=bank['e'],bank['r']
        a=(self.qs(h)@bank['ks'].transpose(1,2)/math.sqrt(128)).masked_fill(~bank['valid'][:,None,:],-torch.inf)
        b=(self.qo(h)@bank['ko'].transpose(1,2)/math.sqrt(128)).masked_fill(~bank['valid'][:,None,:],-torch.inf)
        la,lb=a.log_softmax(-1),b.log_softmax(-1)
        s=self.qr(h)@bank['kr'].transpose(1,2)/math.sqrt(128)
        if off:s=torch.zeros_like(s);r=torch.zeros_like(r)
        joint=(la[:,:,:,None]+lb[:,:,None,:]).flatten(-2)+s
        t=joint.masked_fill(~bank['pairvalid'][:,None,:],-torch.inf).softmax(-1).reshape(len(h),h.shape[1],13,13)
        alpha,beta=la.exp(),lb.exp()
        ws,wo=(alpha,beta) if arm=='I' else (t.sum(-1),t.sum(-2))
        contexts=torch.cat([ws@e,wo@e,t.flatten(-2)@r],-1)
        return self.output(contexts),dict(alpha=alpha,beta=beta,joint=t)


class Captioner(nn.Module):
    def __init__(self,base,arm):
        super().__init__()
        if arm not in ('I','J'):raise ValueError('Unknown arm')
        self.base=base.requires_grad_(False);self.arm=arm
        self.teacher=copy.deepcopy(base.decoder).requires_grad_(False)
        for layer in self.base.decoder.transformer_decoder.layers[1:]:layer.requires_grad_(True)
        self.reader=PairReader()
        self.eval()  # SCST generation and rescoring use the identical no-dropout policy.

    @torch.no_grad()
    def inputs(self,b):
        memory,padding=self.base.build_memory(b['visual'],b['prompt'],b['mask'])
        patch=self.base.encoder.relu(self.base.encoder.vis_projection(b['visual'][:,1:]))
        return dict(memory=memory,padding=padding,patch=patch,
            **{k:b[k] for k in ('roi','union','boxes','confidence','geometry','valid','interaction')})

    def hidden(self,ids,x):
        d=self.base.decoder
        return d.transformer_decoder(d.pos_encoder(d.embedding(ids)*math.sqrt(d.d_model)),x['memory'],
            tgt_mask=d.causal_mask(ids.shape[1],ids.device),memory_key_padding_mask=x['padding'])

    def logits(self,ids,x,bank=None,last=False,off=False,diagnostics=False):
        h=self.hidden(ids,x)
        if last:h=h[:,-1:]
        if bank is None:bank=self.reader.bank(x)
        delta,trace=self.reader.read(h,bank,self.arm,off)
        logits=self.base.decoder.fc_out(h+delta)
        return (logits,trace) if diagnostics else logits

    @torch.no_grad()
    def teacher_logits(self,ids,x):
        return self.teacher(ids,x['memory'],memory_key_padding_mask=x['padding'])

    @torch.no_grad()
    def generate(self,x,tok,sample=False,baseline=False,off=False,trace=False):
        bank=None if baseline else self.reader.bank(x)
        ids=torch.full((len(x['memory']),1),tok.bos_idx,device=x['memory'].device,dtype=torch.long)
        live=torch.ones(len(ids),device=ids.device,dtype=torch.bool);masks=[];chosen=[]
        for _ in range(29):
            logits=(self.teacher_logits(ids,x)[:,-1] if baseline else self.logits(ids,x,bank,last=True,off=off)[:,0]).float()
            if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite rollout')
            token=torch.multinomial(logits.softmax(-1),1).flatten() if sample else logits.argmax(-1)
            token=torch.where(live,token,tok.pad_idx);masks.append(live.clone())
            if trace:chosen.append(logits.log_softmax(-1).gather(1,token[:,None]).flatten())
            ids=torch.cat([ids,token[:,None]],1);live=live&token.ne(tok.eos_idx)
            if not live.any():break
        result=(ids,torch.stack(masks,1))
        return result+(torch.stack(chosen,1),) if trace else result

    def relation_loss(self,b):
        x=self.inputs(b);logits=self.logits(b['ids'],x)
        chosen=logits.log_softmax(-1).gather(-1,b['targets'][...,None]).squeeze(-1)
        tail=(chosen*b['tail']).sum(-1);predicate=(chosen*b['predicate']).sum(-1)
        plus,minus=tail[::2],tail[1::2]
        loss=(-plus+F.softplus(minus-plus)).mean()
        return loss,dict(tail_gap=(plus-minus).detach(),predicate_gap=(predicate[::2]-predicate[1::2]).detach())
