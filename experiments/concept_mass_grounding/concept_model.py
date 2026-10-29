"""Native A decoder and exact prefix-free lexical event mass; user GPU only."""
import copy
import torch
from torch import nn
from concept_data import POLICY

class NativeCaptioner(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base.requires_grad_(False);self.teacher=copy.deepcopy(base.decoder).requires_grad_(False)
        for layer in base.decoder.transformer_decoder.layers[1:]:layer.requires_grad_(True)
        self.eval()
    @torch.no_grad()
    def inputs(self,b):return dict(zip(('memory','padding'),self.base.build_memory(b['visual'],b['prompt'],b['mask'])))
    def logits(self,ids,x,teacher=False):
        d=self.teacher if teacher else self.base.decoder
        return d(ids,x['memory'],memory_key_padding_mask=x['padding'])
    @torch.no_grad()
    def generate(self,x,tok,sample=False,teacher=False,trace=False):
        ids=torch.full((len(x['memory']),1),tok.bos_idx,device=x['memory'].device,dtype=torch.long)
        live=torch.ones(len(ids),device=ids.device,dtype=torch.bool);masks=[];chosen=[]
        for _ in range(29):
            logits=self.logits(ids,x,teacher)[:,-1].float()
            if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite policy')
            next_token=torch.multinomial(logits.softmax(-1),1).flatten() if sample else logits.argmax(-1)
            next_token=torch.where(live,next_token,tok.pad_idx);masks.append(live.clone())
            if trace:chosen.append(logits.log_softmax(-1).gather(-1,next_token[:,None]).flatten())
            ids=torch.cat([ids,next_token[:,None]],1);live=live&next_token.ne(tok.eos_idx)
            if not live.any():break
        result=(ids,torch.stack(masks,1))
        return result+(torch.stack(chosen,1),) if trace else result
    def class_logmass(self,sequence,x,events,lex,root_logits,teacher=False):
        if not events:return root_logits.new_empty(0)
        lp=root_logits.float().log_softmax(-1);terms=[];branch=[]
        for event_index,event in enumerate(events):
            b,p,c=event['batch'],event['position'],event['category']
            for path in lex['events'][c]:
                if len(path)>POLICY['max_len']-1-p:continue
                value=lp[b,p,path[0]];term_index=len(terms);terms.append((event_index,value))
                if len(path)>1:
                    prefix=sequence[b,:p+1].detach().tolist()
                    branch.append((term_index,b,p,prefix+path[:-1],path[1:]))
        # Chunk size affects memory only. Every lexical branch/event is retained.
        for first in range(0,len(branch),POLICY['event_chunk']):
            group=branch[first:first+POLICY['event_chunk']];length=max(len(r[3]) for r in group)
            ids=sequence.new_zeros((len(group),length));batch=[]
            for j,(_,b,_,prefix,_) in enumerate(group):ids[j,:len(prefix)]=sequence.new_tensor(prefix);batch.append(b)
            ix=sequence.new_tensor(batch);xx={k:v.index_select(0,ix) for k,v in x.items()}
            logp=self.logits(ids,xx,teacher).float().log_softmax(-1)
            for j,(term_index,_,p,_,tail) in enumerate(group):
                event_index,value=terms[term_index]
                value=value+torch.stack([logp[j,p+1+k,token] for k,token in enumerate(tail)]).sum()
                terms[term_index]=(event_index,value)
        values=[]
        for i in range(len(events)):
            value=torch.logsumexp(torch.stack([v for e,v in terms if e==i]),0)
            if value.detach()>1e-5:raise ValueError('Alias union exceeds probability one')
            values.append(value.clamp_max(-torch.finfo(value.dtype).eps))
        return torch.stack(values)

def negative_losses(logq):
    # Stable log(1-exp(logq)), without probability/length normalization.
    return -torch.where(logq < -.6931471805599453,torch.log1p(-logq.exp()),torch.log(-torch.expm1(logq)))

def global_mean(local_sum,local_count):
    import torch.distributed as dist
    count=local_sum.new_tensor(float(local_count));dist.all_reduce(count)
    # Explicit gradient averaging follows this: multiply by world size to recover global event mean.
    return local_sum*(dist.get_world_size()/count.clamp_min(1)),int(count.item())
