"""Observe existing eval-mode cross-attention; never replace its output or add parameters."""
import math
import torch
from torch.nn import functional as F

RTOL, ATOL = 1e-5, 1e-6


def attention(module, query, key, value, key_padding_mask=None, attn_mask=None, **kwargs):
    if module.training or not module.batch_first or not module._qkv_same_embed_dim:
        raise ValueError('Probe supports this baseline in eval, batch-first, equal-dimension MHA only.')
    if module.bias_k is not None or module.bias_v is not None or module.add_zero_attn:
        raise ValueError('Unexpected extra keys in baseline attention.')
    if attn_mask is not None or kwargs.get('is_causal',False):
        raise ValueError('Cross-attention must have no causal/memory attn_mask; decoder self-attention is untouched.')
    if query.ndim!=3 or key.ndim!=3 or value.shape!=key.shape:
        raise ValueError('Unexpected MHA input shape.')
    width=module.embed_dim; heads=module.num_heads; dim=width//heads
    bias=module.in_proj_bias
    q=F.linear(query,module.in_proj_weight[:width],None if bias is None else bias[:width])
    k=F.linear(key,module.in_proj_weight[width:2*width],None if bias is None else bias[width:2*width])
    q=q.reshape(q.shape[0],q.shape[1],heads,dim).transpose(1,2)
    k=k.reshape(k.shape[0],k.shape[1],heads,dim).transpose(1,2)
    scores=torch.matmul(q/math.sqrt(dim),k.transpose(-1,-2))
    if key_padding_mask is not None:
        if key_padding_mask.shape!=(query.shape[0],key.shape[1]):
            raise ValueError('Wrong cross-attention padding shape.')
        mask=key_padding_mask[:,None,None,:]
        if mask.dtype==torch.bool: scores=scores.masked_fill(mask,-torch.inf)
        elif mask.is_floating_point(): scores=scores+mask
        else: raise ValueError('Unsupported padding dtype.')
    result=scores.softmax(-1)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Nonfinite observation; reject all-masked cross attention.')
    return result


class RouteProbe:
    def __init__(self, model):
        self.model=model; self.weights={}; self.handles=[]

    def __enter__(self):
        sites={'decoder':self.model.decoder.transformer_decoder.layers[-1].multihead_attn,
               'qformer':self.model.encoder.qformer.layers[-1].cross_attn}
        for name,module in sites.items():
            def observe(module,args,kwargs,name=name):
                fields=('query','key','value')
                call=dict(kwargs)
                for i,arg in enumerate(args):
                    if i>=3: raise ValueError('Unexpected positional MHA options.')
                    call[fields[i]]=arg
                self.weights[name]=attention(module,**call)
                # A pre-hook returning None preserves both the actual call and its output.
                return None
            self.handles.append(module.register_forward_pre_hook(observe,with_kwargs=True))
        return self

    def clear(self): self.weights.clear()

    def __exit__(self,*args):
        for handle in self.handles: handle.remove()
        self.handles.clear(); self.weights.clear()


def route(decoder_weights,qformer_weights,legacy_length,mask):
    d=decoder_weights.mean(1); c=qformer_weights.mean(1)
    if d.shape[-1]!=legacy_length+32 or c.shape[1:]!=(32,197):
        raise ValueError('Only unchanged [legacy;32Q] and CLS+196 patch keys are supported.')
    if mask.shape!=(d.shape[0],196) or bool(((mask<0)|(mask>1)).any()):
        raise ValueError('Expected batch x196 fractional union masks in [0,1].')
    dq=d[:,:,legacy_length:]; paths=torch.bmm(dq,c[:,:,1:])
    return dict(decoder_q_mass=dq.sum(-1),patch_path_mass=paths.sum(-1),
        cls_path_mass=torch.bmm(dq,c[:,:,:1]).squeeze(-1),
        bbox_mass=(paths*mask[:,None,:]).sum(-1),paths=paths)


def numeric_checks(device):
    """Synthetic checks only; run by the user on Kaggle, not during notebook creation."""
    torch.manual_seed(17)
    mha=torch.nn.MultiheadAttention(32,4,batch_first=True,dropout=.1).to(device).eval()
    q=torch.randn(2,5,32,device=device,requires_grad=True)
    k=torch.randn(2,11,32,device=device,requires_grad=True)
    pad=torch.zeros(2,11,dtype=torch.bool,device=device);pad[0,-3:]=True
    for padding in (pad,torch.zeros_like(pad,dtype=torch.float).masked_fill(pad,-torch.inf)):
        _,expected=mha(q,k,k,key_padding_mask=padding,need_weights=True,average_attn_weights=False)
        actual=attention(mha,q,k,k,key_padding_mask=padding)
        torch.testing.assert_close(actual,expected,rtol=RTOL,atol=ATOL)
        assert bool((actual[0,:,:,-3:]==0).all())
    (-actual[0,0,0,0].log()).backward()
    assert mha.in_proj_weight.grad is not None and float(mha.in_proj_weight.grad.norm())>0
    # Controlled two-path distribution: .5 decoder mass on Q, .75 Q mass on patches.
    d=torch.zeros(1,1,2,34,device=device);d[:,:,:,:2]=.25;d[:,:,:,2:]=.5/32
    c=torch.zeros(1,1,32,197,device=device);c[:,:,:,0]=.25;c[:,:,:,1:]=.75/196
    full=route(d,c,2,torch.ones(1,196,device=device))
    empty=route(d,c,2,torch.zeros(1,196,device=device))
    torch.testing.assert_close(full['bbox_mass'],torch.full((1,2),.375,device=device),rtol=RTOL,atol=ATOL)
    assert bool((empty['bbox_mass']==0).all())
    torch.testing.assert_close(full['decoder_q_mass'],full['patch_path_mass']+full['cls_path_mass'],rtol=RTOL,atol=ATOL)
    # Q/patch slices must retain missing legacy/CLS mass rather than become one.
    assert bool((full['patch_path_mass']<full['decoder_q_mass']).all())
    return dict(status='SYNTHETIC_PASS',device=str(device),rtol=RTOL,atol=ATOL,
        public_mha_weights_checked=True,bool_and_float_padding_checked=True)
