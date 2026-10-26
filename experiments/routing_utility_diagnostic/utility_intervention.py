"""Exact learned MHA value reads; no replacement of text or complete prompt memory."""
import torch
from torch.nn import functional as F
from utility_data import coefficients


class Capture:
    def __init__(self,module):self.module=module
    def __enter__(self):
        def hook(module,args,kwargs,output):
            call=dict(kwargs)
            for name,value in zip(('query','key','value'),args):call[name]=value
            self.value=call['value'].detach()
            self.read=output[0].detach().clone()
        self.handle=self.module.register_forward_hook(hook,with_kwargs=True)
        return self
    def __exit__(self,*args):self.handle.remove()


def one_hot_reads(module,values):
    if not module.batch_first or not module._qkv_same_embed_dim:raise ValueError('Unsupported baseline MHA.')
    d=module.embed_dim;b=module.in_proj_bias
    projected=F.linear(values,module.in_proj_weight[2*d:],None if b is None else b[2*d:])
    return F.linear(projected,module.out_proj.weight,module.out_proj.bias)


class Shift:
    def __init__(self,module,positions,delta,coefficient):
        self.module,self.positions,self.delta,self.coefficient=module,positions,delta,coefficient
        if not 0<=coefficient<=.100000001:raise ValueError('Correction exceeds locked convex coefficient.')
    def __enter__(self):
        def hook(module,args,kwargs,output):
            if self.coefficient==0:return None
            result=output[0].clone()
            result[:,self.positions]+=self.coefficient*self.delta
            return result,output[1]
        self.handle=self.module.register_forward_hook(hook,with_kwargs=True)
        return self
    def __exit__(self,*args):self.handle.remove()


def correction_plan(module,values,read,positions,plus,minus):
    o=one_hot_reads(module,values);original=read[:,positions]
    dp=o[:,plus:plus+1].expand_as(original)-original
    dm=o[:,minus:minus+1].expand_as(original)-original
    norms=(float(dp.norm()),float(dm.norm()));weights=coefficients(*norms)
    if weights is None:return None
    lp,lm,h=weights
    return dict(T=(dp,lp),W=(dm,lm),positive_norm=norms[0],negative_norm=norms[1],matched_norm=h)


def synthetic_checks():
    """User-run tensor checks, not a local model test."""
    from route_probe import attention
    torch.manual_seed(42)
    m=torch.nn.MultiheadAttention(32,4,batch_first=True,dropout=0).cuda().eval()
    query=torch.randn(1,5,32,device='cuda');values=torch.randn(1,8,32,device='cuda')
    with torch.no_grad():
        baseline=m(query,values,values,need_weights=False)[0]
        o=one_hot_reads(m,values)
        # Every head selecting the same Q is exactly its concatenated learned value projection.
        v=F.linear(values,m.in_proj_weight[64:],m.in_proj_bias[64:]).reshape(1,8,4,8).transpose(1,2)
        weights=attention(m,query,values,values)
        plan=correction_plan(m,values,baseline,[1,2],2,6)
        assert plan is not None
        changes=[]
        for condition,q in [('T',2),('W',6)]:
            delta,lam=plan[condition]
            with Shift(m,[1,2],delta,lam):actual=m(query,values,values,need_weights=False)[0]
            mixed=weights.clone();mixed[:,:,1:3]*=(1-lam);mixed[:,:,1:3,q]+=lam
            expected=F.linear((mixed@v).transpose(1,2).reshape(1,5,32),m.out_proj.weight,m.out_proj.bias)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
            assert torch.equal(actual[:,[0,3,4]],baseline[:,[0,3,4]])
            changes.append((actual-baseline).norm())
        torch.testing.assert_close(changes[0],changes[1],rtol=1e-5,atol=1e-6)
        with Shift(m,[1,2],None,0):zero=m(query,values,values,need_weights=False)[0]
        assert torch.equal(zero,baseline)
        forced=F.linear(v[:,:,2:3].transpose(1,2).reshape(1,1,32),m.out_proj.weight,m.out_proj.bias)
        torch.testing.assert_close(o[:,2:3],forced,rtol=1e-5,atol=1e-6)
    print('MHA one-hot/convex output equivalence, untouched rows, matched norm, zero parity PASS',flush=True)
