"""Factory/context helpers; Torch imported only by user-run functions."""
from pathlib import Path
import numpy as np
from semantic_data import SemanticContext

BASE_INIT='5c580d0dfbd0ed828b7697d9ead7575dc7f069e8e3ca51acc776e6b23d9c06bd'

def make_model(vocab,cfg):
    from patch_model import PatchCaptioner
    if not cfg.get('semantic'):return PatchCaptioner(vocab)
    import torch
    from semantic_patch_model import SemanticPatchCaptioner
    from full_runtime import tensor_digest
    banks=[torch.from_numpy(np.load(Path(cfg['stage'])/('sem_'+kind+'_bank.npy'),allow_pickle=False)) for kind in ('noun','verb')]
    model=SemanticPatchCaptioner(vocab,*banks)
    if tensor_digest(model.base)!=BASE_INIT:raise ValueError('Core baseline initialization differs from199')
    return model

def context_for(cfg):return SemanticContext(cfg['stage']) if cfg.get('semantic') else None

def neural_guards(model,x,p):
    """Actual GPU checks before training, including nonzero adapter mask locality."""
    import copy
    import torch
    from full_runtime import tensor_digest
    if tensor_digest(model.base)!=BASE_INIT:raise ValueError('Changed baseline core')
    with torch.no_grad():
        expected=model.base.encode(x);actual=model.encode(x,p)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for module,k in ((model.object_prompt,8),(model.relation_prompt,3)):
            test=copy.deepcopy(module).eval()
            torch.nn.init.eye_(test.attention.out_proj.weight);test.attention.out_proj.bias.fill_(.1)
            visual=expected[:2];slots=torch.randn(2,k,test.slot[0].in_features,device=x.device);support=torch.zeros(2,k,196,device=x.device)
            valid=torch.zeros(2,k,dtype=torch.bool,device=x.device);support[0,0,0]=1;valid[0,0]=True
            result=test(visual,slots,support,valid)
            if not torch.isfinite(result).all():raise ValueError('Nonfinite empty-slot attention')
            torch.testing.assert_close(result[1],visual[1],rtol=0,atol=0)
            torch.testing.assert_close(result[0,1:],visual[0,1:],rtol=0,atol=0)
            if torch.equal(result[0,0],visual[0,0]):raise ValueError('Nonzero adapter inactive on supported patch')
            perturbed=slots.clone();perturbed[:,1:]+=100
            torch.testing.assert_close(test(visual,perturbed,support,valid),result,rtol=0,atol=0)
            changed=slots.clone();changed[0,0,:512]+=1
            if torch.allclose(test(visual,changed,support,valid)[0,0],result[0,0],rtol=0,atol=1e-7):raise ValueError('Prompt text channel has no effect with active output')
            del test
    return dict(core_initial_sha256=BASE_INIT,initial_memory_equal_baseline=True,empty_slot_finite_identity=True,nonzero_adapter_mask_locality=True,
        trainable_parameters=sum(v.numel() for v in model.parameters()),frozen_bank_values=model.noun_bank.numel()+model.verb_bank.numel())
