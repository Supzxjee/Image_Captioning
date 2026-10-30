"""User GPU-only actual-path parity, construction and connectivity checks."""
import math
import torch
from torch.nn import functional as F
from kv_pilot_data import PARTNER


def preflight(model,data,tok,device,rank):
    for name,p in model.named_parameters():
        expected = name.startswith(('reader.','base.decoder.transformer_decoder.layers.1.',
                                     'base.decoder.transformer_decoder.layers.2.'))
        if p.requires_grad != expected:raise ValueError('Wrong trainable scope '+name)
    eligible = [i for i in range(4000) if data.arrays['pair_valid'][i].any()]
    indices = eligible[:2]
    if len(indices) != 2:raise ValueError('No suitable training preflight images')
    x = model.inputs(data.batch(indices,device))
    seq = torch.stack([tok.encode(data.meta['rows'][i]['references'][0],max_len=30)[0] for i in indices]).to(device)
    with torch.no_grad():
        original = model.logits(seq[:,:-1],x,teacher=True)
        actual = model.logits(seq[:,:-1],x)
        error = float((actual-original).abs().max())
        torch.testing.assert_close(actual,original,atol=1e-5,rtol=1e-5)
        a,_ = model.generate(x,tok,teacher=True)
        b,_ = model.generate(x,tok)
        if not torch.equal(a,b):raise ValueError('Initial C0 greedy parity failed')
        torch.manual_seed(731+rank)
        sample,active,trace = model.generate(x,tok,sample=True,trace=True)
        lp = model.logits(sample[:,:-1],x).log_softmax(-1).gather(-1,sample[:,1:,None]).squeeze(-1)
        policy_error = float((lp[active]-trace[active]).abs().max())
        torch.testing.assert_close(lp[active],trace[active],atol=1e-5,rtol=1e-5)
        reader = model.reader
        heads = torch.randn(2,7,512,device=device)
        partner = torch.tensor(PARTNER+[90],device=device)
        # Run E and D with exactly the same learned modules, test context, not zero Wo.
        saved_arm = reader.arm
        construction = {}
        for arm in ('D','E'):
            reader.arm = arm
            keys,values,valid = reader.bank(x['features'])
            c1 = reader.context(heads,(keys,values,valid))
            c2 = reader.context(heads,(keys[:,partner],values,valid))
            delta = float((c1-c2).abs().max())
            construction[arm] = delta
            if arm == 'E':
                torch.testing.assert_close(values[:,:90],values[:,:90][:,reader.partner],atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(c1,c2,atol=1e-6,rtol=1e-5)
            elif delta <= 1e-7:raise ValueError('Control context was vacuous/degenerate')
        reader.arm = saved_arm
        # Nonzero temporary readout avoids a null-path test passing merely because Wo=0.
        torch.nn.init.xavier_uniform_(reader.o.weight)
        empty = dict(x,features=dict(x['features'],valid=torch.zeros_like(x['features']['valid'])))
        null_logits = model.logits(seq[:,:-1],empty)
        torch.testing.assert_close(null_logits,original,atol=1e-5,rtol=1e-5)
    model.zero_grad(set_to_none=True)
    logits = model.logits(seq[:,:-1],x)
    F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=tok.pad_idx).backward()
    connectivity = {}
    for prefix in ('reader.visual.','reader.tp.','reader.wg.','reader.f.','reader.q.','reader.k.',
                   'reader.v.','reader.o.','base.decoder.transformer_decoder.layers.1.',
                   'base.decoder.transformer_decoder.layers.2.'):
        norm = sum(float(p.grad.detach().abs().sum()) for n,p in model.named_parameters()
                   if n.startswith(prefix) and p.grad is not None)
        if not math.isfinite(norm) or norm <= 0:raise ValueError('Caption gradient unreachable '+prefix)
        connectivity[prefix] = norm
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise ValueError('Frozen tensors received gradients')
    model.zero_grad(set_to_none=True)
    facts = data.aux['partitions']['fit'][:4]
    scores = reader.fact_scores(reader.features(data.fact_batch(facts,device)),facts)
    labels = scores.new_tensor([r['label'] for r in facts])[:,None].expand_as(scores)
    F.binary_cross_entropy_with_logits(scores,labels).backward()
    for prefix in ('reader.ts.','reader.tobj.','reader.text_mlp.','reader.tp.','reader.f.'):
        norm = sum(float(p.grad.detach().abs().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)
        if not math.isfinite(norm) or norm <= 0:raise ValueError('Auxiliary gradient unreachable '+prefix)
        connectivity[prefix+'aux'] = norm
    model.zero_grad(set_to_none=True)
    return dict(status='PREFLIGHT_PASS',max_initial_logit_error=error,max_policy_rescore_error=policy_error,
        key_only_swap_context_errors=construction,null_only_native_parity=True,temporary_CE_not_training=True,
        connectivity=connectivity,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        new_parameters=sum(p.numel() for p in reader.parameters()),numeric_reference='FP32; no AMP')


@torch.no_grad()
def intervention(model,data,tok,indices,device):
    records = []
    for first in range(0,len(indices),8):
        group = indices[first:first+8]
        x = model.inputs(data.batch(group,device))
        seq,_ = model.generate(x,tok)
        active = seq[:,1:].ne(tok.pad_idx)
        logits = model.logits(seq[:,:-1],x).float()
        norm = model.last_read_stats['contribution_norm_mean']
        off = model.logits(seq[:,:-1],x,reader_off=True).float()
        logp,logq = logits.log_softmax(-1),off.log_softmax(-1)
        kl = ((logp.exp()*(logp-logq)).sum(-1)*active).sum(-1)/active.sum(-1).clamp_min(1)
        for i,index in enumerate(group):
            records.append(dict(coco_id=data.meta['rows'][index]['coco_id'],reader_off_logit_kl=float(kl[i]),
                batch_mean_contribution_norm=float(norm)))
    return dict(records=records,scope='Zero Wo(context), keep adapted decoder; not raw C0')
