"""Runtime checks inside each pilot notebook; zero optimizer updates."""
from datetime import timedelta
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pilot_data import save
from route_probe import RouteProbe, route, numeric_checks, RTOL, ATOL
from routing_model import global_span_mean, auxiliary


def ddp_checks(rank,local):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.p=torch.nn.Parameter(torch.tensor(2.,device=local))
        def forward(self,values):
            return global_span_mean(self.p*values.sum(),len(values))[0]
    toy=DDP(Toy(),device_ids=[local])
    for values,expected in [(([1.,1.,1.],[7.]),2.5),(([2.,2.,2.],[]),2.),(([],[]),0.)]:
        toy.zero_grad(set_to_none=True)
        loss=toy(torch.tensor(values[rank],device=local));loss.backward()
        torch.testing.assert_close(toy.module.p.grad,torch.tensor(expected,device=local),rtol=0,atol=0)
    del toy
    # Token mean within the span, then a global span mean (not log of the mean mass).
    paths=torch.zeros(5,2,196,device=local,requires_grad=True)
    scale=torch.tensor([.4,.8],device=local)[None,:,None]
    paths=paths+scale/196
    batch=dict(span_valid=torch.tensor([[True]],device=local),span_caption=torch.tensor([[3]],device=local),
        span_positions=torch.tensor([[[0,1]]],device=local),span_token_valid=torch.tensor([[[True,True]]],device=local),
        span_masks=torch.full((1,1,196),.5,device=local),span_person=torch.tensor([[False]],device=local),
        image_id=torch.tensor([12],device=local))
    parts=dict(paths=paths,decoder_q_mass=paths.sum(-1),cls_path_mass=paths.sum(-1)*0)
    for arm,factor in [('B',1.),('R',.5)]:
        loss,_=auxiliary(parts,batch,arm)
        expected=-(torch.tensor([.4,.8],device=local)*factor+1e-8).log().mean()
        torch.testing.assert_close(loss,expected,rtol=RTOL,atol=ATOL)


def check_teacher_expansion(observed,memory,padding,targets,masks):
    if not torch.equal(observed['targets'],targets) or not torch.equal(observed['masks'],masks):
        raise ValueError('Teacher caption tokens/padding changed.')
    if observed['memory'].shape[0]!=len(memory)*5 or observed['padding'].shape[0]!=len(memory)*5:
        raise ValueError('Expected exactly five references per image.')
    for image in range(len(memory)):
        for caption in range(5):
            row=image*5+caption
            if not torch.equal(observed['memory'][row],memory[image]) or not torch.equal(observed['padding'][row],padding[image]):
                raise ValueError('Teacher memory is not image-major five-reference expansion.')


def _preflight(protocol):
    from captioning.scst_runtime import configure_strict_runtime, model_digest, runtime_manifest
    from run_pilot import corpus,cuda_batch,seed_all
    from run_routing import loader,build
    cfg=json.loads(Path(protocol).read_text());out=Path(cfg['output'])
    rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('Use T4x2 for one arm.')
    torch.cuda.set_device(local);torch.set_num_threads(2);configure_strict_runtime(True)
    if torch.cuda.get_device_name(local)!='Tesla T4':raise ValueError('Locked hardware: T4x2.')
    dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    checks=[numeric_checks(torch.device('cpu')),numeric_checks(torch.device('cuda',local))]
    ddp_checks(rank,local)
    seed_all(42)
    data,tok=corpus(cfg)
    batches=loader(cfg,data,tok,'train',rank,2)
    # Deterministically take two supervised TRAIN rows; sampler state is not reused for training.
    chosen=[]
    for i in range(len(batches.dataset)):
        if batches.dataset[i]['span_valid'].any():chosen.append(i)
        if len(chosen)==2:break
    from torch.utils.data._utils.collate import default_collate
    batch=cuda_batch(default_collate([batches.dataset[i] for i in chosen]))
    model=build(cfg,tok);before=model_digest(model)
    assert set(model.state_dict())=={'base.'+k for k in model.base.state_dict()}
    with RouteProbe(model.base) as probe:
        memory,padding=model.memory(batch)
        targets=batch['captions'].flatten(0,1)
        mask=batch['caption_mask'].flatten(0,1)
        observed={}
        def capture(module,args,kwargs):
            observed.update(targets=args[0],memory=args[1],masks=kwargs['tgt_key_padding_mask'],
                            padding=kwargs['memory_key_padding_mask'])
        handle=model.decoder.register_forward_pre_hook(capture,with_kwargs=True)
        try:
            logits=model.teacher(memory,padding,targets[:,:-1],mask[:,:-1])
        finally:handle.remove()
        check_teacher_expansion(observed,memory,padding,targets[:,:-1],mask[:,:-1])
        dw=probe.weights['decoder'];cw=probe.weights['qformer']
        assert dw.shape[0]==len(memory)*5 and cw.shape[0]==len(memory)
        # Check actual decoder inputs exactly. Different GEMM batch sizes need not
        # produce matching FP32 logits; same-batch observer parity is checked below.
        parts=route(dw,cw.repeat_interleave(5,0),batch['legacy'].shape[1],dw.new_ones((len(dw),196)))
        aux,_=auxiliary(parts,batch,cfg['arm'])
        aux.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads) or not any(float(g.norm())>0 for g in grads):
        raise ValueError('Nonfinite/zero auxiliary gradient.')
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        memory,padding=model.memory(batch)
        plain=model.teacher(memory,padding,targets[:,:-1],mask[:,:-1])
    torch.testing.assert_close(plain,logits.detach(),rtol=RTOL,atol=ATOL)
    loss,ce,_=model(batch,None,None,None,0.,tok.pad_idx)
    torch.testing.assert_close(ce,torch.nn.functional.cross_entropy(plain.flatten(0,1),targets[:,1:].reshape(-1),ignore_index=tok.pad_idx),rtol=RTOL,atol=ATOL)
    loss.backward()
    if model_digest(model)!=before:raise ValueError('Preflight changed weights.')
    save(out/f'preflight_rank{rank}.json',dict(status='PREFLIGHT_PASS',rank=rank,optimizer_updates=0,
        synthetic=checks,ddp_uneven_and_zero_counts=True,multiword_mean_checked=True,
        five_reference_alignment_checked=True,expansion_check='exact actual decoder inputs; image-major five references',
        cross_batch_logits_comparison=False,same_batch_logits_parity_checked=True,
        ce_parity_checked=True,finite_nonzero_aux_gradients=True,
        no_added_parameters=True,state_sha256=before,runtime=runtime_manifest()))
    dist.destroy_process_group()
    print('PREFLIGHT PASS, rank',rank,'zero optimizer updates',flush=True)


def preflight(protocol):
    try:
        _preflight(protocol)
    finally:
        if dist.is_initialized():dist.destroy_process_group()
