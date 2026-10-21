"""User-run CSA two-phase pilot. No training or model loading on import."""
import argparse
from collections import defaultdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np

from pilot_data import digest, identity, save
from hidden_data import INPUT_HASHES, COHORT_SHA, locked_groups
from phase_policy import UPDATES, ADAM, schedule, contract, freeze_base, parameter_groups


def verify_sources(cfg):
    for name, expected in cfg['implementation_sha256'].items():
        if digest(Path(__file__).parent/name) != expected:
            raise ValueError('Source differs: '+name)


def prepare(config):
    import h5py
    import torch
    from captioning.visual_cache import VisualCache
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.object_reward import make_targets
    cfg = json.loads(Path(config).read_text())
    verify_sources(cfg)
    out, stage = Path(cfg['output']), Path(cfg['stage'])
    if (out/'protocol.json').exists() or stage.exists():
        raise FileExistsError('Preserve existing stage/protocol; use fresh output/session.')
    begin = time.monotonic()
    for key, expected in INPUT_HASHES.items():
        print('Hashing pinned input:',key,flush=True)
        if digest(cfg[key]) != expected:
            raise ValueError('Pinned input differs: '+key)
    cfg['input_sha256'] = INPUT_HASHES
    dataset = json.loads(Path(cfg['dataset_json']).read_text())
    groups = locked_groups(dataset)
    rows = groups['train']+groups['val']
    cfg.update(groups=groups,cohort_sha256=COHORT_SHA)
    cfg['contract_sha256'] = contract(cfg)
    save(out/'groups.json',groups)
    targets = make_targets(groups,dataset,Path(cfg['chair'])/'chair.pkl',
                           json.loads(Path(cfg['detections']).read_text()),'aliases_v2','unique_count_v2')
    save(out/'train_object_targets.json',targets)
    print('Loading legacy tokens once; staging CLIP/legacy only, no UPT features.',flush=True)
    bundle = torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
    names = [str(Path('/images')/dataset['images'][r['image_id']]['filepath']/r['filename']) for r in rows]
    legacy = align_prompt_cache(bundle['data'],names)
    if len(legacy) != len(rows):
        raise ValueError('Missing legacy prompt rows.')
    length = legacy[names[0]]['tokens'].shape[0]
    required = len(rows)*(197*768*4+length*512*4+length*8)
    stage.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(stage.parent).free < required+2*1024**3:
        raise RuntimeError('Insufficient stage disk space plus2GiB reserve.')
    cache = VisualCache([cfg['visual_cache']],id_key='coco_id')
    try:
        cache.require_ids([r['coco_id'] for r in rows],'CSA train10k/dev1000')
        by_file = defaultdict(list)
        for index,row in enumerate(rows):
            file_id,position = cache.index[row['coco_id']]
            by_file[file_id].append((position,index))
        with h5py.File(stage,'w') as f:
            for key,shape,dtype in [('visual',(len(rows),197,768),'f4'),('legacy',(len(rows),length,512),'f4'),
                                    ('legacy_mask',(len(rows),length),'i8')]:
                f.create_dataset(key,shape=shape,dtype=dtype,chunks=(1,*shape[1:]))
            f.create_dataset('image_ids',data=np.asarray([r['coco_id'] for r in rows]))
            done = 0
            for file_id,positions in sorted(by_file.items()):
                # Keep the source dataset handle alive, avoiding metadata lookup per image.
                with h5py.File(cache.paths[file_id],'r') as source:
                    features = source['features']
                    for position,index in sorted(positions):
                        old = legacy[names[index]]
                        tokens,mask = old['tokens'].float().numpy(),old['mask'].long().numpy()
                        visual = features[position].astype(np.float32)
                        if tokens.shape != (length,512) or mask.shape != (length,):
                            raise ValueError('Legacy shape differs.')
                        if not np.isfinite(visual).all() or not np.isfinite(tokens).all() or not np.isin(mask,[0,1]).all():
                            raise ValueError('Invalid staged values.')
                        for key,value in [('visual',visual),('legacy',tokens),('legacy_mask',mask)]:
                            f[key][index] = value
                        done += 1
                        if done == 1 or done % 250 == 0:
                            print(f'Staged exact CLIP/legacy: {done}/{len(rows)}',flush=True)
            f.attrs.update(complete=True,cohort_sha256=COHORT_SHA)
        cfg.update(stage_sha256=digest(stage),object_targets_sha256=digest(out/'train_object_targets.json'),
                   preparation_seconds=time.monotonic()-begin)
    finally:
        cache.close()
    save(out/'protocol.json',cfg)
    print('Preparation complete; starting two-phase user-run pilot.',flush=True)


def loader(cfg,dataset,tokenizer,split,rank):
    import h5py
    import torch
    from torch.utils.data import Dataset,DataLoader,DistributedSampler,Subset
    rows = cfg['groups'][split]
    offset = 0 if split == 'train' else 10000
    class Staged(Dataset):
        def __init__(self):
            self.f = h5py.File(cfg['stage'],'r')
            if not self.f.attrs.get('complete') or self.f.attrs['cohort_sha256'] != COHORT_SHA:
                raise ValueError('Incomplete/different stage.')
            # Cache dataset handles for the whole loader lifetime.
            self.arrays = {k:self.f[k] for k in ('visual','legacy','legacy_mask','image_ids')}
        def __len__(self):
            return len(rows)
        def __getitem__(self,index):
            row = rows[index]
            i = offset+index
            if int(self.arrays['image_ids'][i]) != row['coco_id']:
                raise ValueError('Stage image identity differs.')
            result = {k:torch.from_numpy(self.arrays[k][i]) for k in ('visual','legacy','legacy_mask')}
            encoded = [tokenizer.encode(s['raw']) for s in dataset['images'][row['image_id']]['sentences'][:5]]
            if len(encoded) != 5:
                raise ValueError('Expected five references.')
            result.update(captions=torch.stack([e[0] for e in encoded]),caption_mask=torch.stack([e[1] for e in encoded]),
                          image_id=torch.tensor(row['image_id']))
            return result
    staged = Staged()
    if split == 'train':
        sampler = DistributedSampler(staged,num_replicas=2,rank=rank,shuffle=True,seed=42,drop_last=False)
        return DataLoader(staged,batch_size=4,sampler=sampler,num_workers=0,pin_memory=True,drop_last=False)
    return DataLoader(Subset(staged,list(range(rank,len(rows),2))),batch_size=16,num_workers=0,pin_memory=True)


def predict(model,batches,tokenizer,condition):
    import torch
    from captioning.scst import rollout,decode
    from run_pilot import cuda_batch
    records = []
    with torch.no_grad():
        for i,batch in enumerate(batches):
            batch = cuda_batch(batch)
            memory,padding = model.memory(batch,condition)
            seq,_ = rollout(model,memory,padding,tokenizer,sample=False)
            records.extend(dict(image_id=iid,caption=text) for iid,text in
                           zip(batch['image_id'].cpu().tolist(),decode(seq,tokenizer)))
            if i % 10 == 0:
                print(f'Evaluation {condition}: {len(records)} local images',flush=True)
    return records


def worker(config,phase):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from captioning.scst import TrainingCider,rollout,decode
    from captioning.object_reward import TrainingObjectReward,combine_rewards
    from captioning.scst_runtime import configure_strict_runtime,runtime_manifest,model_digest,tensor_digest
    from csa_core import CSACore,load_banks
    from csa_model import CSAModel
    from run_pilot import corpus,build_model,seed_all,cuda_batch
    cfg = json.loads(Path(config).read_text())
    verify_sources(cfg)
    if cfg['contract_sha256'] != contract(cfg):
        raise ValueError('Recipe contract differs.')
    rank,local = int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 2:
        raise RuntimeError('One arm requires T4x2.')
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    configure_strict_runtime(True)
    dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    out = Path(cfg['output'])
    try:
        if digest(cfg['checkpoint']) != INPUT_HASHES['checkpoint'] or digest(cfg['stage']) != cfg['stage_sha256']:
            raise ValueError('Checkpoint/stage changed.')
        if digest(out/'train_object_targets.json') != cfg['object_targets_sha256']:
            raise ValueError('Reward targets changed.')
        seed_all(42)
        dataset,tokenizer = corpus(cfg)
        base = build_model(dict(cfg,arm='A'),tokenizer).base
        core = CSACore(Path(cfg['reference'])/'csa_net.py',load_banks(cfg['reference']))
        model = CSAModel(base,core).cuda().eval()
        initial_base = model_digest(base)
        banks_before = {n:tensor_digest(b) for n,b in core.named_buffers()}
        if phase == 2:
            checkpoint = torch.load(cfg['phase1_checkpoint'],map_location='cpu',weights_only=False)
            if checkpoint['contract_sha256'] != cfg['contract_sha256'] or checkpoint['phase'] != 1 or checkpoint['updates'] != UPDATES:
                raise ValueError('Phase1 checkpoint contract differs.')
            if checkpoint['caption_word2idx'] != tokenizer.word2idx or checkpoint['initial_base_sha256'] != initial_base:
                raise ValueError('Phase1 baseline/vocabulary differs.')
            if not checkpoint['baseline_unchanged'] or not checkpoint['banks_unchanged']:
                raise ValueError('Phase1 freeze verification absent.')
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            if model_digest(model) != checkpoint['final_full_sha256']:
                raise ValueError('Phase1 checkpoint state identity differs.')
            if model_digest(base) != initial_base:
                raise ValueError('Phase1 changed the frozen baseline.')
            if {n:tensor_digest(b) for n,b in core.named_buffers()} != banks_before:
                raise ValueError('Phase1 changed frozen semantic banks.')
            del checkpoint
            save(out/f'phase2_transition_rank{rank}.json',dict(
                 phase1_checkpoint_sha256=digest(cfg['phase1_checkpoint']),baseline_unchanged=True,
                 phase1_final_equals_phase2_initial=True,optimizer_reset=True,sampler_epoch=0,
                 rank_seed=42+rank,banks_unchanged=True))
        phase_initial_full = model_digest(model)
        train = loader(cfg,dataset,tokenizer,'train',rank)
        if len(train) != UPDATES:
            raise ValueError('Expected1250 updates per phase.')
        train_order = [cfg['groups']['train'][i]['image_id'] for i in train.sampler]
        save(out/f'phase{phase}_order_rank{rank}.json',train_order)
        if phase == 2:
            validation = loader(cfg,dataset,tokenizer,'val',rank)
            evaluation_begin = time.monotonic()
            save(out/f'origin_rank{rank}.json',predict(model,validation,tokenizer,'off'))
            save(out/f'origin_timing_rank{rank}.json',dict(seconds=time.monotonic()-evaluation_begin))
            refs = {i:[s['raw'] for s in row['sentences'][:5]] for i,row in enumerate(dataset['images'])
                    if row['split'] in ('train','restval')}
            print(f'[rank {rank}] Building full TRAIN IDF: {len(refs)} images.',flush=True)
            reward = TrainingCider(refs,reference_cache_size=256)
            objects = TrainingObjectReward(json.loads((out/'train_object_targets.json').read_text())['data'],'aliases_v2','unique_count_v2')
        freeze_base(model,phase)
        # Each phase is a separate process: no stale DDP hooks or optimizer state.
        # Match A's rank-specific seed before optimizer/DDP and train iterator creation.
        seed_all(42+rank)
        optimizer = torch.optim.AdamW(parameter_groups(model,phase),**ADAM)
        assert len(optimizer.state) == 0
        save(out/f'phase{phase}_optimizer_rank{rank}.json',
             [dict((k,v) for k,v in g.items() if k != 'params') for g in optimizer.param_groups])
        ddp = DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
        save(out/f'phase{phase}_runtime_rank{rank}.json',runtime_manifest())
        save(out/f'phase{phase}_training_start_rank{rank}.json',dict(
             rank_seed=42+rank,sampler_seed=42,sampler_epoch=0,sampler_order_sha256=identity(train_order),
             cpu_rng_sha256=tensor_digest(torch.get_rng_state()),
             cuda_rng_sha256=tensor_digest(torch.cuda.get_rng_state()),optimizer_states=len(optimizer.state)))
        history = []
        start = time.monotonic()
        for step,batch in enumerate(train,1):
            batch = cuda_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            policy = schedule(phase,step)
            optimizer.param_groups[-1]['lr'] = policy['core_lr']
            sample = active = advantage = None
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.monotonic()
            if policy['scst_weight']:
                with torch.no_grad():
                    memory,padding = model.memory(batch)
                    sample,active = rollout(model,memory,padding,tokenizer,sample=True)
                    greedy,_ = rollout(model,memory,padding,tokenizer,sample=False)
                ids = batch['image_id'].cpu().tolist()
                sampled,greedy_text = decode(sample,tokenizer),decode(greedy,tokenizer)
                sr = combine_rewards(reward.score(ids,sampled),objects.score(ids,sampled)[0],.5)
                gr = combine_rewards(reward.score(ids,greedy_text),objects.score(ids,greedy_text)[0],.5)
                advantage = torch.from_numpy(sr-gr).cuda()
                del memory,padding
            loss,ce,rl = ddp(batch,sample,active,advantage,policy['scst_weight'],tokenizer.pad_idx)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss.')
            loss.backward()
            core_norm = sum(float(p.grad.detach().square().sum()) for p in core.parameters() if p.grad is not None)**.5
            base_norm = sum(float(p.grad.detach().square().sum()) for p in base.parameters() if p.grad is not None)**.5
            if phase == 1 and any(p.grad is not None for p in base.parameters()):
                raise AssertionError('Frozen baseline received gradients.')
            if not np.isfinite(core_norm) or not core_norm > 0:
                raise FloatingPointError('Missing/nonfinite CSA gradient.')
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            optimizer.step()
            torch.cuda.synchronize()
            row = dict(step=step,phase=phase,loss=float(loss.detach()),ce=float(ce),scst=float(rl),
                       **policy,base_grad_norm=base_norm,core_grad_norm=core_norm,global_grad_norm=norm,
                       clip_coefficient=min(1.,1./(norm+1e-6)),seconds=time.monotonic()-begin,
                       elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
            history.append(row)
            if step == 1 or step % 25 == 0 or step == UPDATES:
                save(out/f'phase{phase}_history_rank{rank}.json',history)
                print(f'[CSA phase{phase} rank{rank}] '+json.dumps(row),flush=True)
        final_base,final_full = model_digest(base),model_digest(model)
        if phase == 1 and final_base != initial_base:
            raise AssertionError('Baseline state changed in phase1.')
        if {n:tensor_digest(b) for n,b in core.named_buffers()} != banks_before:
            raise AssertionError('Frozen banks changed.')
        hashes = [None,None]
        dist.all_gather_object(hashes,final_full)
        if hashes[0] != hashes[1]:
            raise AssertionError('DDP final model states differ.')
        if rank == 0:
            state = dict(format='csa_two_phase_v1',phase=phase,updates=UPDATES,
                         model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                         caption_word2idx=tokenizer.word2idx,contract_sha256=cfg['contract_sha256'],
                         initial_base_sha256=initial_base,baseline_unchanged=final_base == initial_base,
                         banks_unchanged=True,cohort_sha256=COHORT_SHA,final_full_sha256=final_full)
            name = 'phase1_checkpoint.bin' if phase == 1 else 'checkpoint.bin'
            temporary = out/(name+'.partial')
            torch.save(state,temporary,_use_new_zipfile_serialization=False)
            temporary.replace(out/name)
            del state
        dist.barrier()
        save(out/f'phase{phase}_complete_rank{rank}.json',dict(status='COMPLETE',updates=UPDATES,
             phase=phase,rank=rank,initial_base_sha256=initial_base,initial_full_sha256=phase_initial_full,
             final_base_sha256=final_base,final_full_sha256=final_full,banks_unchanged=True,
             baseline_unchanged=final_base == initial_base,optimizer_started_empty=True,
             trainable_base_parameters=sum(p.numel() for p in base.parameters() if p.requires_grad),
             trainable_core_parameters=sum(p.numel() for p in core.parameters() if p.requires_grad),
             contract_sha256=cfg['contract_sha256'],training_seconds=history[-1]['elapsed_seconds'],test_used=False))
        if phase == 2:
            evaluation_begin = time.monotonic()
            for condition in ('on','off'):
                save(out/f'{condition}_rank{rank}.json',predict(model,validation,tokenizer,condition))
            save(out/f'final_generation_timing_rank{rank}.json',dict(seconds=time.monotonic()-evaluation_begin))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def evaluate(config):
    import subprocess
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    cfg = json.loads(Path(config).read_text())
    begin = time.monotonic()
    out = Path(cfg['output'])
    done = [json.loads((out/f'phase2_complete_rank{r}.json').read_text()) for r in (0,1)]
    assert all(d['status'] == 'COMPLETE' and d['updates'] == UPDATES for d in done)
    for key in ('initial_base_sha256','initial_full_sha256','final_base_sha256','final_full_sha256','contract_sha256'):
        assert done[0][key] == done[1][key],key
    dataset = json.loads(Path(cfg['dataset_json']).read_text())
    expected = {r['image_id']:r for r in cfg['groups']['val']}
    ids = sorted(expected)
    ptb = PTBTokenizer()
    refs = ptb.tokenize({i:[{'caption':s['raw']} for s in dataset['images'][i]['sentences'][:5]] for i in ids})
    metrics,predictions = {},{}
    for condition in ('origin','on','off'):
        rows = sum((json.loads((out/f'{condition}_rank{r}.json').read_text()) for r in (0,1)),[])
        rows.sort(key=lambda r:r['image_id'])
        assert len(rows) == 1000 and [r['image_id'] for r in rows] == ids
        predictions[condition] = {r['image_id']:r['caption'] for r in rows}
        save(out/f'{condition}_predictions.json',rows)
        hyp = ptb.tokenize({r['image_id']:[{'caption':r['caption']}] for r in rows})
        score,per_image = Cider().compute_score(refs,hyp)
        save(out/f'{condition}_per_image_cider.json',[dict(image_id=i,CIDEr=float(v)) for i,v in zip(ids,per_image)])
        chair_input,chair_output = out/f'{condition}_chair_input.json',out/f'{condition}_chair_output.json'
        save(chair_input,[dict(image_id=expected[r['image_id']]['coco_id'],caption=r['caption']) for r in rows])
        subprocess.run([sys.executable,'-u',str(Path(cfg['chair'])/'chair.py'),'--cap_file',str(chair_input),
                        '--cache',str(Path(cfg['chair'])/'chair.pkl'),'--save_path',str(chair_output)],cwd=cfg['chair'],check=True)
        chair = json.loads(chair_output.read_text())
        sentences = chair['sentences']
        assert len(sentences) == 1000 and {s['image_id'] for s in sentences} == {r['coco_id'] for r in expected.values()}
        bad = sum(len(s['mscoco_hallucinated_words']) for s in sentences)
        mentions = sum(len(s['mscoco_generated_words']) for s in sentences)
        correct = sum(len(set(s['mscoco_generated_words'])&set(s['mscoco_gt_words'])) for s in sentences)
        total = sum(len(set(s['mscoco_gt_words'])) for s in sentences)
        hallucinated = sum(bool(s['mscoco_hallucinated_words']) for s in sentences)
        reaggregated = dict(CHAIRs=hallucinated/1000,CHAIRi=bad/mentions if mentions else 0.,Recall=correct/total if total else 0.)
        assert all(abs(v-chair['overall_metrics'][k]) < 1e-12 for k,v in reaggregated.items())
        metrics[condition] = dict(CIDEr=float(score),chair=chair['overall_metrics'],
             counts=dict(images=1000,hallucinated_captions=hallucinated,hallucinated_mentions=bad,object_mentions=mentions,
                         recalled_unique_objects=correct,reference_unique_objects=total))
    save(out/'run_summary.json',dict(status='CSA_TWO_PHASE_COMPLETE_WAIT_MATCHED_A_COMPARISON',metrics=metrics,
         sensitivity=dict(off_changed_captions=sum(predictions['off'][i] != predictions['on'][i] for i in ids)),
         initial_base_sha256=done[0]['initial_base_sha256'],final_full_sha256=done[0]['final_full_sha256'],
         checkpoint_sha256=digest(out/'checkpoint.bin'),protocol_sha256=digest(config),
         cohort_sha256=COHORT_SHA,evaluation_reference_tokens_sha256=identity(refs),test_used=False,
         scoring_seconds=time.monotonic()-begin,preparation_seconds=cfg['preparation_seconds'],
         selection='Final phase2 checkpoint only; dev reused; no rerank/best-checkpoint selection.',
         limits=['CSA has1250 extra core-only updates vs A; not matched compute/capacity.',
                 'Off after training is not A. CHAIR does not measure relation correctness.',
                 'Global clipping couples base/core update scales. One seed and reused dev.']))


def archive(config):
    cfg = json.loads(Path(config).read_text())
    out = Path(cfg['output'])
    with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.py','.log'):
                z.write(p,p.relative_to(out))
    print('SEND:',str(out)+'_review.zip','KEEP: checkpoint.bin and phase1_checkpoint.bin',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=('prepare','worker','evaluate','archive'))
    p.add_argument('--config',required=True)
    p.add_argument('--phase',type=int,choices=(1,2))
    a = p.parse_args()
    if a.mode == 'worker':
        if a.phase is None:
            p.error('--phase required for worker')
        worker(a.config,a.phase)
    else:
        globals()[a.mode](a.config)
