"""User-run Kaggle workers. Never execute/import this module for local checks."""
import argparse
import importlib.metadata
import time
from pathlib import Path
import numpy as np
from real_bank_data import load_prepared, sha, save, read, batch

REVISION='57c216476eefef5ab752ec549e440a49ae4ae5f3'
MODEL='openai/clip-vit-base-patch16'
MODEL_SHA='ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0'

def cache(args):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import CLIPModel, CLIPTokenizer
    root,p,arrays,images,vocab,bank=load_prepared(args.prepared_root)
    if not torch.cuda.is_available(): raise RuntimeError('Enable GPU for this worker')
    if tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])<(2,6): raise RuntimeError('Torch>=2.6 required')
    files=['config.json','merges.txt','vocab.json','tokenizer_config.json','special_tokens_map.json','tokenizer.json','pytorch_model.bin']
    downloaded={}
    for name in files:
        print('Pinned CLIP download:',name,flush=True)
        path=Path(hf_hub_download(MODEL,name,revision=REVISION,cache_dir=args.hf_cache,token=False))
        downloaded[name]=sha(path)
    if downloaded['pytorch_model.bin']!=MODEL_SHA: raise ValueError('Official checkpoint SHA differs')
    tokenizer=CLIPTokenizer.from_pretrained(path.parent,local_files_only=True)
    model,info=CLIPModel.from_pretrained(path.parent,local_files_only=True,use_safetensors=False,
        weights_only=True,trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
    allowed={'text_model.embeddings.position_ids','vision_model.embeddings.position_ids'}
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs') or set(info.get('unexpected_keys',[]))-allowed:
        raise ValueError('CLIP loading keys differ: '+str(info))
    if model.config.projection_dim!=512 or model.config.text_config.max_position_embeddings!=77: raise ValueError('CLIP dimensions differ')
    model.requires_grad_(False).eval().to('cuda:0')
    result={}
    for kind in ('noun','verb'):
        words=[r['word'] for r in bank[kind]]
        prompts=[bank[kind+'_template'].format(lemma=w) for w in words]
        chunks=[]
        for start in range(0,len(words),64):
            tokens=tokenizer(prompts[start:start+64],padding=True,truncation=False,return_tensors='pt')
            if tokens.input_ids.shape[1]>77: raise ValueError('Overlong text; truncation forbidden')
            with torch.inference_mode():
                vectors=model.get_text_features(**{k:v.to('cuda:0') for k,v in tokens.items()})
                norms=vectors.float().norm(dim=-1,keepdim=True)
                if not torch.isfinite(vectors).all() or not (norms>0).all(): raise ValueError('Invalid CLIP vector')
                chunks.append((vectors.float()/norms).cpu().numpy())
            print(kind,start+len(chunks[-1]),'/',len(words),flush=True)
        result[kind+'_vectors']=np.concatenate(chunks)
        result[kind+'_words']=np.asarray(words)
    np.savez_compressed(args.output/'semantic_banks.npz',**result)
    save(args.output/'cache_manifest.json',dict(model=MODEL,revision=REVISION,downloaded_sha256=downloaded,
        loading_info=info,precision='FP32, L2 normalized, eval/frozen',templates={k:bank[k+'_template'] for k in ('noun','verb')},
        source_bank_sha256=p['overlay_sha256']['bank_words_corrected.json'],visual_revision='unknown legacy cache',
        software={k:importlib.metadata.version(k) for k in ('torch','transformers','huggingface-hub','tokenizers','numpy','ftfy')}))
    save(args.output/'cache_receipt.json',dict(status='SEMANTIC_BANK_CACHE_COMPLETE',inputs=p,
        payload_sha256={k:sha(args.output/k) for k in ('semantic_banks.npz','cache_manifest.json')}))
    print('SEMANTIC BANK CACHE COMPLETE. Preserve output even if smoke fails.',flush=True)

def smoke(args):
    import torch
    from model import Config, CSACaptioner, xe_loss, beam_search
    root,p,a,images,vocab,bank=load_prepared(args.prepared_root)
    receipt=read(args.output/'cache_receipt.json')
    if receipt['status']!='SEMANTIC_BANK_CACHE_COMPLETE' or receipt['inputs']!=p: raise ValueError('Cache receipt differs')
    for name,h in receipt['payload_sha256'].items():
        if sha(args.output/name)!=h: raise ValueError('Changed text cache')
    with np.load(args.output/'semantic_banks.npz',allow_pickle=False) as f:
        vectors=[]
        for kind in ('noun','verb'):
            v=f[kind+'_vectors']; words=f[kind+'_words'].tolist()
            if words!=[x['word'] for x in bank[kind]] or v.shape!=(len(words),512) or v.dtype!=np.float32 or not np.isfinite(v).all() or not np.allclose(np.linalg.norm(v,axis=1),1,atol=2e-6):
                raise ValueError('Invalid semantic bank')
            vectors.append(torch.from_numpy(v.copy()))
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model=CSACaptioner(Config(vocab=len(vocab)),*vectors).to('cuda:0')
    optimizer=torch.optim.Adam(model.parameters(),lr=8e-6,betas=(.9,.99)); losses=[]
    before=[model.nouns.clone(),model.verbs.clone()]
    with torch.no_grad():
        for agg,values in ((model.noun_agg,model.nouns),(model.verb_agg,model.verbs)):
            torch.testing.assert_close(agg(values)[0],agg(values.flip(0))[0],rtol=1e-4,atol=2e-5)
    for start in (0,2):
        model.train(); x,mask,seq=batch(a,images,p['train_indices'][start:start+2])
        x,mask,seq=[torch.from_numpy(v).to('cuda:0') for v in (x,mask,seq)]
        optimizer.zero_grad(set_to_none=True); logits=model(x,mask,seq[:,:-1]); loss=xe_loss(logits,seq[:,1:])
        if not torch.isfinite(logits).all() or not torch.isfinite(loss): raise ValueError('Nonfinite forward')
        loss.backward()
        for name,param in model.named_parameters():
            if param.grad is None or not torch.isfinite(param.grad).all(): raise ValueError('Missing/nonfinite gradient: '+name)
        for name,param in model.named_parameters():
            if name in ('project.weight','noun_agg.wa.weight','verb_agg.wa.weight') or 'gate' in name:
                if not param.grad.abs().sum()>0: raise ValueError('Disconnected gradient: '+name)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True); optimizer.step()
        losses.append(dict(step=start//2+1,loss=float(loss.detach()),grad_norm=float(norm)))
        print('Real XE smoke:',losses[-1],flush=True)
    if not torch.equal(before[0],model.nouns) or not torch.equal(before[1],model.verbs) or model.nouns.requires_grad or model.verbs.requires_grad:
        raise ValueError('Semantic buffers changed')
    model.eval(); x,mask,_=batch(a,images,p['val_indices']); x=torch.from_numpy(x).cuda(); mask=torch.from_numpy(mask).cuda()
    with torch.inference_mode(): results=beam_search(model,*model.encode(x,mask),mask,beams=5,maximum=20)
    captions=[]
    for i,(sequences,scores) in zip(p['val_indices'],results):
        ids=sequences[0].cpu().tolist()[1:]; ended=2 in ids; content=ids[:ids.index(2)] if ended else ids
        captions.append(dict(coco_id=images[i]['coco_id'],ids=ids,caption=' '.join(vocab[t] for t in content if t!=0),
            eos=ended,score=float(scores[0]),quality_evidence=False))
    save(args.output/'smoke_summary.json',dict(status='REAL_INPUT_SMOKE_COMPLETE',steps=losses,train_ids=p['train_ids'],
        val_ids=p['val_ids'],captions=captions,bank_buffers_unchanged=True,quality_evaluation=False,checkpoint_saved=False,
        peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['cache','smoke']); parser.add_argument('--prepared-root',required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--hf-cache',default='/tmp/csa-hf-cache')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True); globals()[args.mode](args)
