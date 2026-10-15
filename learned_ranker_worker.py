"""Two manual-Kaggle GPU workers: frozen beams or frozen CLIP features."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from captioning.inference_journal import InferenceJournal


def run(a):
    cfg=json.loads(Path(a.config).read_text());out=Path(cfg['output'])
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('Worker requires GPU.')
    rows=cfg['images'][a.rank::cfg['workers']]
    journal=InferenceJournal(out/f'{a.phase}_{a.rank}.jsonl',{'phase':a.phase,'rank':a.rank,'protocol':cfg},[r['image_id'] for r in rows])
    todo=rows[len(journal.records):]
    print(f'{a.phase} shard{a.rank}: {len(journal.records)}/{len(rows)} resumed',flush=True)
    if not todo:return
    if a.phase=='beams':
        from captioning.config import Config
        from captioning.data import load_data,load_visual_input
        from captioning.models import UniversalVisionEncoder,CaptionDecoder,ImageCaptioningModel
        from captioning.checkpoints import load_checkpoint
        from captioning.inference import generate_caption_candidates
        config=Config(mode='candidates',split='val',checkpoint=cfg['checkpoint'],base_path=cfg['base_path'],
            dataset_json_path=cfg['dataset_json_path'],prompt_cache_path=cfg['prompt_cache_path'],
            visual_cache=[cfg['visual_cache']],visual_adapter='qformer',num_visual_queries=32,qformer_layers=2,num_workers=0,device='cuda')
        data=load_data(config)
        model=ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False,visual_adapter='qformer',num_visual_queries=32,qformer_layers=2),CaptionDecoder(data.vocab_size)).cuda()
        load_checkpoint(model,cfg['checkpoint'],data.tokenizer);model.eval()
        lookup={int(r['eval_id']):r for frame in (data.train_df,data.val_df) for _,r in frame.iterrows()}
        try:
            for image in todo:
                row=lookup[image['image_id']]
                assert row['filename']==image['filename'] and int(row['coco_id'])==image['coco_id']
                prompt=data.prompt_cache[row['image']]
                candidates=generate_caption_candidates(model,load_visual_input(row,data.transform,data.visual_cache),prompt['tokens'],prompt['mask'],data.tokenizer,beam_size=5,candidate_count=5,device='cuda')
                journal.append({k:image[k] for k in ('image_id','coco_id','filename')}|{'candidates':candidates})
                if len(journal.records)%10==0:print(f'beams {len(journal.records)}/{len(rows)}',flush=True)
        finally:data.visual_cache.close()
    else:
        from transformers import CLIPModel,AutoTokenizer
        from captioning.visual_cache import VisualCache
        from captioning.multiscore_reranking import rerank_multiscore
        from captioning.object_consistency import index_detections
        from run_hoi_caption_validation import sha
        model=CLIPModel.from_pretrained('openai/clip-vit-base-patch16',revision=cfg['clip_revision']).cuda().eval().requires_grad_(False)
        tokenizer=AutoTokenizer.from_pretrained('openai/clip-vit-base-patch16',revision=cfg['clip_revision'])
        cache=VisualCache([cfg['visual_cache']],id_key='coco_id')
        beams={r['image_id']:r for r in json.loads((out/'beam_candidates.json').read_text())['data']}
        detections=index_detections(json.loads(Path(cfg['detections']).read_text()))
        directory=out/'private_features';directory.mkdir(exist_ok=True)
        try:
            with torch.no_grad():
                for image in todo:
                    record=beams[image['image_id']];candidates=record['candidates'];n=len(candidates)
                    raw=torch.from_numpy(cache.read(image['coco_id'])).cuda()
                    im=F.normalize(model.visual_projection(model.vision_model.post_layernorm(raw[:1])),dim=-1)[0]
                    encoded=tokenizer([c['caption'] for c in candidates],padding='max_length',truncation=True,max_length=77,return_tensors='pt')
                    encoded={k:v.cuda() for k,v in encoded.items()}
                    text=model.text_model(**encoded)
                    pooled=F.normalize(model.text_projection(text.pooler_output),dim=-1)
                    clip=(2.5*(pooled*im).sum(-1).clamp_min(0)).tolist()
                    record={**record,'candidates':[{**c,'clipscore':clip[i]} for i,c in enumerate(candidates)]}
                    _,enriched=rerank_multiscore(record,detections[image['filename']],.4,.3)
                    f=torch.zeros(5,6);tg=torch.zeros(5,512);tt=torch.zeros(5,77,512);tm=torch.zeros(5,77,dtype=torch.bool)
                    for i,c in enumerate(enriched):f[i]=torch.tensor([c['decoder_score'],c['clip_score_normalized'],c['object_score'],max(-2.,c['avg_logprob']/10),c['length']/30,i/4])
                    tg[:n]=pooled.cpu();tt[:n]=text.last_hidden_state.cpu();tm[:n]=encoded['attention_mask'].cpu().bool()
                    fixed=torch.zeros(5,dtype=torch.float64)
                    fixed[:n]=torch.tensor([c['combined_score'] for c in enriched],dtype=torch.float64)
                    payload={'scores':f,'fixed_scores':fixed,'image_global':im.cpu().half(),'text_global':tg.half(),'text_tokens':tt.half(),
                             'text_mask':tm,'patches':raw[1:].cpu().half(),'candidate_mask':torch.arange(5)<n,
                             'image_id':image['image_id'],'coco_id':image['coco_id']}
                    path=directory/f"{image['image_id']}.pt";temporary=path.with_suffix('.partial');torch.save(payload,temporary);temporary.replace(path)
                    journal.append({**record,'feature_file':str(path),'feature_sha256':sha(path)})
                    if len(journal.records)%10==0:print(f'features {len(journal.records)}/{len(rows)}',flush=True)
        finally:cache.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--rank',type=int,required=True)
    p.add_argument('--phase',choices=['beams','features'],required=True);run(p.parse_args())
