"""One independently assigned GPU shard: frozen beam or UPT inference."""
import argparse
import json
import random
import time
from pathlib import Path
import torch
from captioning.inference_journal import InferenceJournal


def run(a):
    cfg = json.loads(Path(a.config).read_text(encoding='utf-8'))
    if cfg.get('strict_reproducibility', False):
        from captioning.scst_runtime import configure_strict_runtime
        configure_strict_runtime(True)
    torch.set_num_threads(2)
    random.seed(42); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg.get('require_gpu',True) and device != 'cuda': raise RuntimeError('GPU worker received no CUDA device.')
    image_rows = cfg['images'][a.rank::cfg['workers']]
    output = Path(cfg['output'])
    journal = InferenceJournal(output/f'{a.phase}_{a.rank}.jsonl',
        {'phase':a.phase,'rank':a.rank,'protocol':cfg}, [r['image_id'] for r in image_rows])
    done = len(journal.records); started = time.monotonic()
    print(f'{a.phase} GPU shard {a.rank}: {done}/{len(image_rows)} resumed on {device}',flush=True)
    if done == len(image_rows): return
    if a.phase == 'beams':
        from captioning.config import Config
        from captioning.data import load_data, load_visual_input
        from captioning.models import UniversalVisionEncoder,CaptionDecoder,ImageCaptioningModel
        from captioning.checkpoints import load_checkpoint
        from captioning.inference import generate_caption_candidates
        config = Config(mode='candidates',split=cfg.get('split','val'),checkpoint=cfg['checkpoint'],
            base_path=cfg['base_path'],dataset_json_path=cfg['dataset_json_path'],
            prompt_cache_path=cfg['prompt_cache_path'],visual_cache=[cfg['visual_cache']],
            visual_adapter='qformer',num_visual_queries=32,qformer_layers=2,num_workers=0,device=device)
        data = load_data(config)
        encoder = UniversalVisionEncoder(load_backbone=False,visual_adapter='qformer',num_visual_queries=32,qformer_layers=2)
        model = ImageCaptioningModel(encoder,CaptionDecoder(data.vocab_size)).to(device)
        load_checkpoint(model,config.checkpoint,data.tokenizer); model.eval()
        lookup = {int(r['eval_id']):r for _,r in getattr(data,cfg.get('split','val')+'_df').iterrows()}
        try:
            for image in image_rows[done:]:
                row = lookup[image['image_id']]
                assert row['filename'] == image['filename'] and int(row['coco_id']) == image['coco_id']
                entry = data.prompt_cache[row['image']]
                candidates = generate_caption_candidates(model,load_visual_input(row,data.transform,data.visual_cache),
                    entry['tokens'],entry['mask'],data.tokenizer,beam_size=5,candidate_count=5,device=device)
                journal.append({k:image[k] for k in ('image_id','coco_id','filename')} | {'candidates':candidates})
                if len(journal.records)%10 == 0: print(f'beams {a.rank}: {len(journal.records)}/{len(image_rows)} | {time.monotonic()-started:.1f}s',flush=True)
        finally: data.visual_cache.close()
    else:
        from PIL import Image
        from captioning.upt_reference import load_reference,prepare_image,infer
        from captioning.hoi_caption_reranking import evidence_inventory,candidate_evidence
        metadata = json.loads((Path(cfg['upt_directory'])/'hicodet/instances_train2015.json').read_text())
        correspondence = [[] for _ in metadata['objects']]
        for _,obj,verb in metadata['correspondence']: correspondence[obj].append(verb)
        model, runtime = load_reference(cfg['upt_directory'],cfg['pocket_directory'],cfg['upt_checkpoint'],correspondence)
        model.to(device)
        print('UPT runtime:',json.dumps(runtime),flush=True)
        candidates = json.loads(Path(cfg['clip_candidates']).read_text(encoding='utf-8'))
        lookup = {r['image_id']:r for r in candidates['data']}
        for image in image_rows[done:]:
            path = Path(cfg['base_path'])/image['filepath']/image['filename']
            with Image.open(path) as pixels: prediction = infer(model,prepare_image(pixels).to(device))
            inventory = evidence_inventory(prediction,metadata['objects'],metadata['verbs'],correspondence)
            record = lookup[image['image_id']]
            enriched = [{**c,'hoi_evidence':candidate_evidence(c['caption'],inventory)} for c in record['candidates']]
            journal.append({**record,'candidates':enriched,
                'hoi_image_status':prediction.get('empty_reason','evaluated'),
                'hoi_pair_count':len(prediction['subject_indices'])})
            if len(journal.records)%10 == 0: print(f'upt {a.rank}: {len(journal.records)}/{len(image_rows)} | {time.monotonic()-started:.1f}s',flush=True)
    print(f'{a.phase} shard {a.rank} COMPLETE: {len(journal.records)}',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--rank',type=int,required=True)
    p.add_argument('--phase',choices=['beams','upt'],required=True)
    run(p.parse_args())
