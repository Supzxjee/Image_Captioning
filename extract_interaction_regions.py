"""Feature-only CLIP shared-union region extraction; one GPU per shard."""
import argparse,json,hashlib,time
from pathlib import Path
import torch
from captioning.relation_candidates import union_crop_box
from captioning.interaction_regions import boxes_in_crop,pool_regions,REGIONS
from evaluate_frozen_residual import newline_hashes


def main():
    p=argparse.ArgumentParser()
    for name in ('images','model','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--protocol',default='audit/interaction_region_protocol.json')
    p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=2)
    args=p.parse_args();torch.set_num_threads(2)
    if not 0<=args.shard<args.shards or not torch.cuda.is_available():raise ValueError('Select two GPU workers.')
    from PIL import Image
    from transformers import CLIPModel,CLIPProcessor
    protocol=json.loads(Path(args.protocol).read_text());root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    model=CLIPModel.from_pretrained(args.model,local_files_only=True).to('cuda').eval().requires_grad_(False)
    processor=CLIPProcessor.from_pretrained(args.model,local_files_only=True)
    if (model.config.vision_config.image_size,model.config.vision_config.patch_size,model.config.vision_config.hidden_size)!=(224,16,768):
        raise ValueError('Unexpected CLIP vision dimensions.')
    rows=[]
    for split in ('train','val','transfer','full100'):
        path=Path('audit')/f'interaction_region_{split}_pairs.json'
        if protocol['splits'][split]['pair_json_sha256'] not in newline_hashes(path):raise ValueError('Pair metadata changed.')
        rows.extend((split,i,row) for i,row in enumerate(json.loads(path.read_text())))
    records=[];started=time.monotonic()
    for index,(split,split_index,row) in enumerate(rows):
        if index%args.shards!=args.shard:continue
        with Image.open(Path(args.images)/row['filepath']/row['filename']) as source:
            image=source.convert('RGB');width,height=image.size
            crop=union_crop_box(row['subject_box'],row['object_box'],width,height,padding=protocol['union_padding'])
            boxes=boxes_in_crop(row['subject_box'],row['object_box'],crop,width,height)
            union=image.crop(crop).resize((224,224),Image.Resampling.BICUBIC)
        with torch.inference_mode():
            inputs=processor(images=union,return_tensors='pt').to('cuda')
            visual=model.vision_model(**inputs)
            if visual.last_hidden_state.shape[1:]!=(197,768):raise ValueError('Unexpected patch grid.')
            patches=model.vision_model.post_layernorm(visual.last_hidden_state[0,1:]).float()
            features,valid,mass=pool_regions(patches,visual.pooler_output[0].float(),boxes)
        records.append({'index':index,'split':split,'split_index':split_index,
                        'regions':features.cpu(),'valid':valid.cpu(),'mass':mass.cpu(),
                        'crop_xyxy_pixels':list(crop),'boxes_in_crop':boxes,'image_size':[width,height]})
        if len(records)%25==0:print(f'Worker{args.shard}: {len(records)} pairs | {time.monotonic()-started:.0f}s',flush=True)
    torch.save(records,root/f'regions_{args.shard}.pt')
    print(f'Worker{args.shard} complete: {len(records)} pairs; regions {REGIONS}',flush=True)


if __name__=='__main__':main()
