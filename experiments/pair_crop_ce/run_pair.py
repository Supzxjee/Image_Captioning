"""Manual Kaggle cache gate and matched CE training, reusing the checked DDP worker."""
import argparse
import json
import os
from pathlib import Path
import shutil
import numpy as np
import run_phrase as rp
from phrase_data import digest,identity,geometry
from pair_data import compare_features,validate_stage
from crop_geometry import crop_rectangle,patch_centers

REVISION='57c216476eefef5ab752ec549e440a49ae4ae5f3'


def seal(cfg):
    return identity({k:v for k,v in cfg.items() if k!='protocol_sha256'})


def prepare(config):
    import torch
    from captioning.scst_runtime import configure_strict_runtime,model_digest,runtime_manifest
    cfg,m=rp.load(config);out=Path(cfg['output']);shared=Path(cfg['shared'])
    assert cfg['variant'] in ('FULLFRAME','PAIRCROP')
    assert not out.exists(),'Use a fresh OUTPUT; do not overwrite a completed experiment'
    assert digest(cfg['checkpoint'])==rp.BASE_SHA and digest(cfg['dataset_json'])==rp.DATA_SHA
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_name(i)=='Tesla T4' for i in (0,1))
    out.mkdir(parents=True);rp.save(out/'locked_manifest.json',m)
    (out/'source').mkdir()
    for name in cfg['source_sha256']:shutil.copy2(Path(__file__).parent/name,out/'source'/name)
    if cfg['variant']=='FULLFRAME':
        assert not shared.exists(),'Use a fresh shared folder; no overwrite'
        shared.mkdir();create_shared(cfg,m,shared)
    receipt=json.loads((shared/'shared_receipt.json').read_text())
    assert receipt['receipt_sha256']==identity({k:v for k,v in receipt.items() if k!='receipt_sha256'})
    assert receipt['status']=='ALL_FEATURE_COMPATIBILITY_GATES_PASS'
    assert receipt['manifest_sha256']==cfg['manifest_sha256'] and receipt['source_sha256']==cfg['source_sha256']
    assert receipt['checkpoint_sha256']==rp.BASE_SHA and receipt['dataset_sha256']==rp.DATA_SHA
    stage=shared/'feature_cache.bin';assert digest(stage)==receipt['feature_cache_sha256']
    validate_stage(stage,m)
    configure_strict_runtime(True);torch.manual_seed(42)
    _,tok=rp.corpus(cfg);assert identity(tok.word2idx)==receipt['vocabulary_sha256']
    for group in m['groups']:
        for text in group['descriptions']:rp.encode(tok,text,m['max_length'])
    cfg.update(stage=str(stage),stage_sha256=digest(stage),stage_tensor_sha256=receipt['feature_cache_sha256'],
        shared_receipt_sha256=receipt['receipt_sha256'],shared_receipt_file_sha256=digest(shared/'shared_receipt.json'),checkpoint_sha256=rp.BASE_SHA,dataset_sha256=rp.DATA_SHA,
        base_sha256=receipt['base_sha256'],vocabulary_sha256=receipt['vocabulary_sha256'],
        optimizer=dict(name='AdamW',evidence_lr=1e-4,adapter_lr=1e-5,weight_decay=.01,clip_norm=1.),
        loss='positive mean-token CE including EOS; CROSS=0; same expanded scoring compute both arms',
        coco_images=str(shared/'images'),runtime=runtime_manifest())
    if cfg['variant']=='PAIRCROP':
        reference=json.loads((shared/'fullframe_complete_rank0.json').read_text())
        prior=json.loads((shared/'fullframe_protocol.json').read_text())
        assert seal(prior)==prior['protocol_sha256']==reference['protocol_sha256']
        assert prior['variant']=='FULLFRAME' and prior['source_sha256']==cfg['source_sha256']
        assert prior['shared_receipt_sha256']==receipt['receipt_sha256']
        assert reference['updates']==300 and reference['original_parameters_unchanged']
        cfg['fullframe_initial_sha256']=reference['initial_sha256']
        assert cfg['runtime']==reference['runtime'],'Match the original runtime before training treatment'
    cfg['protocol_sha256']=seal(cfg);rp.save(config,cfg);rp.save(out/'protocol.json',cfg)
    rp.save(out/'shared_receipt.json',receipt)
    print('PREPARE PASS:',cfg['variant'],'shared hashes valid; no training yet',flush=True)


def create_shared(cfg,m,shared):
    import torch
    import torchvision
    import transformers
    from PIL import Image,ImageFile
    from torchvision import transforms
    from transformers import CLIPModel
    from captioning.visual_cache import VisualCache
    from captioning.scst_runtime import configure_strict_runtime,model_digest,runtime_manifest
    from inspect_visual_metadata import inspect
    from audit_cache_metadata import audit
    configure_strict_runtime(True);torch.cuda.set_device(0);torch.manual_seed(42)
    metadata=inspect(cfg['visual_cache']);rp.save(shared/'original_metadata.json',metadata);audit(metadata)
    dataset,tok=rp.corpus(cfg);records={r['cocoid']:r for r in dataset['images']}
    events={e['event_id']:e for group in m['groups'] for e in group['events']}
    images={e['image_id']:e for e in events.values()};ids=sorted(images);eids=sorted(events)
    for iid,e in images.items():
        assert records[iid]['filename']==e['filename'] and records[iid]['split'] in ('train','restval')
        rp.image_path(cfg,e)
    ImageFile.LOAD_TRUNCATED_IMAGES=False
    transform=transforms.Compose([transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR,antialias=True),
        transforms.ToTensor(),transforms.Normalize([.48145466,.4578275,.40821073],[.26862954,.26130258,.27577711])])
    clip=CLIPModel.from_pretrained('openai/clip-vit-base-patch16',revision=REVISION,attn_implementation='eager')
    actual_revision=getattr(clip.config,'_commit_hash',None)
    assert actual_revision in (None,REVISION),'Resolved CLIP revision differs from pinned revision'
    vision=clip.vision_model.float().cuda().eval().requires_grad_(False);del clip
    vision_hash=model_digest(vision)
    base=rp.base_model(cfg,tok).cuda().eval();base_hash=model_digest(base)
    old=VisualCache([cfg['visual_cache']],id_key='coco_id');old.require_ids(ids,'locked crop experiment')
    full=[];global_tokens=[];image_receipts=[];compatibility=[];crops=[];coords=[];crop_receipts=[]
    (shared/'images').mkdir()

    def open_image(e):
        image=Image.open(rp.image_path(cfg,e));image.load()
        assert image.getexif().get(274,1)==1 and image.size==(e['width'],e['height']),'EXIF or coordinate size mismatch'
        return image.convert('RGB')

    def encode(image):
        result=vision(pixel_values=transform(image)[None].cuda()).last_hidden_state.float()
        assert result.shape==(1,197,768) and torch.isfinite(result).all()
        assert torch.isfinite(result.half()).all()
        return result

    try:
        with torch.no_grad():
            for k,iid in enumerate(ids,1):
                e=images[iid];fresh=encode(open_image(e));array=fresh[0].cpu().numpy();stored=fresh.half().float()
                previous=old.read(iid)
                check=compare_features(array,previous);check['stored_quantized_check']=compare_features(stored[0].cpu().numpy(),previous)
                compatibility.append(dict(image_id=iid,**check))
                rp.save(shared/'compatibility.json',compatibility)
                assert check['pass_gate'] and check['stored_quantized_check']['pass_gate'],f'FEATURE_COMPATIBILITY_FAILED image {iid}; inspect compatibility.json, do not relax thresholds'
                full.append(stored[0].cpu().numpy().astype(np.float16))
                g=base.encoder.qformer(base.encoder.relu(base.encoder.vis_projection(stored)))
                assert g.shape==(1,32,512) and torch.isfinite(g).all();global_tokens.append(g[0].cpu().numpy())
                path=rp.image_path(cfg,e);image_receipts.append(dict(image_id=iid,filename=e['filename'],sha256=digest(path)))
                shutil.copy2(path,shared/'images'/e['filename'])
                print('Fullframe compatibility:',k,'/',len(ids),'PASS',flush=True)
            for k,event_id in enumerate(eids,1):
                e=events[event_id];rectangle=crop_rectangle(e['subject_box'],e['object_box'],e['width'],e['height'])
                image=open_image(e);view=image.crop(tuple(rectangle));features=encode(view)
                crops.append(features[0,1:].half().cpu().numpy());xy=np.asarray(patch_centers(rectangle,e['width'],e['height']),dtype=np.float32);coords.append(xy)
                s,o=np.array(e['subject_box']),np.array(e['object_box']);union=np.r_[np.minimum(s[:2],o[:2]),np.maximum(s[2:],o[2:])]
                support=[int(((xy>=b[:2])&(xy<=b[2:])).all(-1).sum()) for b in (s,o,union)]
                crop_receipts.append(dict(event_id=event_id,image_id=e['image_id'],rectangle=rectangle,patch_support_SO_union=support))
                print('Crop encoded:',k,'/',len(eids),flush=True)
    finally:old.close()
    assert model_digest(base)==base_hash and model_digest(vision)==vision_hash
    with (shared/'feature_cache.bin').open('xb') as writer:
        np.savez(writer,image_ids=np.asarray(ids),event_ids=np.asarray(eids),full=np.stack(full),global_memory=np.stack(global_tokens),
                 crop=np.stack(crops),crop_xy=np.stack(coords),full_xy=np.asarray(patch_centers([0,0,1,1],1,1),dtype=np.float32))
    receipt=dict(status='ALL_FEATURE_COMPATIBILITY_GATES_PASS',manifest_sha256=cfg['manifest_sha256'],source_sha256=cfg['source_sha256'],
        checkpoint_sha256=rp.BASE_SHA,dataset_sha256=rp.DATA_SHA,vocabulary_sha256=identity(tok.word2idx),base_sha256=base_hash,
        requested_clip_revision=REVISION,resolved_config_revision=actual_revision,vision_weights_sha256=vision_hash,
        feature_cache_sha256=digest(shared/'feature_cache.bin'),image_ids=ids,event_ids=eids,image_receipts=image_receipts,crop_receipts=crop_receipts,
        runtime=runtime_manifest(),torchvision=torchvision.__version__,transformers=transformers.__version__,
        pixel_pipeline='RGB strictJPEG orientation1; bilinear antialias224square CLIPnormalize; no centercrop/letterbox',
        feature_gate='all fullframe rawFP32 and quantizedFP16 vs old: elementwise abs<=.01+.005abs(old), relativeL2<=1e-3',
        no_test=True,weights_unchanged=True)
    receipt['receipt_sha256']=identity(receipt);rp.save(shared/'shared_receipt.json',receipt);rp.save(shared/'locked_manifest.json',m)
    print('SHARED CACHE READY; all gates PASS, original base/CLIP unchanged',flush=True)


def build(cfg,tokenizer):
    from captioning.scst_runtime import model_digest
    from pair_model import PairModel
    base=rp.base_model(cfg,tokenizer)
    model=PairModel(base.decoder,base.encoder.vis_projection).cuda().eval()
    if cfg['variant']=='PAIRCROP':assert model_digest(model)==cfg['fullframe_initial_sha256'],'Treatment initialization differs from control'
    return model


class Batches:
    def __init__(self,cfg,m,tokenizer):
        import torch
        with np.load(cfg['stage'],allow_pickle=False) as z:
            self.index={int(i):k for k,i in enumerate(z['image_ids'])};self.event_index={str(i):k for k,i in enumerate(z['event_ids'])}
            self.global_memory=torch.from_numpy(z['global_memory']).cuda()
            if cfg['variant']=='FULLFRAME':
                p=torch.from_numpy(z['full'][:,1:].astype(np.float32)).cuda();xy=torch.from_numpy(z['full_xy']).cuda()[None].expand(len(p),-1,-1)
            else:p=torch.from_numpy(z['crop'].astype(np.float32)).cuda();xy=torch.from_numpy(z['crop_xy']).cuda()
            self.packed=torch.cat((p,xy),-1)
        self.variant=cfg['variant'];self.groups={g['group_id']:g for g in m['groups']}
        self.mean=np.asarray(m['geometry_fit_mean'],dtype=np.float32);self.std=np.asarray(m['geometry_fit_std'],dtype=np.float32)
        self.encoded={g['group_id']:[rp.encode(tokenizer,t,m['max_length']) for t in g['descriptions']] for g in m['groups']}
    def __call__(self,keys):
        import torch
        events=[e for key in keys for e in self.groups[key]['events']]
        rows=[self.index[e['image_id']] for e in events]
        views=rows if self.variant=='FULLFRAME' else [self.event_index[e['event_id']] for e in events]
        boxes=torch.tensor([[e['subject_box'],e['object_box']] for e in events],dtype=torch.float32,device='cuda')
        geo=torch.from_numpy(np.stack([(geometry(e['subject_box'],e['object_box'])-self.mean)/self.std for e in events])).cuda()
        targets=torch.stack([s for key in keys for _ in range(2) for s in self.encoded[key]]).cuda()
        positive=torch.tensor([self.groups[key]['positive_indices'] for key in keys],device='cuda')
        return self.packed[views],self.global_memory[rows],boxes,geo,targets,positive


def worker(config,smoke=False):
    cfg,m=rp.load(config)
    if not smoke:
        assert seal(cfg)==cfg['protocol_sha256'] and cfg['arm']=='CE' and m['cross_weight']==0.
        assert digest(Path(cfg['shared'])/'shared_receipt.json')==cfg['shared_receipt_file_sha256']
    else:
        import test_phrase_model,test_pair_model
        test_phrase_model.synthetic=test_pair_model.synthetic
    rp.build=build;rp.Batches=Batches
    rp.worker(config,smoke=smoke)


def finish(config):
    import torch
    cfg,m=rp.load(config);out=Path(cfg['output'])
    complete=[json.loads((out/f'complete_rank{r}.json').read_text()) for r in (0,1)]
    for key in ('initial_sha256','final_sha256','frozen_sha256'):assert complete[0][key]==complete[1][key]
    assert all(v['updates']==300 and v['original_parameters_unchanged'] for v in complete)
    phases={}
    for phase in ('before','after'):
        rows=sum([json.loads((out/f'{phase}_rank{r}.json').read_text()) for r in (0,1)],[]);rows.sort(key=lambda x:x['group_id'])
        assert {r['group_id'] for r in rows}=={g['group_id'] for g in m['groups']};phases[phase]=rows;rp.save(out/f'{phase}_records.json',rows)
    packet=torch.load(out/'adaptation.bin',map_location='cpu',weights_only=True)
    packet.update(format='pair_crop_ce_adaptation_v1',variant=cfg['variant'],shared_receipt_sha256=cfg['shared_receipt_sha256'])
    torch.save(packet,out/'adaptation.bin')
    if cfg['variant']=='FULLFRAME':
        shared=Path(cfg['shared'])
        for name in ('complete_rank0.json','protocol.json','before_records.json','after_records.json'):
            shutil.copy2(out/name,shared/('fullframe_'+name))
    rp.save(out/'summary.json',dict(status='MATCHED_PAIR_SOURCE_CE_COMPLETE',variant=cfg['variant'],updates=300,initial_sha256=complete[0]['initial_sha256'],
        final_sha256=complete[0]['final_sha256'],shared_receipt_sha256=cfg['shared_receipt_sha256'],
        metrics={phase:{part:rp.metrics(rows,keys) for part,keys in m['partitions'].items()} for phase,rows in phases.items()},
        no_test=True,human_semantic_verdicts_available=False,
        limits=['8new-to-adaptation groups, not independent whole-system test','Ranking/exactmatch are not primary success measures; blind human phrase review still needed',
                'Crop changes framing/context/density; does not isolate resolution','14fit/7exposeddev; no new SCST/full-caption result']))
    # Keep global/probe metrics out of stdout to avoid changing treatment after looking at control.
    print('COMPLETE:',cfg['variant'],'300 CE updates; keep outputs unopened until both arms finish',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','smoke','worker','finish','archive']);p.add_argument('--config',required=True);a=p.parse_args()
    if a.mode in ('smoke','worker'):worker(a.config,smoke=a.mode=='smoke')
    elif a.mode=='archive':rp.archive(a.config)
    else:globals()[a.mode](a.config)
