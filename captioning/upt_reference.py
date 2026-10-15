"""Inference compatibility for the pinned official UPT-R50 HICO checkpoint.

Upstream model math/weights are imported unchanged. Only optional package imports,
unneeded ImageNet initialization and unavailable TorchVision NMS are adapted.
"""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image

UPT_REVISION='192743045e4c5f6827299fdc15809f5b97ac1b8e'
POCKET_REVISION='bff287098bb73427fd8d5525b2ac9867fb56629d'
NMS_LIBRARY=None


def cpu_nms(boxes,scores,threshold):
    if not len(boxes):return torch.empty(0,dtype=torch.long,device=boxes.device)
    order=torch.argsort(scores,descending=True,stable=True);keep=[]
    area=(boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(-1)
    while len(order):
        i=order[0];keep.append(i)
        rest=order[1:]
        lo=torch.maximum(boxes[i,:2],boxes[rest,:2]);hi=torch.minimum(boxes[i,2:],boxes[rest,2:])
        intersection=(hi-lo).clamp_min(0).prod(-1)
        iou=intersection/(area[i]+area[rest]-intersection).clamp_min(1e-12)
        order=rest[iou<=threshold]
    return torch.stack(keep)


def prepare_image(image):
    w,h=image.size;size=800
    if max(w,h)/min(w,h)*size>1333:size=int(round(1333*min(w,h)/max(w,h)))
    if (w<=h and w==size) or (h<=w and h==size):nw,nh=w,h
    elif w<h:nw,nh=size,int(size*h/w)
    else:nw,nh=int(size*w/h),size
    image=image.convert('RGB').resize((nw,nh),Image.Resampling.BILINEAR)
    pixels=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
    mean=pixels.new_tensor([.485,.456,.406])[:,None,None]
    std=pixels.new_tensor([.229,.224,.225])[:,None,None]
    return (pixels-mean)/std


def load_reference(upt_directory,pocket_directory,checkpoint,correspondence):
    global NMS_LIBRARY
    # The local torch nightly cannot load the stable TorchVision C++ extension.
    # Register its missing schema so pure-Python ResNet/transforms can import.
    if not hasattr(torch.ops.torchvision,'nms'):
        NMS_LIBRARY=torch.library.Library('torchvision','DEF')
        NMS_LIBRARY.define('nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor')
    import torchvision
    if not torchvision.extension._has_ops():
        torchvision.ops.boxes.nms=cpu_nms
        torchvision.ops.nms=cpu_nms
    pocket_path=Path(pocket_directory)/'pocket/models/transformers.py'
    spec=importlib.util.spec_from_file_location('_upt_pocket_transformers',pocket_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pocket=types.ModuleType('pocket');pocket.models=module;sys.modules['pocket']=pocket
    root=Path(upt_directory)
    sys.path[:0]=[str(root),str(root/'detr')]
    import upt
    import models.backbone as backbone
    backbone.is_main_process=lambda:False  # No unused ImageNet download before strict checkpoint loading.
    args=SimpleNamespace(backbone='resnet50',dilation=False,position_embedding='sine',repr_dim=512,
        hidden_dim=256,enc_layers=6,dec_layers=6,dim_feedforward=2048,dropout=.1,nheads=8,
        num_queries=100,pre_norm=False,aux_loss=True,set_cost_class=1,set_cost_bbox=5,set_cost_giou=2,
        bbox_loss_coef=5,giou_loss_coef=2,eos_coef=.1,alpha=.5,gamma=.2,device='cpu',pretrained='',
        human_idx=0,num_classes=117,box_score_thresh=.2,fg_iou_thresh=.5,min_instances=3,max_instances=15)
    model=upt.build_detector(args,correspondence)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    state=saved['model_state_dict']
    incompatible=model.load_state_dict(state,strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:raise ValueError('Incomplete UPT checkpoint.')
    model.eval().requires_grad_(False)
    return model,{'torchvision':torchvision.__version__,'native_torchvision_ops':torchvision.extension._has_ops(),
        'nms':'native' if torchvision.extension._has_ops() else 'deterministic pure torch IoU NMS; stable input-order tie break',
        'checkpoint_keys':list(saved),'state_tensors':len(state),'parameters':sum(p.numel() for p in model.parameters())}


def infer(model,pixels):
    captured={}
    def hook(module,inputs,outputs):
        logits,prior,h,o,objects,_=outputs
        props=inputs[2][0]
        captured.update({k:props[k].detach().cpu() for k in ('boxes','scores','labels')})
        captured.update(logits=logits.detach().cpu(),prior=prior[0].detach().cpu(),
            subject_indices=h[0].detach().cpu(),object_indices=o[0].detach().cpu())
    handle=model.interaction_head.register_forward_hook(hook)
    try:
        with torch.inference_mode():result=model([pixels])[0]
    finally:handle.remove()
    captured['size']=torch.tensor(pixels.shape[-2:])
    captured['official_output']={k:result[k].detach().cpu() for k in ('boxes','pairing','scores','labels','objects','size')}
    return captured
