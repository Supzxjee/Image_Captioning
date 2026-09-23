"""Explicit pipeline orchestration; no training on import."""
import random
import numpy as np
import torch
from .config import post_train_test_config
from .data import load_data, build_train_loader
from .models import UniversalVisionEncoder, CaptionDecoder, ImageCaptioningModel

def run(config):
    accelerator = None
    if config.mode == 'train':
        from accelerate import Accelerator, DataLoaderConfiguration
        from accelerate.utils import set_seed
        accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(split_batches=True))
        # Keep the sampler seed identical so every rank slices the same global
        # batch sequence. CUDA dropout is made rank-specific after prepare().
        set_seed(config.seed, device_specific=False)
        config.device = accelerator.device
        config.is_main_process = accelerator.is_main_process
        accelerator.print(f'Using Accelerate: {accelerator.num_processes} process(es) | '
                          f'global batch size: {config.batch_size} | device: {config.device}')
    else:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.is_main_process = True
        print(f'Using device: {config.device}', flush=True)
    data = load_data(config)
    needs_backbone = (config.mode == 'verify-cache' or not config.visual_cache or
                      (config.mode == 'train' and config.test_after_train and
                       config.test_visual_source == 'images'))
    if config.is_main_process:
        print('Initializing encoder: ' + ('loading CLIP vision backbone.' if needs_backbone
              else 'visual cache active; skipping unused CLIP vision backbone.'), flush=True)
    encoder = UniversalVisionEncoder(visual_precision=config.visual_precision,
                                     load_backbone=needs_backbone).to(config.device)
    if config.is_main_process:
        print('Encoder initialized.', flush=True)
    if config.mode == 'verify-cache':
        from .cache_verification import verify_cache
        try:
            verify_cache(encoder, data, config)
        finally:
            data.visual_cache.close()
        return
    decoder = CaptionDecoder(vocab_size=data.vocab_size).to(config.device)
    model = ImageCaptioningModel(encoder, decoder).to(config.device)
    try:
        if config.mode == 'train':
            from .training import train_model
            is_main_process = train_model(model, build_train_loader(config, data), data, config,
                                          accelerator=accelerator)
            if config.test_after_train and is_main_process:
                from .evaluation import evaluate_model
                from copy import copy
                test_data = copy(data)
                if config.test_visual_source == 'images':
                    test_data.visual_cache = None
                    print('Full test uses original images with the selected preprocessing/precision.', flush=True)
                evaluate_model(model, test_data, post_train_test_config(config))
        else:
            from .evaluation import evaluate_model
            evaluate_model(model, data, config)
    finally:
        if data.visual_cache is not None:
            data.visual_cache.close()
