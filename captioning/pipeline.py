"""Explicit pipeline orchestration; no training on import."""
import random
import numpy as np
import torch
from .config import post_train_test_config
from .data import load_data, build_train_loader
from .models import UniversalVisionEncoder, CaptionDecoder, ImageCaptioningModel

def run(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {config.device}', flush=True)
    data = load_data(config)
    encoder = UniversalVisionEncoder().to(config.device)
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
            train_model(model, build_train_loader(config, data), data, config)
            if config.test_after_train:
                from .evaluation import evaluate_model
                evaluate_model(model, data, post_train_test_config(config))
        else:
            from .evaluation import evaluate_model
            evaluate_model(model, data, config)
    finally:
        if data.visual_cache is not None:
            data.visual_cache.close()
