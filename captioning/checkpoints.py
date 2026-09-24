"""Model-only warm start and strict evaluation checkpoint loading."""
import torch

def load_checkpoint(model, path, tokenizer, warm_start=False):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('caption_word2idx', tokenizer.word2idx) != tokenizer.word2idx:
        raise RuntimeError('Checkpoint caption vocabulary differs from current dataset.')
    checkpoint_adapter = checkpoint.get('visual_adapter', 'direct')
    model_adapter = getattr(model.encoder, 'visual_adapter', 'direct')
    if checkpoint_adapter != model_adapter:
        raise RuntimeError(f'Checkpoint visual adapter is {checkpoint_adapter!r}, but model '
                           f'was created with {model_adapter!r}.')
    if checkpoint_adapter == 'qformer':
        expected = (getattr(model.encoder, 'num_visual_queries', None),
                    getattr(model.encoder, 'qformer_layers', None))
        saved = (checkpoint.get('num_visual_queries'), checkpoint.get('qformer_layers'))
        if saved != expected:
            raise RuntimeError(f'Q-Former checkpoint configuration {saved} differs from model '
                               f'configuration {expected}.')
    encoder_state = checkpoint['encoder_state_dict']
    if model.encoder.feature_extractor is None:
        # Older cached-feature checkpoints unnecessarily stored the frozen CLIP
        # vision backbone. It was never used on the cached path.
        encoder_state = {key: value for key, value in encoder_state.items()
                         if not key.startswith('feature_extractor.')}
    if warm_start:
        result = model.encoder.load_state_dict(encoder_state, strict=False)
        allowed = {'prompt_visual_gate.weight', 'prompt_visual_gate.bias'}
        if set(result.missing_keys) - allowed or result.unexpected_keys:
            raise RuntimeError(f'Incompatible encoder checkpoint: {result}')
    else:
        model.encoder.load_state_dict(encoder_state)
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    return checkpoint
