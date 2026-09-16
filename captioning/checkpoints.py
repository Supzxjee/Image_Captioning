"""Model-only warm start and strict evaluation checkpoint loading."""
import torch

def load_checkpoint(model, path, tokenizer, warm_start=False):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('caption_word2idx', tokenizer.word2idx) != tokenizer.word2idx:
        raise RuntimeError('Checkpoint caption vocabulary differs from current dataset.')
    if warm_start:
        result = model.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        allowed = {'prompt_visual_gate.weight', 'prompt_visual_gate.bias'}
        if set(result.missing_keys) - allowed or result.unexpected_keys:
            raise RuntimeError(f'Incompatible encoder checkpoint: {result}')
    else:
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    return checkpoint
