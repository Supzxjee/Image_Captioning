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
        checkpoint_conditioned = bool(checkpoint.get('prompt_conditioned_qformer', False))
        model_conditioned = bool(getattr(model.encoder, 'prompt_conditioned_qformer', False))
        if checkpoint_conditioned != model_conditioned:
            raise RuntimeError(f'Checkpoint prompt-conditioned setting is '
                               f'{checkpoint_conditioned}, but model setting is '
                               f'{model_conditioned}.')
    checkpoint_uses_itc = float(checkpoint.get('itc_weight', 0.0)) > 0
    model_uses_itc = bool(getattr(model.encoder, 'use_itc', False))
    if checkpoint_uses_itc != model_uses_itc:
        raise RuntimeError(f'Checkpoint ITC setting is {checkpoint_uses_itc}, but model ITC '
                           f'setting is {model_uses_itc}.')
    checkpoint_uses_objects = bool(checkpoint.get('object_semantic_alignment', False))
    model_uses_objects = bool(getattr(model.encoder, 'object_semantic_alignment', False))
    if checkpoint_uses_objects != model_uses_objects and not (
            warm_start and model_uses_objects and not checkpoint_uses_objects):
        raise RuntimeError(f'Checkpoint object alignment setting is {checkpoint_uses_objects}, '
                           f'but model setting is {model_uses_objects}.')
    checkpoint_uses_cascade = bool(checkpoint.get('cascade_semantic_alignment', False))
    model_uses_cascade = bool(getattr(model.encoder, 'cascade_semantic_alignment', False))
    if checkpoint_uses_cascade != model_uses_cascade and not (
            warm_start and model_uses_cascade and not checkpoint_uses_cascade):
        raise RuntimeError(f'Checkpoint cascade alignment setting is {checkpoint_uses_cascade}, '
                           f'but model setting is {model_uses_cascade}.')
    if checkpoint_uses_cascade and model_uses_cascade and not warm_start:
        checkpoint_selector = checkpoint.get('cascade_selector_mode', 'object_context')
        model_selector = getattr(model.encoder, 'cascade_selector_mode', 'object_context')
        if checkpoint_selector != model_selector:
            raise RuntimeError(f'Checkpoint cascade selector mode is {checkpoint_selector!r}, '
                               f'but model mode is {model_selector!r}.')
    checkpoint_uses_grounded = bool(checkpoint.get('grounded_relation_alignment', False))
    model_uses_grounded = bool(getattr(model.encoder, 'grounded_relation_alignment', False))
    saved_memory = bool(checkpoint.get('decoder_relation_memory', False))
    model_memory = bool(getattr(model.encoder, 'decoder_relation_memory', False))
    if saved_memory != model_memory and not (warm_start and model_memory and not saved_memory):
        raise RuntimeError('Checkpoint decoder relation memory setting differs; explicitly warm-start a new training run.')
    if warm_start and getattr(model.encoder, 'grounded_relation_version', 'v1') == 'verified_v3':
        if checkpoint.get('grounded_relation_version') not in {'appearance_v2', 'verified_v3'}:
            raise RuntimeError('Verified V3 requires the trained appearance V2 verifier checkpoint.')
    if checkpoint_uses_grounded and model_uses_grounded:
        saved_version = checkpoint.get('grounded_relation_version', 'v1')
        model_version = getattr(model.encoder, 'grounded_relation_version', 'v1')
        upgrading_verified = warm_start and saved_version == 'appearance_v2' and model_version == 'verified_v3'
        if saved_version != model_version and not upgrading_verified:
            raise RuntimeError(f'Grounded relation versions differ: checkpoint={saved_version}, '
                               f'model={model_version}. Start V2 from the baseline Q-Former '
                               'checkpoint, not the collapsed E3 V1 checkpoint.')
        if saved_version == model_version == 'verified_v3' and not warm_start:
            if checkpoint.get('relation_verified_scale', 0.1) != model.encoder.relation_verified_scale:
                raise RuntimeError('Verified V3 fusion scale differs from saved checkpoint.')
    if checkpoint_uses_grounded != model_uses_grounded and not (
            warm_start and model_uses_grounded and not checkpoint_uses_grounded):
        raise RuntimeError(f'Checkpoint grounded relation setting is '
                           f'{checkpoint_uses_grounded}, but model setting is '
                           f'{model_uses_grounded}.')
    encoder_state = checkpoint['encoder_state_dict']
    if model.encoder.feature_extractor is None:
        # Older cached-feature checkpoints unnecessarily stored the frozen CLIP
        # vision backbone. It was never used on the cached path.
        encoder_state = {key: value for key, value in encoder_state.items()
                         if not key.startswith('feature_extractor.')}
    if warm_start:
        result = model.encoder.load_state_dict(encoder_state, strict=False)
        allowed = {'prompt_visual_gate.weight', 'prompt_visual_gate.bias'}
        if checkpoint_uses_grounded and model_uses_grounded and upgrading_verified:
            allowed.update(key for key in result.missing_keys
                           if key.startswith('relation_alignment.verified_visual_projection.'))
        # A cached-feature checkpoint may omit the frozen CLIP vision backbone.
        # When continuing from original images, the backbone has just been loaded
        # from the same pretrained CLIP release, so these missing frozen weights
        # are expected and must not invalidate the trainable warm start.
        if model.encoder.feature_extractor is not None:
            allowed.update(key for key in result.missing_keys
                           if key.startswith('feature_extractor.'))
        if model_uses_objects and not checkpoint_uses_objects:
            allowed.update(key for key in result.missing_keys
                           if key.startswith('object_alignment.'))
        if model_uses_cascade and not checkpoint_uses_cascade:
            allowed.update(key for key in result.missing_keys
                           if key.startswith('cascade_alignment.'))
        if model_uses_grounded and not checkpoint_uses_grounded:
            allowed.update(key for key in result.missing_keys
                           if key.startswith('relation_alignment.'))
        if set(result.missing_keys) - allowed or result.unexpected_keys:
            raise RuntimeError(f'Incompatible encoder checkpoint: {result}')
    else:
        model.encoder.load_state_dict(encoder_state)
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    return checkpoint
