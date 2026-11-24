"""User-run checks on learned M2-only inference; never alter model weights."""


def inference_guard(model, patches, memory, rank, out):
    import torch
    from real_bank_data import save
    count = sum(p.numel() for p in model.parameters())
    if count != 49239288 or model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Expected frozen eval M2-only 49,239,288 parameters')
    if hasattr(model, 'object_prompt') or hasattr(model, 'relation_prompt'):
        raise ValueError('M3 adapter present in bank-only inference')
    for bank, shape in ((model.core.nouns, (5269, 512)), (model.core.verbs, (1594, 512))):
        if tuple(bank.shape) != shape or not torch.isfinite(bank).all():
            raise ValueError('Missing/nonfinite trained bank buffer')
        torch.testing.assert_close(bank.float().norm(dim=1), torch.ones(shape[0], device=bank.device),
                                   rtol=0, atol=2e-6)
    valid = torch.ones(patches.shape[:2], dtype=torch.bool, device=patches.device)
    expected = torch.cat(model.core.encode(patches, valid), 1)
    poisoned = dict(object_text=torch.full((len(patches), 8, 512), float('nan'), device=patches.device),
                    relation_text=None, object_weights=None, relation_geometry=None)
    changed = model.encode(patches, poisoned)
    torch.testing.assert_close(memory, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(memory, changed, rtol=1e-5, atol=1e-5)
    save(out/f'bank_only_inference_rank{rank}.json', dict(passed=True, images=len(patches),
        parameters=count, M3_absent=True, M2_memories_verified=True,
        image_specific_inputs_ignored=True, bank_shapes_and_L2_verified=True,
        max_memory_difference=float((memory-changed).abs().max()), training=False))
