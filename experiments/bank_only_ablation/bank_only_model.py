"""M2-only captioner; no image-specific prompt/ROI adapter."""
import hashlib
import torch
from csa_patch_model import CSAPatchCaptioner


class BankOnlyCaptioner(CSAPatchCaptioner):
    def __init__(self, vocab, nouns, verbs):
        # Preserve the full constructor's RNG draw order for all retained tensors.
        super().__init__(vocab, nouns, verbs)
        del self.object_prompt
        del self.relation_prompt

    def streams(self, patches, prompts):
        if patches.ndim != 3 or patches.shape[1:] != (196, 768):
            raise ValueError('Expected B x 196 x 768 CLIP patches without CLS')
        valid = torch.ones(patches.shape[:2], dtype=torch.bool, device=patches.device)
        return self.core.encode(patches, valid)


class BankOnlyContext:
    """Keep the runner interface without reading image-specific semantics."""
    def __init__(self, stage):
        pass

    def events(self, data, events, device):
        return {}

    def images(self, indices, device):
        return {}


def contract_guard(model, vocab, cfg, rank, device, out):
    """Synthetic neural checks, to be run only by the user on Kaggle."""
    from real_bank_data import save
    from full_runtime import tensor_digest
    initial = tensor_digest(model)
    cpu_rng = torch.get_rng_state()
    try:
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(cfg['seed']).get_state())
        reference = CSAPatchCaptioner(vocab, model.core.nouns.detach().cpu(), model.core.verbs.detach().cpu()).eval()
    finally:
        torch.set_rng_state(cpu_rng)
    actual = model.state_dict()
    expected = {k: v for k, v in reference.state_dict().items()
                if not k.startswith(('object_prompt.', 'relation_prompt.'))}
    if set(actual) != set(expected) or hasattr(model, 'object_prompt') or hasattr(model, 'relation_prompt'):
        raise ValueError('Only M3 must be removed; retained state differs')
    digest = hashlib.sha256()
    for name, tensor in sorted(actual.items()):
        value = tensor.detach().cpu()
        if not torch.equal(value, expected[name]):
            raise ValueError('Retained initialization differs: '+name)
        digest.update(name.encode('utf-8'))
        digest.update(value.contiguous().numpy().tobytes())
    count = sum(p.numel() for p in model.parameters())
    full_count = sum(p.numel() for p in reference.parameters())
    c = model.core.c
    if vocab != 10000 or count != 49239288 or full_count != 53183736:
        raise ValueError('Expected M2-only 49,239,288 vs full 53,183,736 parameters')
    if (c.layers, c.dim, c.heads, c.hidden, c.dropout) != (3, 512, 8, 2048, .2):
        raise ValueError('Decoder recipe differs')
    if model.core.nouns.shape != (5269, 512) or model.core.verbs.shape != (1594, 512):
        raise ValueError('Wrong v130 bank dimensions')
    model.eval()
    reference = reference.to(device)
    with torch.inference_mode():
        x = torch.randn(2, 196, 768, device=device)
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
        memory = model.encode(x, {})
        poisoned = dict(object_text=torch.full((2, 8, 512), float('nan'), device=device),
                        relation_text=torch.full((2, 3, 512), float('nan'), device=device),
                        object_weights=None, object_box=None, relation_geometry=None)
        torch.testing.assert_close(memory, model.encode(x, poisoned), rtol=0, atol=0)
        vo, vr = reference.core.encode(x, valid)
        torch.testing.assert_close(memory, torch.cat((vo, vr), 1), rtol=1e-5, atol=1e-5)
        prefix = torch.tensor([[1, 4, 5, 6], [1, 7, 8, 9]], device=device)
        torch.testing.assert_close(model.decode(memory, prefix), reference.core.decode(prefix, vo, vr, valid),
                                   rtol=1e-5, atol=1e-5)
        # Positive control: alter only the disposable reference, never the training model.
        reference.core.nouns.zero_()
        reference.core.verbs.zero_()
        changed = torch.cat(reference.core.encode(x, valid), 1)
        sensitivity = float((memory-changed).abs().mean())
        if sensitivity <= 1e-7:
            raise ValueError('No measurable bank conditioning in positive control')
    if tensor_digest(model) != initial:
        raise ValueError('Contract guard changed the training model')
    save(out/f'bank_only_contract_rank{rank}.json', dict(
        passed=True, seed=cfg['seed'], trainable_parameters=count, full_parameters=full_count,
        removed_M3_parameters=full_count-count, state_tensor_count=len(actual),
        retained_initial_sha256=digest.hexdigest(), retained_initialization_equal=True,
        M1_M2_retained=True, M3_removed=True, image_specific_inputs_ignored=True,
        memories_equal_original_M2=True, decoder_implementation_equal=True,
        bank_conditioning_mean_memory_difference=sensitivity, original_state_unchanged=True,
        synthetic_inputs=True, trained_checkpoint_loaded=False))
