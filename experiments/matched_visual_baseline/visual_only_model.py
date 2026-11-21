"""Visual-only control retaining the final model's exact parallel decoder."""
import hashlib
import torch
from csa_patch_model import CSAPatchCaptioner


class VisualOnlyCaptioner(CSAPatchCaptioner):
    def __init__(self, vocab):
        # Keep the original RNG draw order for every retained tensor; no trained weights.
        dummy = torch.zeros(1, 512)
        super().__init__(vocab, dummy, dummy)
        for name in ('noun_agg', 'verb_agg', 'object1', 'object2', 'relation', 'nouns', 'verbs'):
            delattr(self.core, name)
        del self.object_prompt
        del self.relation_prompt

    def streams(self, patches, p):
        if patches.ndim != 3 or patches.shape[1:] != (196, 768):
            raise ValueError('Expected B x 196 x 768 CLIP patches without CLS')
        if not torch.isfinite(patches).all():
            raise ValueError('Nonfinite visual features')
        visual = self.core.project(patches)
        return visual, visual


class VisualOnlyContext:
    """Runner interface only: no semantic data are read, built or sent to the GPU."""
    def __init__(self, stage):
        pass

    def events(self, data, events, device):
        return {}

    def images(self, indices, device):
        return {}


def contract_guard(model, vocab, cfg, rank, device, out):
    """USER-RUN neural checks before discarded timing steps, never run locally."""
    from real_bank_data import save
    cpu_rng = torch.get_rng_state()
    try:
        # Set only the CPU generator, preserving the owning rank's CUDA generator.
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(cfg['seed']).get_state())
        dummy = torch.zeros(1, 512)
        reference = CSAPatchCaptioner(vocab, dummy, dummy).eval()
    finally:
        torch.set_rng_state(cpu_rng)
    expected_prefixes = ('core.project.', 'core.embedding.', 'core.decoder.', 'core.head.')
    actual = model.state_dict()
    expected = {k: v for k, v in reference.state_dict().items() if k.startswith(expected_prefixes)}
    if set(actual) != set(expected):
        raise ValueError('Unexpected baseline state: semantic module retained or decoder tensor missing')
    digest = hashlib.sha256()
    for name, tensor in sorted(actual.items()):
        value = tensor.detach().cpu()
        if not torch.equal(value, expected[name]):
            raise ValueError('Shared initialization differs from full model: ' + name)
        digest.update(name.encode('utf-8'))
        digest.update(value.contiguous().numpy().tobytes())
    c = model.core.c
    if (c.layers, c.dim, c.heads, c.hidden, c.dropout) != (3, 512, 8, 2048, .2):
        raise ValueError('Decoder recipe differs from final model')
    baseline_parameters = sum(p.numel() for p in model.parameters())
    full_parameters = sum(p.numel() for p in reference.parameters())
    model.eval()
    reference = reference.to(device)
    with torch.inference_mode():
        x = torch.randn(2, 196, 768, device=device)
        prefix = torch.tensor([[1, 4, 5, 6], [1, 7, 8, 9]], device=device)
        memory = model.encode(x, {})
        changed = model.encode(x, {'object_text': torch.full((2, 8, 512), float('nan'), device=device)})
        torch.testing.assert_close(memory, changed, rtol=0, atol=0)
        vo, vr = memory.split(196, dim=1)
        torch.testing.assert_close(vo, model.core.project(x), rtol=0, atol=0)
        torch.testing.assert_close(vo, vr, rtol=0, atol=0)
        logits = model.decode(memory, prefix)
        valid = torch.ones(vo.shape[:2], dtype=torch.bool, device=device)
        torch.testing.assert_close(logits, reference.core.decode(prefix, vo, vr, valid), rtol=1e-5, atol=1e-5)
        difference = float((logits - model.decode(model.encode(x.flip(0), {}), prefix)).abs().mean())
        if difference <= 1e-7:
            raise ValueError('Visual conditioning absent in baseline')
    save(out / f'baseline_contract_rank{rank}.json', dict(
        passed=True, seed=cfg['seed'], layers=c.layers, dropout=c.dropout,
        baseline_trainable_parameters=baseline_parameters,
        full_model_trainable_parameters=full_parameters,
        removed_trainable_parameters=full_parameters-baseline_parameters,
        common_tensor_count=len(actual), common_initial_sha256=digest.hexdigest(),
        common_initialization_equal=True, decoder_implementation_equal=True,
        two_streams_equal_projected_visual=True, prompt_ignored=True,
        visual_conditioning_mean_logit_difference=difference,
        dummy_banks_initialization_only=True, trained_checkpoint_loaded=False))

