"""Full captioner with only M3's image-specific text zeroed, in every forward."""
import copy
import hashlib
import torch
from csa_patch_model import CSAPatchCaptioner


def zero_prompt_text(prompts):
    changed = dict(prompts)
    for name in ('object_text', 'relation_text'):
        changed[name] = torch.zeros_like(prompts[name])
    return changed


class ZeroTextCaptioner(CSAPatchCaptioner):
    def streams(self, patches, prompts):
        return super().streams(patches, zero_prompt_text(prompts))


def contract_guard(model, vocab, cfg, rank, device, out):
    """USER-RUN synthetic guard. Copies are modified; original training state is not."""
    from real_bank_data import save
    from full_runtime import tensor_digest
    initial = tensor_digest(model)
    cpu_rng = torch.get_rng_state()
    try:
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(cfg['seed']).get_state())
        reference = CSAPatchCaptioner(vocab, model.core.nouns.detach().cpu(), model.core.verbs.detach().cpu()).eval()
    finally:
        torch.set_rng_state(cpu_rng)
    full_state, zero_state = reference.state_dict(), model.state_dict()
    if set(full_state) != set(zero_state):
        raise ValueError('Zero-text model changed full-model state keys')
    digest = hashlib.sha256()
    for name, tensor in sorted(zero_state.items()):
        value = tensor.detach().cpu()
        if not torch.equal(value, full_state[name]):
            raise ValueError('Full-model initialization differs: '+name)
        digest.update(name.encode('utf-8'))
        digest.update(value.contiguous().numpy().tobytes())
    count = sum(p.numel() for p in model.parameters())
    if vocab != 10000 or count != 53183736 or count != sum(p.numel() for p in reference.parameters()):
        raise ValueError('Expected unchanged 53,183,736 full-model parameters')
    probe = copy.deepcopy(model).to(device).eval()
    reference = reference.to(device)
    with torch.no_grad():
        # Initial zero output projections otherwise make the text-invariance check vacuous.
        for branch in (probe.object_prompt, probe.relation_prompt):
            torch.nn.init.eye_(branch.attention.out_proj.weight)
        reference.load_state_dict(probe.state_dict(), strict=True)
    with torch.inference_mode():
        x = torch.randn(2, 196, 768, device=device)
        objects = torch.zeros(2, 8, 196, device=device)
        objects[:, 0, :98] = 1/98
        objects[:, 1, 98:] = 1/98
        subjects = torch.zeros(2, 3, 196, device=device)
        targets = torch.zeros_like(subjects)
        subjects[:, 0] = objects[:, 0]
        targets[:, 0] = objects[:, 1]
        p = dict(object_text=torch.randn(2, 8, 512, device=device),
                 relation_text=torch.randn(2, 3, 512, device=device),
                 object_weights=objects, subject_weights=subjects, target_weights=targets,
                 object_box=torch.zeros(2, 8, 4, device=device),
                 relation_geometry=torch.randn(2, 3, 8, device=device),
                 object_valid=torch.zeros(2, 8, dtype=torch.bool, device=device),
                 relation_valid=torch.zeros(2, 3, dtype=torch.bool, device=device))
        p['object_valid'][:, :2] = True
        p['relation_valid'][:, 0] = True
        p['object_box'][:, 0] = torch.tensor([0., 0., .5, 1.], device=device)
        p['object_box'][:, 1] = torch.tensor([.5, 0., 1., 1.], device=device)
        zeroed = zero_prompt_text(p)
        if any(torch.count_nonzero(zeroed[name]).item() for name in ('object_text', 'relation_text')):
            raise ValueError('M3 text not fully zero')
        if any(zeroed[key] is not value for key, value in p.items() if key not in ('object_text','relation_text')):
            raise ValueError('Zeroing changed ROI, geometry, valid mask or another field')
        other = dict(p, object_text=torch.randn_like(p['object_text'])*7,
                     relation_text=torch.randn_like(p['relation_text'])*7)
        a, b = probe.encode(x, p), probe.encode(x, other)
        expected = reference.encode(x, zeroed)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(a, expected, rtol=1e-5, atol=1e-5)
        sensitive = float((reference.encode(x, p)-reference.encode(x, other)).abs().mean())
        if sensitive <= 1e-7:
            raise ValueError('Positive control insensitive to text; guard is uninformative')
    if tensor_digest(model) != initial:
        raise ValueError('Synthetic guard changed original model state')
    save(out/f'zero_text_contract_rank{rank}.json', dict(
        passed=True, seed=cfg['seed'], prompt_text_mode='zero', trainable_parameters=count,
        full_trainable_parameters=count, same_state_keys_and_initial_tensors=True,
        initial_state_sha256=digest.hexdigest(), state_tensor_count=len(zero_state),
        M1_M2_banks_unchanged=True, ROI_geometry_masks_unchanged=True,
        input_prompt_dict_not_mutated=True, text_change_invariant=True,
        equals_full_receiving_zero_text=True, positive_control_mean_memory_change=sensitive,
        output_projection_enabled_only_on_copies=True, original_state_unchanged=True,
        synthetic_inputs=True, trained_checkpoint_loaded=False,
        limit='Synthetic mechanism test, not caption quality; full initial constructor reconstructed.'))
