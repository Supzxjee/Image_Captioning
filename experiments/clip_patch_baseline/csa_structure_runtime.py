"""Neural structure guards, executed only inside the user's Kaggle worker."""


def neural_guards(model, x, prefix, p):
    import copy
    import torch
    x = x[:2]; prefix = prefix[:2]; p = {k: v[:2] for k, v in p.items()}
    captured = {}
    def first(_, args, output): captured['selected'] = output[1].detach().clone()
    def second(_, args): captured['received'] = args[1].detach().clone()
    handles = [model.core.object1.register_forward_hook(first), model.core.object2.register_forward_pre_hook(second)]
    try:
        with torch.no_grad():
            memory = model.encode(x, p)
            if memory.shape != (2, 392, 512) or not torch.isfinite(memory).all():
                raise ValueError('Wrong/nonfinite dual memory')
            torch.testing.assert_close(captured['selected'], captured['received'], rtol=0, atol=0)
            vo, vr = memory.split(196, dim=1)
            valid = torch.ones((2, 196), dtype=torch.bool, device=x.device)
            baseline = model.decode(memory, prefix)
            torch.testing.assert_close(baseline, model.core.decode(prefix, vo, vr, valid), rtol=0, atol=0)
            effects = {}
            perturbation = torch.linspace(-.2, .2, 512, device=x.device).view(1, 1, 512)
            for role, a, b in [('object', vo + perturbation, vr), ('relation', vo, vr + perturbation)]:
                effect = float((baseline - model.decode(torch.cat((a, b), dim=1), prefix)).abs().max())
                if not effect > 1e-7:
                    raise ValueError('Decoder does not use '+role+' stream')
                effects[role] = effect
            for module, slots in [(model.object_prompt, 8), (model.relation_prompt, 3)]:
                test = copy.deepcopy(module).eval()
                torch.nn.init.eye_(test.attention.out_proj.weight)
                test.attention.out_proj.bias.fill_(.1)
                visual = vo; inputs = torch.randn(2, slots, test.slot[0].in_features, device=x.device)
                support = torch.zeros(2, slots, 196, device=x.device)
                active = torch.zeros(2, slots, dtype=torch.bool, device=x.device)
                support[0, 0, 0] = 1; active[0, 0] = True
                result = test(visual, inputs, support, active)
                if not torch.isfinite(result).all(): raise ValueError('Nonfinite masked prompt')
                torch.testing.assert_close(result[1], visual[1], rtol=0, atol=0)
                torch.testing.assert_close(result[0, 1:], visual[0, 1:], rtol=0, atol=0)
                invalid = inputs.clone(); invalid[:, 1:] += 100
                torch.testing.assert_close(test(visual, invalid, support, active), result, rtol=0, atol=0)
                changed = inputs.clone(); changed[0, 0, :512] += 1
                if torch.allclose(result[0, 0], test(visual, changed, support, active)[0, 0], rtol=0, atol=1e-7):
                    raise ValueError('Prompt text channel inactive under nonzero output probe')
    finally:
        for handle in handles: handle.remove()
    return dict(two_object_layers=True, one_object_guided_relation_layer=True, object_semantic_carry=True,
                separate_memories=True, packaging_decode_parity=True, independent_stream_max_logit_change=effects,
                masked_prompt_locality=True, invalid_slot_invariance=True, nonzero_output_text_sensitivity=True,
                decoder_layers=len(model.core.decoder), dropout=model.core.c.dropout,
                no_strength_gate=not hasattr(model, 'strength_gate'),
                limits='Executed on initialized model and controlled probes; not grounding accuracy or learned prompt benefit.')
