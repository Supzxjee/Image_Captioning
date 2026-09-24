"""Single-image beam search, unchanged decoding policy from H1.2."""
import torch
import torch.nn.functional as F
from .config import MAX_CAPTION_LEN

def _decode_tokens(token_ids, tokenizer):
    words = [tokenizer.idx2word.get(token_id, tokenizer.UNK) for token_id in token_ids]
    words = [word for word in words if word not in {tokenizer.PAD, tokenizer.BOS, tokenizer.EOS}]
    return ' '.join(words)


def generate_caption_candidates(model, image, cached_prompt_tokens, prompt_mask, tokenizer,
                                max_len=MAX_CAPTION_LEN, beam_size=5, candidate_count=5,
                                device=None):
    """Return distinct final beams while preserving the original raw-score ordering."""
    if beam_size < 1 or candidate_count < 1 or candidate_count > beam_size:
        raise ValueError('Require 1 <= candidate_count <= beam_size.')
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        cached_prompt_tokens = cached_prompt_tokens.unsqueeze(0).to(device, dtype=torch.float32)
        prompt_mask = prompt_mask.unsqueeze(0).to(device, dtype=torch.long)
        memory, memory_pad_mask = model.build_memory(image, cached_prompt_tokens, prompt_mask)

        beams = [([tokenizer.bos_idx], 0.0)]
        for _ in range(max_len - 1):
            candidates = []
            for token_ids, score in beams:
                if token_ids[-1] == tokenizer.eos_idx:
                    candidates.append((token_ids, score))
                    continue

                target = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
                logits = model.decoder(
                    tgt=target,
                    memory=memory,
                    memory_key_padding_mask=memory_pad_mask,
                )
                log_probs = F.log_softmax(logits[0, -1], dim=-1)
                top_log_probs, top_ids = log_probs.topk(beam_size)
                for log_prob, token_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                    candidates.append((token_ids + [token_id], score + log_prob))

            beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_size]
            if all(token_ids[-1] == tokenizer.eos_idx for token_ids, _ in beams):
                break

    results, seen = [], set()
    for token_ids, score in beams:
        caption = _decode_tokens(token_ids, tokenizer)
        if caption in seen:
            continue
        seen.add(caption)
        generated_length = max(len(token_ids) - 1, 1)
        results.append({
            'caption': caption,
            'logprob': float(score),
            'avg_logprob': float(score / generated_length),
            'length': generated_length,
        })
        if len(results) == candidate_count:
            break
    if not results:
        raise RuntimeError('Beam search produced no caption candidate.')
    return results


def generate_caption_beam_search(model, image, cached_prompt_tokens, prompt_mask, tokenizer,
                                 max_len=MAX_CAPTION_LEN, beam_size=5, device=None):
    return generate_caption_candidates(
        model, image, cached_prompt_tokens, prompt_mask, tokenizer,
        max_len=max_len, beam_size=beam_size, candidate_count=1, device=device,
    )[0]['caption']
