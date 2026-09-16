"""Single-image beam search, unchanged decoding policy from H1.2."""
import torch
import torch.nn.functional as F
from .config import MAX_CAPTION_LEN

def generate_caption_beam_search(model, image, cached_prompt_tokens, prompt_mask, tokenizer, max_len=MAX_CAPTION_LEN, beam_size=5, device=None):
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

    best_tokens = beams[0][0]
    words = [tokenizer.idx2word.get(token_id, tokenizer.UNK) for token_id in best_tokens]
    words = [word for word in words if word not in {tokenizer.PAD, tokenizer.BOS, tokenizer.EOS}]
    return ' '.join(words)
