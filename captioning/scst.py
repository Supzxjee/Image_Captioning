"""SCST primitives, independent of CLIP and the training data loader.

Reward uses CIDEr-D (clipping, sigma=6, x10), fixed training-image IDF,
and NLTK Treebank normalization. Published metrics still use official PTB.
"""
from collections import Counter, OrderedDict
import math
import numpy as np
import torch
import torch.nn.functional as F


def normalize(text):
    from nltk.tokenize import TreebankWordTokenizer
    punctuation = {"''", "'", '``', '`', '.', '?', '!', ',', ':', '-', '--', '...', ';',
                   '(', ')', '[', ']', '{', '}'}
    return ' '.join(w for w in TreebankWordTokenizer().tokenize(text.lower())
                    if w not in punctuation)


def ngrams(text):
    words = text.split()
    return Counter(tuple(words[i:i+n]) for n in range(1, 5)
                   for i in range(len(words)-n+1))


class TrainingCider:
    """Cached reference vectors; never recompute document frequency per batch."""
    def __init__(self, references, reference_cache_size=None):
        if reference_cache_size is not None and reference_cache_size < 1:
            raise ValueError('Reference cache size must be positive or None.')
        self.reference_cache_size = reference_cache_size
        if len(references) < 2 or any(len(v) != 5 for v in references.values()):
            raise ValueError('Need at least two training images with five references each.')
        self.references = {k: [normalize(s) for s in refs] for k, refs in references.items()}
        self.df = Counter()
        for refs in self.references.values():
            self.df.update(set(g for ref in refs for g in ngrams(ref)))
        self.log_count = math.log(len(references))
        self.vectors = ({k: [self.vector(ref) for ref in refs]
                         for k, refs in self.references.items()} if reference_cache_size is None else OrderedDict())

    def reference_vectors(self, image_id):
        if image_id not in self.references: raise KeyError(image_id)
        if image_id not in self.vectors:
            self.vectors[image_id] = [self.vector(ref) for ref in self.references[image_id]]
        if self.reference_cache_size is not None:
            self.vectors.move_to_end(image_id)
            while len(self.vectors) > self.reference_cache_size: self.vectors.popitem(last=False)
        return self.vectors[image_id]

    def vector(self, text):
        vectors, norms, length = [{} for _ in range(4)], [0.]*4, 0
        for g, count in ngrams(text).items():
            weight = count * (self.log_count - math.log(max(1, self.df.get(g, 0))))
            vectors[len(g)-1][g] = weight
            norms[len(g)-1] += weight**2
            if len(g) == 2:
                length += count
        return vectors, [math.sqrt(n) for n in norms], length

    def score(self, ids, captions):
        if len(ids) != len(captions):
            raise ValueError('Reward IDs and captions differ.')
        scores = []
        for image_id, text in zip(ids, captions):
            hyp, hn, hl = self.vector(normalize(text))
            score = 0.
            reference_vectors = self.reference_vectors(image_id)
            for ref, rn, rl in reference_vectors:
                similarity = 0.
                for n in range(4):
                    if hn[n] and rn[n]:
                        similarity += sum(min(v, ref[n].get(g, 0.))*ref[n].get(g, 0.)
                                          for g, v in hyp[n].items()) / (hn[n]*rn[n])
                score += similarity/4 * math.exp(-((hl-rl)**2)/72.)
            scores.append(10.*score/len(reference_vectors))
        return np.asarray(scores, dtype=np.float32)


@torch.no_grad()
def rollout(model, memory, padding, tokenizer, sample, max_len=30):
    """No dropout: sample and greedy use the same distribution as rescoring.

    Keep PAD/BOS probabilities, matching the existing beam decoder policy.
    The EOS action counts; actions after the first EOS do not count.
    """
    sequence = torch.full((memory.size(0), 1), tokenizer.bos_idx,
                          dtype=torch.long, device=memory.device)
    active = torch.ones(memory.size(0), dtype=torch.bool, device=memory.device)
    actions, masks = [], []
    for _ in range(max_len-1):
        logits = model.decoder(sequence, memory, memory_key_padding_mask=padding)[:, -1].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite rollout logits.')
        token = (torch.multinomial(logits.softmax(-1), 1).squeeze(1) if sample
                 else logits.argmax(-1))
        token = torch.where(active, token, tokenizer.pad_idx)
        actions.append(token)
        masks.append(active.clone())
        sequence = torch.cat([sequence, token[:, None]], dim=1)
        active = active & token.ne(tokenizer.eos_idx)
        if not active.any():
            break
    return sequence, torch.stack(masks, dim=1)


def policy_loss(logits, actions, mask, advantage):
    """Standard sequence-sum REINFORCE; greedy reward has no gradient."""
    if logits.shape[:2] != actions.shape or actions.shape != mask.shape:
        raise ValueError('Policy action/mask shape mismatch.')
    selected = F.log_softmax(logits.float(), -1).gather(-1, actions[..., None]).squeeze(-1)
    return -(selected*mask).sum(1).mul(advantage.detach()).mean()


def decode(sequence, tokenizer):
    captions = []
    for row in sequence.detach().cpu().tolist():
        words = []
        for token in row[1:]:
            if token == tokenizer.eos_idx:
                break
            if token not in (tokenizer.pad_idx, tokenizer.bos_idx):
                words.append(tokenizer.idx2word.get(token, tokenizer.UNK))
        captions.append(' '.join(words))
    return captions
