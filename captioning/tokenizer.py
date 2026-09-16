from collections import Counter
import torch
from .config import MAX_CAPTION_LEN

class CaptionTokenizer:
    PAD = '<pad>'
    BOS = '<bos>'
    EOS = '<eos>'
    UNK = '<unk>'

    def __init__(self, texts, max_vocab_size=10000):
        self.word2idx = {self.PAD: 0, self.BOS: 1, self.EOS: 2, self.UNK: 3}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())
        for idx, (word, _) in enumerate(counter.most_common(max_vocab_size - 4), start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    @property
    def pad_idx(self):
        return self.word2idx[self.PAD]

    @property
    def bos_idx(self):
        return self.word2idx[self.BOS]

    @property
    def eos_idx(self):
        return self.word2idx[self.EOS]

    def encode(self, text, max_len=MAX_CAPTION_LEN):
        words = text.lower().split()
        ids = [self.bos_idx] + [self.word2idx.get(w, self.word2idx[self.UNK]) for w in words] + [self.eos_idx]
        ids = ids[:max_len]
        padding_mask = [False] * len(ids)
        if len(ids) < max_len:
            pad_count = max_len - len(ids)
            ids.extend([self.pad_idx] * pad_count)
            padding_mask.extend([True] * pad_count)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(padding_mask, dtype=torch.bool)
