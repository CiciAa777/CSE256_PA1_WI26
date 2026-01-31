# DAN_bpe.py
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F

from sentiment_data import read_sentiment_examples
from utils import Indexer
from bpe import BPE


def build_bpe_and_indexer(train_file: str, num_merges: int, vocab_size_cap: int = None):
    """
    Train BPE on train.txt and build a subword indexer.
    vocab_size_cap: optional hard cap on #subword types kept (besides PAD/UNK)
    """
    train_exs = read_sentiment_examples(train_file)
    all_words = []
    for ex in train_exs:
        all_words.extend(ex.words)

    bpe = BPE(num_merges=num_merges)
    bpe.train(all_words)

    # Build subword vocabulary by encoding train data and counting pieces
    piece_counts = {}
    for ex in train_exs:
        pieces = bpe.encode_sentence(ex.words)
        for p in pieces:
            piece_counts[p] = piece_counts.get(p, 0) + 1

    # Sort by frequency
    sorted_pieces = sorted(piece_counts.items(), key=lambda x: x[1], reverse=True)
    if vocab_size_cap is not None:
        sorted_pieces = sorted_pieces[:vocab_size_cap]

    indexer = Indexer()
    indexer.add_and_get_index("PAD")
    indexer.add_and_get_index("UNK")

    for p, _ in sorted_pieces:
        indexer.add_and_get_index(p)

    return bpe, indexer


class SentimentDatasetBPE(Dataset):
    """
    Returns variable-length subword-id sequences.
    We'll pad in collate_fn.
    """
    def __init__(self, infile: str, bpe: BPE, indexer: Indexer):
        self.examples = read_sentiment_examples(infile)
        self.bpe = bpe
        self.indexer = indexer
        self.pad_idx = indexer.index_of("PAD")
        self.unk_idx = indexer.index_of("UNK")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        pieces = self.bpe.encode_sentence(ex.words)

        ids = []
        for p in pieces:
            pi = self.indexer.index_of(p)
            if pi == -1:
                pi = self.unk_idx
            ids.append(pi)

        return torch.tensor(ids, dtype=torch.long), torch.tensor(ex.label, dtype=torch.long)


def bpe_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    T = lengths.max().item()
    pad_idx = 0

    X = torch.full((len(xs), T), pad_idx, dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, :len(x)] = x
    y = torch.stack(list(ys))
    return X, lengths, y


class DAN_BPE(nn.Module):
    """
    Same DAN idea, but embeddings are random-init and vocab is subwords.
    Returns logits (use CrossEntropyLoss).
    """
    def __init__(self, vocab_size: int, emb_dim: int = 100, hidden_size: int = 200, dropout: float = 0.3):
        super().__init__()
        self.pad_idx = 0
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=self.pad_idx)
        nn.init.uniform_(self.embedding.weight, a=-0.1, b=0.1)
        with torch.no_grad():
            self.embedding.weight[self.pad_idx].zero_()

        self.ff = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, X, lengths):
        emb = self.embedding(X)  # (B, T, D)
        mask = (X != self.pad_idx).unsqueeze(-1).type_as(emb)
        sum_emb = (emb * mask).sum(dim=1)
        denom = lengths.unsqueeze(1).clamp_min(1).type_as(sum_emb)
        avg_emb = sum_emb / denom
        return self.ff(avg_emb)  # logits
