# DANmodels.py

from typing import List, Tuple
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from sentiment_data import read_sentiment_examples, WordEmbeddings


class SentimentDatasetDAN(Dataset):
    """
    General idea:
      - sentiment_data returns tokens (strings)
      - DAN needs word indices (ints) to use nn.Embedding
    Each item returns (tensor[word_ids], label) with variable length.
    Padding happens later in collate_fn (per-batch).
    """
    def __init__(self, infile: str, word_indexer):
        self.examples = read_sentiment_examples(infile)
        self.word_indexer = word_indexer
        self.pad_idx = 0
        self.unk_idx = self.word_indexer.index_of("UNK")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        ids = []
        for w in ex.words:
            wi = self.word_indexer.index_of(w)
            if wi == -1:
                wi = self.unk_idx
            ids.append(wi)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(ex.label, dtype=torch.long)


def dan_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    General idea:
      - A batch must be a rectangular tensor (B, T)
      - Sentences have different lengths -> pad within the batch to the max T in this batch
    Returns:
      X_padded: (B, T) long
      lengths: (B,) long  (# real tokens per sentence)
      y: (B,) long
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)

    T = lengths.max().item()
    pad_idx = 0
    X = torch.full((len(xs), T), pad_idx, dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, :len(x)] = x
    y = torch.stack(list(ys))
    return X, lengths, y


class DAN(nn.Module):
    """
    General idea:
      - embed each token -> (B, T, D)
      - average token embeddings per sentence (ignore PAD) -> (B, D)
      - feedforward classifier -> logits (B, 2)
    """
    def __init__(
        self,
        embeddings,
        hidden_size=200,
        num_layers=2,       
        dropout=0.3,
        dropout_mode="hidden",
        freeze_embeddings=False,
        random_init=False,
        random_emb_dim=50       
    ):
        super().__init__()

        # Embedding layer
        if random_init:
            vocab_size = len(embeddings.word_indexer)
            emb_dim = random_emb_dim
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        else:
            self.embedding = embeddings.get_initialized_embedding_layer(frozen=freeze_embeddings)
            emb_dim = embeddings.get_embedding_length()

        self.emb_dim = emb_dim
        self.dropout = dropout
        self.dropout_mode = dropout_mode

        self.emb_dropout = nn.Dropout(dropout)
        self.hidden_dropout = nn.Dropout(dropout)

        # Build MLP dynamically based on num_layers
        # num_layers=1: emb_dim -> 2
        # num_layers=2: emb_dim -> hidden -> 2
        # num_layers=3: emb_dim -> hidden -> hidden -> 2
        layers = []
        if num_layers <= 1:
            layers.append(nn.Linear(emb_dim, 2))
        else:
            layers.append(nn.Linear(emb_dim, hidden_size))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, 2))

        self.ff = nn.Sequential(*layers)

    def forward(self, X, lengths):
        """
        X: LongTensor [B, T] (padded word indices)
        lengths: LongTensor [B] (#real tokens per example)
        """
        # Embeddings: [B, T, D]
        emb = self.embedding(X)

        # Mask out PAD tokens (PAD idx = 0)
        mask = (X != 0).float().unsqueeze(-1)         # [B, T, 1]
        emb = emb * mask

        # Sum then divide by lengths, average: [B, D]
        summed = emb.sum(dim=1)
        avg = summed / lengths.unsqueeze(1).float().clamp(min=1.0)

        # Dropout placement
        if self.dropout_mode in ("embed", "both"):
            avg = self.emb_dropout(avg)

        # Feed-forward
        out = avg
        if self.dropout_mode in ("hidden", "both"):
            # apply dropout between ff layers (manually)
            # easiest: run layer-by-layer and drop after ReLU
            for layer in self.ff:
                out = layer(out)
                if isinstance(layer, nn.ReLU):
                    out = self.hidden_dropout(out)
        else:
            out = self.ff(out)

        return out