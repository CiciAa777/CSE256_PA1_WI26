# main.py
# Submission-friendly version:
# - No plotting
# - No experiment sweeps
# - One command per part (DAN / SUBWORDDAN / BOW)

import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
import time
import random

from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from sentiment_data import read_word_embeddings
from DANmodels import SentimentDatasetDAN, dan_collate_fn, DAN
from DAN_bpe import build_bpe_and_indexer, SentimentDatasetBPE, bpe_collate_fn, DAN_BPE


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


# ---------- BOW ----------
def train_epoch_bow(loader, model, loss_fn, optimizer):
    model.train()
    correct = total = 0

    for X, y in loader:
        X = X.float()
        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return correct / total


def eval_epoch_bow(loader, model, loss_fn):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.float()
            logits = model(X)
            loss = loss_fn(logits, y)

            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    return correct / total


# ---------- DAN ----------
def train_epoch_dan(loader, model, loss_fn, optimizer):
    model.train()
    correct = total = 0

    for X, lengths, y in loader:
        logits = model(X, lengths)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return correct / total


def eval_epoch_dan(loader, model, loss_fn):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for X, lengths, y in loader:
            logits = model(X, lengths)
            loss = loss_fn(logits, y)

            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["BOW", "DAN", "SUBWORDDAN"],
        help="Which model to run"
    )
    args = parser.parse_args()

    set_seed(42)

    # ---------------- BOW ----------------
    if args.model == "BOW":
        print("Running BOW models...")

        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW(
            "data/dev.txt",
            vectorizer=train_data.vectorizer,
            train=False
        )

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        dev_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        for name, model in [
            ("NN2", NN2BOW(input_size=512, hidden_size=100)),
            ("NN3", NN3BOW(input_size=512, hidden_size=100)),
        ]:
            print(f"\n{name}:")

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            loss_fn = nn.NLLLoss()

            for epoch in range(1, 101):
                tr_acc = train_epoch_bow(train_loader, model, loss_fn, optimizer)
                dv_acc = eval_epoch_bow(dev_loader, model, loss_fn)

                if epoch % 10 == 0:
                    print(f"Epoch {epoch:03d}: train {tr_acc:.3f} | dev {dv_acc:.3f}")

    # ---------------- DAN (Part 1a + 1b) ----------------
    elif args.model == "DAN":
        print("Running DAN (Part 1a and 1b)...")

        # Use GloVe 50d for indexing
        embs = read_word_embeddings("data/glove.6B.50d-relativized.txt")

        train_data = SentimentDatasetDAN("data/train.txt", embs.word_indexer)
        dev_data = SentimentDatasetDAN("data/dev.txt", embs.word_indexer)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=dan_collate_fn)
        dev_loader = DataLoader(dev_data, batch_size=32, shuffle=False, collate_fn=dan_collate_fn)

        runs = [
            # 1a: pretrained
            ("DAN_glove50", False),
            # 1b: random init
            ("DAN_random50", True),
        ]

        for run_name, random_init in runs:
            print(f"\n{run_name}")

            model = DAN(
                embeddings=embs,
                hidden_size=200,
                num_layers=2,
                dropout=0.3,
                dropout_mode="hidden",
                freeze_embeddings=False,
                random_init=random_init,
                random_emb_dim=50
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss()

            for epoch in range(1, 21):
                tr_acc = train_epoch_dan(train_loader, model, loss_fn, optimizer)
                dv_acc = eval_epoch_dan(dev_loader, model, loss_fn)

                print(f"Epoch {epoch:02d}: train {tr_acc:.3f} | dev {dv_acc:.3f}")

    # ---------------- SUBWORD DAN (Part 2) ----------------
    elif args.model == "SUBWORDDAN":
        print("Running Subword DAN (BPE)...")

        configs = [
            ("BPE_vocab500", 500),
            ("BPE_vocab2000", 2000),
            ("BPE_vocab5000", 5000),
        ]

        for name, vocab_size in configs:
            print(f"\n{name}")

            bpe, subword_indexer = build_bpe_and_indexer(
                "data/train.txt",
                num_merges=500,
                vocab_size_cap=vocab_size
            )

            train_data = SentimentDatasetBPE("data/train.txt", bpe, subword_indexer)
            dev_data = SentimentDatasetBPE("data/dev.txt", bpe, subword_indexer)

            train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=bpe_collate_fn)
            dev_loader = DataLoader(dev_data, batch_size=32, shuffle=False, collate_fn=bpe_collate_fn)

            model = DAN_BPE(
                vocab_size=len(subword_indexer),
                emb_dim=100,
                hidden_size=200,
                dropout=0.3
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss()

            for epoch in range(1, 21):
                tr_acc = train_epoch_dan(train_loader, model, loss_fn, optimizer)
                dv_acc = eval_epoch_dan(dev_loader, model, loss_fn)

                print(f"Epoch {epoch:02d}: train {tr_acc:.3f} | dev {dv_acc:.3f}")


if __name__ == "__main__":
    main()
