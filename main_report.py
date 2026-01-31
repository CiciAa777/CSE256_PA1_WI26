# main.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import random
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from sentiment_data import read_word_embeddings
from DANmodels import SentimentDatasetDAN, dan_collate_fn, DAN
from DAN_bpe import build_bpe_and_indexer, SentimentDatasetBPE, bpe_collate_fn, DAN_BPE

def plot_acc_curve(history, out_path, title):
    """
    history: list of dicts with keys: epoch, train_acc, dev_acc
    Saves a single PNG plot.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    dev_acc = [h["dev_acc"] for h in history]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_acc, label="train acc")
    plt.plot(epochs, dev_acc, label="dev acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

def best_dev(history):
    """
    Returns (best_dev_acc, best_epoch)
    """
    best = max(history, key=lambda h: h["dev_acc"])
    return best["dev_acc"], best["epoch"]

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss

# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def train_epoch_dan(data_loader, model, loss_fn, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, lengths, y in data_loader:
        logits = model(X, lengths)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return correct / total, total_loss / len(data_loader)


def eval_epoch_dan(data_loader, model, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X, lengths, y in data_loader:
            logits = model(X, lengths)
            loss = loss_fn(logits, y)

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    return correct / total, total_loss / len(data_loader)


def experiment_dan(model, train_loader, dev_loader, epochs=20, lr=1e-3, run_name="DAN_run",
                   optim_name="adam", weight_decay=0.0):
    """
    General idea:
      - identical training logic to experiment(), but:
        * loss is CrossEntropyLoss (expects logits)
        * batch includes lengths for masked averaging
    """
    loss_fn = nn.CrossEntropyLoss()
    if optim_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")


    history = []
    for epoch in range(1, epochs + 1):
        tr_acc, tr_loss = train_epoch_dan(train_loader, model, loss_fn, optimizer)
        dv_acc, dv_loss = eval_epoch_dan(dev_loader, model, loss_fn)

        history.append({
            "epoch": epoch,
            "train_acc": tr_acc,
            "dev_acc": dv_acc
        })

        print(f"Epoch {epoch:02d}: train acc {tr_acc:.3f} loss {tr_loss:.3f} | dev acc {dv_acc:.3f} loss {dv_loss:.3f}")

    # Save plot
    plot_acc_curve(history, f"results/{run_name}.png", title=run_name)

    bacc, bepoch = best_dev(history)
    print(f"[{run_name}] BEST dev acc = {bacc:.3f} at epoch {bepoch}")

    return history

def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()
    set_seed(42)

    # Load dataset
    start_time = time.time()

    train_data = SentimentDatasetBOW("data/train.txt")
    dev_data = SentimentDatasetBOW("data/dev.txt", vectorizer=train_data.vectorizer, train=False)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")


    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        
        # ===== Part 1a/1b automated experiments =====

        # Each dict is ONE run.
        # Keep list small & meaningful for the report.
        dan_configs = [
            # 1a: 50d, fine-tune
            dict(name="DAN_glove50_L2_h200_do03_hidden_Adam",
                glove="50", random_init=False, freeze=False,
                num_layers=2, hidden=200, dropout=0.3, dropout_mode="hidden",
                optim="adam", lr=1e-3, wd=0.0, epochs=20),

            # 1a: embeddings 300d 
            dict(name="DAN_glove300_L2_h200_do03_hidden_Adam",
                glove="300", random_init=False, freeze=False,
                num_layers=2, hidden=200, dropout=0.3, dropout_mode="hidden",
                optim="adam", lr=1e-3, wd=0.0, epochs=20),

            # 1a: freeze embeddings 
            dict(name="DAN_glove50_L2_h200_do03_hidden_FROZEN",
                glove="50", random_init=False, freeze=True,
                num_layers=2, hidden=200, dropout=0.3, dropout_mode="hidden",
                optim="adam", lr=1e-3, wd=0.0, epochs=20),

            # 1a: vary depth, num_layers = 1, 3
            dict(name="DAN_glove50_L1_linear_do00",
                glove="50", random_init=False, freeze=False,
                num_layers=1, hidden=200, dropout=0.0, dropout_mode="none",
                optim="adam", lr=1e-3, wd=0.0, epochs=20),

            dict(name="DAN_glove50_L3_h200_do03_hidden",
                glove="50", random_init=False, freeze=False,
                num_layers=3, hidden=200, dropout=0.3, dropout_mode="hidden",
                optim="adam", lr=1e-3, wd=0.0, epochs=20),

            # 1a: dropout placement
            dict(name="DAN_glove50_L2_h200_do03_EMBED",
                glove="50", random_init=False, freeze=False,
                num_layers=2, hidden=200, dropout=0.3, dropout_mode="embed",
                optim="adam", lr=1e-3, wd=0.0, epochs=20),

            dict(name="DAN_glove50_L2_h200_do03_BOTH",
                glove="50", random_init=False, freeze=False,
                num_layers=2, hidden=200, dropout=0.3, dropout_mode="both",
                optim="adam", lr=1e-3, wd=0.0, epochs=20),

            # 1a: optimizer: adamw
            dict(name="DAN_glove50_L2_h200_do03_hidden_AdamW_wd1e-2",
                glove="50", random_init=False, freeze=False,
                num_layers=2, hidden=200, dropout=0.3, dropout_mode="hidden",
                optim="adamw", lr=1e-3, wd=1e-2, epochs=20),
            
            # 1a: optimizer: sgd
            dict(name="DAN_glove50_L2_h200_do03_hidden_SGD_wd1e-2",
                glove="50", random_init=False, freeze=False,
                num_layers=2, hidden=200, dropout=0.3, dropout_mode="hidden",
                optim="sgd", lr=1e-3, wd=1e-2, epochs=20),

            # 1b: random init embeddings
            dict(name="DAN_random50_L2_h200_do03_hidden_Adam",
                glove="50", random_init=True, freeze=False,
                num_layers=2, hidden=200, dropout=0.3, dropout_mode="hidden",
                optim="adam", lr=1e-3, wd=0.0, epochs=20),
            
            dict(name="DAN_random300_L2_h200_do03_hidden_Adam",
                glove="300", random_init=True, freeze=False,
                num_layers=2, hidden=200, dropout=0.3, dropout_mode="hidden",
                optim="adam", lr=1e-3, wd=0.0, epochs=20),
        ]

        # Print summary header
        print("\n=== SUMMARY (copy into report) ===")
        print(f"{'Run':<38} {'BestDev':<8} {'BestEp':<6}")
        print("-" * 56)

        for cfg in dan_configs:
            # Load glove embeddings (only needed for word indexer and glove init case)
            if cfg["glove"] == "50":
                embs = read_word_embeddings("data/glove.6B.50d-relativized.txt")
            else:
                embs = read_word_embeddings("data/glove.6B.300d-relativized.txt")

            train_data = SentimentDatasetDAN("data/train.txt", embs.word_indexer)
            dev_data   = SentimentDatasetDAN("data/dev.txt",   embs.word_indexer)

            train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  collate_fn=dan_collate_fn)
            dev_loader   = DataLoader(dev_data,   batch_size=32, shuffle=False, collate_fn=dan_collate_fn)

            model = DAN(
                embeddings=embs,
                hidden_size=cfg["hidden"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
                dropout_mode=cfg["dropout_mode"],
                freeze_embeddings=cfg["freeze"],
                random_init=cfg["random_init"],
                random_emb_dim=50
            )

            history = experiment_dan(
                model,
                train_loader,
                dev_loader,
                epochs=cfg["epochs"],
                lr=cfg["lr"],
                run_name=cfg["name"],
                optim_name=cfg["optim"],
                weight_decay=cfg["wd"]
            )

            bacc, bepoch = best_dev(history)
            print(f"{cfg['name']:<38} {bacc:<8.3f} {bepoch:<6}")

    elif args.model == "BPE":
        bpe_configs = [
        dict(name="BPE_vocab500_merges500",   vocab=500,  merges=500,  emb_dim=100, hidden=200, dropout=0.3),
        dict(name="BPE_vocab2000_merges500",  vocab=2000, merges=500,  emb_dim=100, hidden=200, dropout=0.3),
        dict(name="BPE_vocab5000_merges500",  vocab=5000, merges=500,  emb_dim=100, hidden=200, dropout=0.3),
        ]


        print("\n=== SUMMARY (copy into report) ===")
        print(f"{'Run':<28} {'BestDev':<8} {'BestEp':<6}")
        print("-" * 46)

        for cfg in bpe_configs:
            bpe, subword_indexer = build_bpe_and_indexer(
                "data/train.txt",
                num_merges=cfg["merges"],
                vocab_size_cap=cfg["vocab"]
            )

            train_data = SentimentDatasetBPE("data/train.txt", bpe, subword_indexer)
            dev_data   = SentimentDatasetBPE("data/dev.txt",   bpe, subword_indexer)

            train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  collate_fn=bpe_collate_fn)
            dev_loader   = DataLoader(dev_data,   batch_size=32, shuffle=False, collate_fn=bpe_collate_fn)

            model = DAN_BPE(
                vocab_size=len(subword_indexer),
                emb_dim=cfg["emb_dim"],
                hidden_size=cfg["hidden"],
                dropout=cfg["dropout"]
            )

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            history = []
            for epoch in range(1, 21):
                model.train()
                tr_correct = tr_total = 0
                tr_loss = 0.0
                for X, lengths, y in train_loader:
                    logits = model(X, lengths)
                    loss = loss_fn(logits, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    tr_loss += loss.item()
                    tr_correct += (logits.argmax(1) == y).sum().item()
                    tr_total += y.size(0)

                model.eval()
                dv_correct = dv_total = 0
                dv_loss = 0.0
                with torch.no_grad():
                    for X, lengths, y in dev_loader:
                        logits = model(X, lengths)
                        loss = loss_fn(logits, y)
                        dv_loss += loss.item()
                        dv_correct += (logits.argmax(1) == y).sum().item()
                        dv_total += y.size(0)

                tr_acc = tr_correct / tr_total
                dv_acc = dv_correct / dv_total
                history.append({"epoch": epoch, "train_acc": tr_acc, "dev_acc": dv_acc})

                print(f"[{cfg['name']}] Epoch {epoch:02d}: train {tr_acc:.3f} | dev {dv_acc:.3f}")

            plot_acc_curve(history, f"results/{cfg['name']}.png", title=cfg["name"])
            bacc, bepoch = best_dev(history)
            print(f"{cfg['name']:<28} {bacc:<8.3f} {bepoch:<6}")

if __name__ == "__main__":
    main()
