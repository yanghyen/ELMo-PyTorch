import csv
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
KST = timezone(timedelta(hours=9))


# -----------------------
# Tokenizer
# -----------------------
def tokenize_text(sentence: str) -> List[str]:
    return TOKEN_PATTERN.findall(sentence)


# -----------------------
# Vocab
# -----------------------
class Vocab:
    def __init__(self):
        self.token_to_id = {"<pad>": 0, "<unk>": 1}
        self.id_to_token = ["<pad>", "<unk>"]

    def build(self, sentences: List[str]):
        for sent in sentences:
            for tok in tokenize_text(sent):
                if tok not in self.token_to_id:
                    self.token_to_id[tok] = len(self.id_to_token)
                    self.id_to_token.append(tok)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(t, 1) for t in tokens]

    def __len__(self):
        return len(self.id_to_token)


# -----------------------
# Dataset
# -----------------------
class SST2Dataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], vocab: Vocab, max_len=64):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sent, label = self.samples[idx]
        tokens = tokenize_text(sent)[:self.max_len]
        ids = self.vocab.encode(tokens)

        # padding
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))

        return torch.tensor(ids), torch.tensor(label)


# -----------------------
# Model (Baseline)
# -----------------------
class BaselineClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, x):
        # x: (B, T)
        emb = self.embedding(x)  # (B, T, D)

        mask = (x != 0).float().unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)

        pooled = summed / denom  # mean pooling

        return self.classifier(pooled)


# -----------------------
# Data loader
# -----------------------
def read_sst2(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append((r["sentence"], int(r["label"])))
    return rows


# -----------------------
# Train loop
# -----------------------
def train():
    data_dir = Path(__file__).resolve().parent / "SST-2"
    metrics_path = Path(__file__).resolve().parent / "baseline_eval_metrics.csv"
    train_data = read_sst2(data_dir / "train.tsv")
    dev_data = read_sst2(data_dir / "dev.tsv")

    # vocab build
    vocab = Vocab()
    vocab.build([s for s, _ in train_data])

    train_ds = SST2Dataset(train_data, vocab)
    dev_ds = SST2Dataset(dev_data, vocab)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BaselineClassifier(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    file_exists = metrics_path.exists()
    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "dev_loss",
                    "dev_acc",
                ]
            )

        for epoch in range(10):
            # train
            model.train()
            total_loss, total_correct, total_count = 0.0, 0.0, 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)

                loss.backward()
                optimizer.step()

                batch_size = int(y.size(0))
                total_loss += loss.item() * batch_size
                total_correct += float((logits.argmax(-1) == y).float().sum().item())
                total_count += batch_size

            # eval
            model.eval()
            dev_loss, dev_correct, dev_count = 0.0, 0.0, 0
            with torch.no_grad():
                for x, y in dev_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    batch_size = int(y.size(0))
                    dev_loss += loss.item() * batch_size
                    dev_correct += float((logits.argmax(-1) == y).float().sum().item())
                    dev_count += batch_size

            epoch_num = epoch + 1
            avg_train_loss = total_loss / max(total_count, 1)
            avg_train_acc = total_correct / max(total_count, 1)
            avg_dev_loss = dev_loss / max(dev_count, 1)
            avg_dev_acc = dev_correct / max(dev_count, 1)

            writer.writerow(
                [
                    datetime.now(KST).isoformat(),
                    epoch_num,
                    avg_train_loss,
                    avg_train_acc,
                    avg_dev_loss,
                    avg_dev_acc,
                ]
            )
            f.flush()

            print(
                f"epoch {epoch_num} | "
                f"train loss={avg_train_loss:.4f} "
                f"train acc={avg_train_acc:.4f} "
                f"dev loss={avg_dev_loss:.4f} "
                f"dev acc={avg_dev_acc:.4f}"
            )


if __name__ == "__main__":
    train()