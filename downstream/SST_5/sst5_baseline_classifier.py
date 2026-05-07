#!/usr/bin/env python3
"""
SST-5 baseline sentiment classifier (no ELMo).

Pipeline:
    token ids -> nn.Embedding -> masked mean pooling -> MLP -> 5 logits

Expected dataset format (TSV):
    - downstream/SST_5/SST-5/train.tsv
    - downstream/SST_5/SST-5/dev.tsv
    - columns: sentence(or text) + label
    - label range: 0..4
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from downstream.SST_2.sst2_elmo_classifier import tokenize_text


KST = timezone(timedelta(hours=9))
NUM_LABELS = 5
DEFAULT_METRICS_CSV = Path(__file__).resolve().parent / "baseline_eval_metrics.csv"
DEFAULT_CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_GLOVE_PATH = (
    Path(__file__).resolve().parents[1] / "SQuAD" / ".glove_cache" / "glove.6B.300d.txt"
)


def ensure_sst5_dataset(data_dir: Path) -> Path:
    train_tsv = data_dir / "train.tsv"
    dev_tsv = data_dir / "dev.tsv"
    test_tsv = data_dir / "test.tsv"
    if train_tsv.is_file() and dev_tsv.is_file() and test_tsv.is_file():
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"SST-5 not found at {data_dir}. "
        "Downloading from Hugging Face dataset: SetFit/sst5"
    )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Auto-download requires the `datasets` package. "
            "Install it with `pip install datasets`, or place "
            "train.tsv/dev.tsv/test.tsv manually under the SST-5 directory."
        ) from exc

    ds = load_dataset("SetFit/sst5")
    if "train" not in ds:
        raise RuntimeError("SetFit/sst5 download succeeded but missing 'train' split")
    if "validation" not in ds:
        raise RuntimeError("SetFit/sst5 download succeeded but missing 'validation' split")
    if "test" not in ds:
        raise RuntimeError("SetFit/sst5 download succeeded but missing 'test' split")

    def _write_split_tsv(split_name: str, out_path: Path) -> None:
        split = ds[split_name]
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["sentence", "label"])
            for row in split:
                sentence = str(row.get("text", "")).strip()
                if not sentence:
                    continue
                label = int(row["label"])
                if not 0 <= label < NUM_LABELS:
                    raise ValueError(f"Invalid label {label} in split {split_name}")
                writer.writerow([sentence, label])

    _write_split_tsv("train", train_tsv)
    _write_split_tsv("validation", dev_tsv)
    _write_split_tsv("test", test_tsv)
    print(f"Prepared SST-5 at: {data_dir}")
    return data_dir


class Vocab:
    def __init__(self) -> None:
        self.token_to_id = {"<pad>": 0, "<unk>": 1}
        self.id_to_token = ["<pad>", "<unk>"]

    def build(self, sentences: List[str]) -> None:
        for sent in sentences:
            for tok in tokenize_text(sent):
                if tok not in self.token_to_id:
                    self.token_to_id[tok] = len(self.id_to_token)
                    self.id_to_token.append(tok)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(t, 1) for t in tokens]

    def __len__(self) -> int:
        return len(self.id_to_token)


def read_sst5(path: Path) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            sent = (r.get("sentence") or r.get("text") or "").strip()
            if not sent:
                continue
            label_raw = r.get("label")
            if label_raw is None:
                continue
            label = int(label_raw)
            if not 0 <= label < NUM_LABELS:
                raise ValueError(f"Invalid SST-5 label {label} in {path}")
            rows.append((sent, label))
    return rows


class SST5Dataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], vocab: Vocab, max_len: int = 64):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sent, label = self.samples[idx]
        tokens = tokenize_text(sent)[: self.max_len]
        ids = self.vocab.encode(tokens)
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class BaselineClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, NUM_LABELS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)  # (B, T, D)
        mask = (x != 0).float().unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / denom
        return self.classifier(pooled)


def load_glove_embeddings(glove_path: Path, vocab: Vocab, embed_dim: int) -> torch.Tensor:
    mat = torch.randn(len(vocab), embed_dim) * 0.05
    mat[0].zero_()
    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != embed_dim + 1:
                continue
            tok = parts[0]
            idx = vocab.token_to_id.get(tok)
            if idx is None:
                continue
            mat[idx] = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
            found += 1
    print(f"GloVe loaded: matched {found}/{len(vocab)} tokens from {glove_path}")
    return mat


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_best_seed_csv(
    csv_path: Path,
    seed: int,
    best_epoch: int,
    best_dev_loss: float,
    best_dev_acc: float,
    test_acc: float,
    stopped_epoch: int,
    ckpt_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(
                [
                    "seed",
                    "best_epoch",
                    "best_dev_loss",
                    "best_dev_acc",
                    "test_acc",
                    "stopped_epoch",
                    "checkpoint_path",
                ]
            )
        writer.writerow(
            [seed, best_epoch, best_dev_loss, best_dev_acc, test_acc, stopped_epoch, str(ckpt_path)]
        )


def train(
    data_dir: Path,
    metrics_path: Path = DEFAULT_METRICS_CSV,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_len: int = 64,
    seeds: Tuple[int, ...] = (13, 17, 23),
    early_stopping_patience: int = 3,
    checkpoint_dir: Path = DEFAULT_CKPT_DIR,
    glove_path: Path = DEFAULT_GLOVE_PATH,
    glove_dim: int = 300,
) -> None:
    data_dir = ensure_sst5_dataset(data_dir)
    train_tsv = data_dir / "train.tsv"
    dev_tsv = data_dir / "dev.tsv"
    test_tsv = data_dir / "test.tsv"

    train_data = read_sst5(train_tsv)
    dev_data = read_sst5(dev_tsv)
    test_data = read_sst5(test_tsv)
    print(f"Loaded SST-5: train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}")

    vocab = Vocab()
    vocab.build([s for s, _ in train_data])

    train_ds = SST5Dataset(train_data, vocab, max_len=max_len)
    dev_ds = SST5Dataset(dev_data, vocab, max_len=max_len)
    test_ds = SST5Dataset(test_data, vocab, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_by_seed_csv = metrics_path.with_name(f"{metrics_path.stem}_best_by_seed.csv")

    print("--- SST-5 baseline training (multi-seed + early stopping) ---")
    final_test_accs: List[float] = []
    for seed in seeds:
        set_global_seed(seed)
        print(f"\n=== Seed {seed} ===")

        model = BaselineClassifier(vocab_size=len(vocab), embed_dim=glove_dim).to(device)
        glove_weight: Optional[torch.Tensor] = None
        if glove_path.is_file():
            glove_weight = load_glove_embeddings(glove_path, vocab, glove_dim)
        else:
            print(f"[warn] GloVe file not found: {glove_path}. Using random init.")
        if glove_weight is not None:
            model.embedding.weight.data.copy_(glove_weight)
            print("Initialized baseline embedding from GloVe.")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        seed_metrics_path = metrics_path.with_name(f"{metrics_path.stem}_seed{seed}.csv")
        best_ckpt = checkpoint_dir / f"sst5_baseline_best_seed{seed}.pt"
        best_dev_loss = float("inf")
        best_dev_acc = 0.0
        best_epoch = 0
        patience_count = 0

        seed_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        need_header = not seed_metrics_path.exists() or seed_metrics_path.stat().st_size == 0
        with open(seed_metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(
                    ["timestamp", "epoch", "train_loss", "train_acc", "dev_loss", "dev_acc"]
                )

            for epoch in range(1, epochs + 1):
                model.train()
                tr_loss_sum = 0.0
                tr_correct = 0.0
                tr_count = 0

                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

                    bs = int(y.size(0))
                    tr_loss_sum += float(loss.detach().cpu()) * bs
                    tr_correct += float((logits.argmax(dim=-1) == y).float().sum().cpu())
                    tr_count += bs

                model.eval()
                dv_loss_sum = 0.0
                dv_correct = 0.0
                dv_count = 0
                with torch.no_grad():
                    for x, y in dev_loader:
                        x, y = x.to(device), y.to(device)
                        logits = model(x)
                        loss = criterion(logits, y)
                        bs = int(y.size(0))
                        dv_loss_sum += float(loss.detach().cpu()) * bs
                        dv_correct += float((logits.argmax(dim=-1) == y).float().sum().cpu())
                        dv_count += bs

                tr_loss = tr_loss_sum / max(tr_count, 1)
                tr_acc = tr_correct / max(tr_count, 1)
                dv_loss = dv_loss_sum / max(dv_count, 1)
                dv_acc = dv_correct / max(dv_count, 1)

                writer.writerow(
                    [
                        datetime.now(KST).isoformat(),
                        epoch,
                        tr_loss,
                        tr_acc,
                        dv_loss,
                        dv_acc,
                    ]
                )
                f.flush()
                print(
                    f"seed {seed} | epoch {epoch:02d} | train loss={tr_loss:.4f} acc={tr_acc:.4f} "
                    f"| dev loss={dv_loss:.4f} acc={dv_acc:.4f}"
                )

                improved = (dv_loss < best_dev_loss) or (
                    dv_loss == best_dev_loss and dv_acc > best_dev_acc
                )
                if improved:
                    best_dev_loss = dv_loss
                    best_dev_acc = dv_acc
                    best_epoch = epoch
                    patience_count = 0
                    torch.save(
                        {
                            "seed": seed,
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_dev_loss": best_dev_loss,
                            "best_dev_acc": best_dev_acc,
                        },
                        best_ckpt,
                    )
                else:
                    patience_count += 1
                    if patience_count >= early_stopping_patience:
                        print(
                            f"Early stopping at epoch {epoch} (seed={seed}, "
                            f"best_epoch={best_epoch}, best_dev_loss={best_dev_loss:.4f})"
                        )
                        break

        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        ts_correct = 0.0
        ts_count = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                ts_correct += float((logits.argmax(dim=-1) == y).float().sum().cpu())
                ts_count += int(y.size(0))
        test_acc = ts_correct / max(ts_count, 1)
        final_test_accs.append(test_acc)
        print(
            f"seed {seed} | best-dev ckpt epoch={ckpt['epoch']} "
            f"| test acc={test_acc:.4f}"
        )

        append_best_seed_csv(
            csv_path=best_by_seed_csv,
            seed=seed,
            best_epoch=best_epoch,
            best_dev_loss=best_dev_loss,
            best_dev_acc=best_dev_acc,
            test_acc=test_acc,
            stopped_epoch=epoch,
            ckpt_path=best_ckpt,
        )
        print(
            f"Seed {seed} done | best_epoch={best_epoch} "
            f"best_dev_loss={best_dev_loss:.4f} best_dev_acc={best_dev_acc:.4f}"
        )

    if final_test_accs:
        mean_test_acc = sum(final_test_accs) / len(final_test_accs)
        print(
            f"Best-checkpoint test acc over seeds {seeds}: "
            f"mean={mean_test_acc:.4f}, values={final_test_accs}"
        )
    print(f"Per-seed epoch metrics saved to: {metrics_path.parent}")
    print(f"Best-by-seed summary: {best_by_seed_csv}")


if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent / "SST-5"
    METRICS_CSV = DEFAULT_METRICS_CSV
    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 1e-3
    MAX_LEN = 64
    SEEDS = (13, 17, 23, 28, 39)
    EARLY_STOPPING_PATIENCE = 3
    CHECKPOINT_DIR = DEFAULT_CKPT_DIR
    GLOVE_PATH = DEFAULT_GLOVE_PATH
    GLOVE_DIM = 300

    train(
        data_dir=DATA_DIR,
        metrics_path=METRICS_CSV,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        max_len=MAX_LEN,
        seeds=SEEDS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        checkpoint_dir=CHECKPOINT_DIR,
        glove_path=GLOVE_PATH,
        glove_dim=GLOVE_DIM,
    )
