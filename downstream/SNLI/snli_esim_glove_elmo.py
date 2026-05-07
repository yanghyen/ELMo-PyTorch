#!/usr/bin/env python3
"""
SNLI ESIM baseline with optional ELMo concatenation at encoder input only.

Architecture (standard ESIM):
GloVe -> BiLSTM encoder -> soft attention -> inference composition BiLSTM
-> pooling (avg + max) -> MLP classifier

ELMo integration requested:
input = [GloVe ; ELMo]
Only used at encoder input (before first BiLSTM).
"""

from __future__ import annotations

import csv
import json
import re
import sys
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
KST = timezone(timedelta(hours=9))
LABEL_TO_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
SNLI_ZIP_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"


_THIS_FILE = Path(__file__).resolve()
_BILM_TF_ROOT = next((p for p in _THIS_FILE.parents if p.name == "bilm-tf"), _THIS_FILE.parents[2])
if str(_BILM_TF_ROOT) not in sys.path:
    sys.path.insert(0, str(_BILM_TF_ROOT))

from downstream.SST_2.sst2_elmo_classifier import (  # noqa: E402
    CharIdEncoder,
    ELMoEmbedding,
    load_pretrained_char_bilm_from_checkpoint,
)


def tokenize_text(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


class Vocab:
    def __init__(self, min_freq: int = 1) -> None:
        self.min_freq = min_freq
        self.token_to_id: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        self.id_to_token: List[str] = ["<pad>", "<unk>"]
        self.freqs: Dict[str, int] = {}

    def add_tokens(self, tokens: Sequence[str]) -> None:
        for tok in tokens:
            self.freqs[tok] = self.freqs.get(tok, 0) + 1

    def build(self) -> None:
        for tok, f in self.freqs.items():
            if f >= self.min_freq and tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.id_to_token)
                self.id_to_token.append(tok)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.token_to_id.get(t, 1) for t in tokens]

    def __len__(self) -> int:
        return len(self.id_to_token)


class SNLIDataset(Dataset):
    def __init__(self, samples: List[Tuple[List[str], List[str], int]], max_len: int = 64) -> None:
        self.samples = samples
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[str], List[str], int]:
        p, h, y = self.samples[idx]
        return p[: self.max_len], h[: self.max_len], y


def read_snli_jsonl(path: Path) -> List[Tuple[List[str], List[str], int]]:
    out: List[Tuple[List[str], List[str], int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            label = row.get("gold_label")
            if label not in LABEL_TO_ID:
                continue
            p_tokens = tokenize_text(row["sentence1"])
            h_tokens = tokenize_text(row["sentence2"])
            out.append((p_tokens, h_tokens, LABEL_TO_ID[label]))
    return out


def build_vocab(train_rows: Sequence[Tuple[List[str], List[str], int]], min_freq: int) -> Vocab:
    vocab = Vocab(min_freq=min_freq)
    for p, h, _ in train_rows:
        vocab.add_tokens(p)
        vocab.add_tokens(h)
    vocab.build()
    return vocab


def load_glove_embeddings(glove_path: Path, vocab: Vocab, embed_dim: int) -> Tensor:
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
            vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
            mat[idx] = vec
            found += 1
    print(f"GloVe loaded: matched {found}/{len(vocab)} tokens from {glove_path}")
    return mat


def masked_softmax(logits: Tensor, mask: Tensor, dim: int) -> Tensor:
    logits = logits.masked_fill(mask == 0, -1e9)
    return torch.softmax(logits, dim=dim)


def sequence_mask(lengths: Tensor, max_len: int) -> Tensor:
    rng = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return (rng < lengths.unsqueeze(1)).float()


def apply_bilstm(bilstm: nn.LSTM, x: Tensor, lengths: Tensor) -> Tensor:
    lengths_cpu = lengths.clamp(min=1).cpu()
    packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
    packed_out, _ = bilstm(packed)
    out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
    return out


class ESIMWithELMoInput(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        glove_dim: int,
        hidden_size: int,
        num_classes: int,
        pad_id: int,
        dropout: float,
        use_elmo: bool,
        bilm: Optional[nn.Module] = None,
        elmo_layers: Optional[int] = None,
        elmo_dim: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.use_elmo = use_elmo
        self.word_embed = nn.Embedding(vocab_size, glove_dim, padding_idx=pad_id)

        if use_elmo:
            if bilm is None or elmo_layers is None:
                raise ValueError("ELMo enabled but bilm/elmo_layers not provided")
            self.elmo = ELMoEmbedding(bilm=bilm, num_layers=elmo_layers)
        else:
            self.elmo = None

        input_dim = glove_dim + (elmo_dim if use_elmo else 0)
        self.input_dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 8, hidden_size),
            nn.ReLU(),
        )
        self.composition = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 8, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def _encode_inputs(
        self,
        token_ids: Tensor,
        char_ids: Optional[Tensor],
        lengths: Tensor,
    ) -> Tensor:
        glove = self.word_embed(token_ids)
        if self.use_elmo:
            if char_ids is None:
                raise ValueError("ELMo mode requires char_ids")
            elmo_vec = self.elmo(char_ids)
            x = torch.cat([glove, elmo_vec], dim=-1)
        else:
            x = glove
        x = self.input_dropout(x)
        return apply_bilstm(self.encoder, x, lengths)

    def forward(
        self,
        p_ids: Tensor,
        p_lens: Tensor,
        h_ids: Tensor,
        h_lens: Tensor,
        p_char_ids: Optional[Tensor] = None,
        h_char_ids: Optional[Tensor] = None,
    ) -> Tensor:
        a = self._encode_inputs(p_ids, p_char_ids, p_lens)
        b = self._encode_inputs(h_ids, h_char_ids, h_lens)

        a_mask = sequence_mask(p_lens, a.size(1))
        b_mask = sequence_mask(h_lens, b.size(1))

        e = torch.matmul(a, b.transpose(1, 2))
        alpha = masked_softmax(e, b_mask.unsqueeze(1), dim=2)
        beta = masked_softmax(e.transpose(1, 2), a_mask.unsqueeze(1), dim=2)
        attended_a = torch.matmul(alpha, b)
        attended_b = torch.matmul(beta, a)

        m_a = torch.cat([a, attended_a, a - attended_a, a * attended_a], dim=-1)
        m_b = torch.cat([b, attended_b, b - attended_b, b * attended_b], dim=-1)
        m_a = self.projection(m_a)
        m_b = self.projection(m_b)

        v_a = apply_bilstm(self.composition, m_a, p_lens)
        v_b = apply_bilstm(self.composition, m_b, h_lens)

        a_mask_u = a_mask.unsqueeze(-1)
        b_mask_u = b_mask.unsqueeze(-1)
        v_a_avg = (v_a * a_mask_u).sum(dim=1) / a_mask_u.sum(dim=1).clamp(min=1e-6)
        v_b_avg = (v_b * b_mask_u).sum(dim=1) / b_mask_u.sum(dim=1).clamp(min=1e-6)
        v_a_max = v_a.masked_fill(a_mask_u == 0, -1e9).max(dim=1).values
        v_b_max = v_b.masked_fill(b_mask_u == 0, -1e9).max(dim=1).values

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=-1)
        return self.classifier(v)


def make_collate_fn(vocab: Vocab, max_len: int, encoder: Optional[CharIdEncoder]):
    def _pad_ids(ids: List[int], pad_len: int) -> List[int]:
        return ids + [0] * (pad_len - len(ids))

    def _collate(
        batch: List[Tuple[List[str], List[str], int]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:

        # 1️⃣ 먼저 truncate된 tokens 만들기
        p_toks = [x[0][:max_len] for x in batch]
        h_toks = [x[1][:max_len] for x in batch]
        y = torch.tensor([x[2] for x in batch], dtype=torch.long)

        # 2️⃣ word ids
        p_ids_list = [vocab.encode(t) for t in p_toks]
        h_ids_list = [vocab.encode(t) for t in h_toks]

        p_lens = torch.tensor([max(1, len(x)) for x in p_ids_list], dtype=torch.long)
        h_lens = torch.tensor([max(1, len(x)) for x in h_ids_list], dtype=torch.long)

        p_max = max(len(x) for x in p_ids_list)
        h_max = max(len(x) for x in h_ids_list)

        p_ids = torch.tensor([_pad_ids(x, p_max) for x in p_ids_list], dtype=torch.long)
        h_ids = torch.tensor([_pad_ids(x, h_max) for x in h_ids_list], dtype=torch.long)

        if encoder is None:
            return p_ids, p_lens, h_ids, h_lens, y, None, None

        # 3️⃣ char ids (동일 tokens 기준)
        # encode_tokens는 BOS 행 + 각 단어 + EOS 행 순서이므로, 단어 i는 인덱스 i+1.
        p_enc = [encoder.encode_tokens(t) for t in p_toks]
        h_enc = [encoder.encode_tokens(t) for t in h_toks]
        p_char = [x[1 : 1 + len(t)] for x, t in zip(p_enc, p_toks)]
        h_char = [x[1 : 1 + len(t)] for x, t in zip(h_enc, h_toks)]

        p_char_len = max(x.size(0) for x in p_char)
        h_char_len = max(x.size(0) for x in h_char)
        cdim = p_char[0].size(1)

        p_char_ids = torch.zeros(len(batch), p_char_len, cdim, dtype=torch.long)
        h_char_ids = torch.zeros(len(batch), h_char_len, cdim, dtype=torch.long)

        for i, x in enumerate(p_char):
            p_char_ids[i, :x.size(0)] = x
        for i, x in enumerate(h_char):
            h_char_ids[i, :x.size(0)] = x

        return p_ids, p_lens, h_ids, h_lens, y, p_char_ids, h_char_ids

    return _collate


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    tot_loss, tot_correct, tot_n = 0.0, 0.0, 0
    for p_ids, p_lens, h_ids, h_lens, y, p_char, h_char in loader:
        p_ids, p_lens = p_ids.to(device), p_lens.to(device)
        h_ids, h_lens = h_ids.to(device), h_lens.to(device)
        y = y.to(device)
        if p_char is not None:
            p_char = p_char.to(device)
            h_char = h_char.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        logits = model(
            p_ids=p_ids,
            p_lens=p_lens,
            h_ids=h_ids,
            h_lens=h_lens,
            p_char_ids=p_char,
            h_char_ids=h_char,
        )
        loss = criterion(logits, y)
        if train_mode:
            loss.backward()
            optimizer.step()

        bs = int(y.size(0))
        tot_loss += float(loss.detach().cpu()) * bs
        tot_correct += float((logits.argmax(dim=-1) == y).float().sum().cpu())
        tot_n += bs
    return tot_loss / max(tot_n, 1), tot_correct / max(tot_n, 1)


def append_metrics_csv(
    path: Path,
    seed: int,
    epoch: int,
    train_loss: float,
    train_acc: float,
    dev_loss: float,
    dev_acc: float,
    use_elmo: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(
                [
                    "timestamp",
                    "seed",
                    "epoch",
                    "use_elmo",
                    "train_loss",
                    "train_acc",
                    "dev_loss",
                    "dev_acc",
                ]
            )
        w.writerow(
            [
                datetime.now(KST).isoformat(),
                seed,
                epoch,
                int(use_elmo),
                train_loss,
                train_acc,
                dev_loss,
                dev_acc,
            ]
        )


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_snli_dataset(data_root: Path) -> Path:
    """
    Ensure SNLI jsonl files exist under data_root/snli_1.0.
    If missing, download and extract snli_1.0.zip under data_root.
    """
    data_root.mkdir(parents=True, exist_ok=True)
    snli_dir = data_root / "snli_1.0"
    train_file = snli_dir / "snli_1.0_train.jsonl"
    dev_file = snli_dir / "snli_1.0_dev.jsonl"
    if train_file.is_file() and dev_file.is_file():
        return snli_dir

    zip_path = data_root / "snli_1.0.zip"
    print(f"SNLI not found at {snli_dir}. Downloading from {SNLI_ZIP_URL}")
    urllib.request.urlretrieve(SNLI_ZIP_URL, zip_path)
    print(f"Downloaded: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)
    print(f"Extracted SNLI archive under: {data_root}")

    if not train_file.is_file() or not dev_file.is_file():
        raise FileNotFoundError(
            f"SNLI extracted but files missing: {train_file} and {dev_file}"
        )
    return snli_dir


def train() -> None:
    data_root = Path(__file__).resolve().parent
    data_dir = ensure_snli_dataset(data_root)
    train_file = data_dir / "snli_1.0_train.jsonl"
    dev_file = data_dir / "snli_1.0_dev.jsonl"
    test_file = data_dir / "snli_1.0_test.jsonl"
    glove_path = (
        Path(__file__).resolve().parents[1]
        / "SQuAD"
        / ".glove_cache"
        / "glove.6B.300d.txt"
    )

    # Config
    use_elmo = False
    max_len = 80
    min_freq = 2
    glove_dim = 300
    hidden_size = 300
    dropout = 0.5
    batch_size = 32
    epochs = 5
    lr = 4e-4
    seeds = [13, 21, 42]
    metrics_path = Path(__file__).resolve().parent / "snli_esim_glove_baseline_metrics.csv"
    bilm_ckpt = _BILM_TF_ROOT / "checkpoints" / "bilm" / "final_model.pt"
    vocab_file = _BILM_TF_ROOT / "bilm" / "data" / "pretrain" / "elmo" / "vocab.txt"

    train_rows = read_snli_jsonl(train_file)
    dev_rows = read_snli_jsonl(dev_file)
    test_rows = read_snli_jsonl(test_file)
    print(f"Loaded SNLI rows: train={len(train_rows)}, dev={len(dev_rows)}, test={len(test_rows)}")

    vocab = build_vocab(train_rows, min_freq=min_freq)
    print(f"Built vocab size: {len(vocab)}")

    train_ds = SNLIDataset(train_rows, max_len=max_len)
    dev_ds = SNLIDataset(dev_rows, max_len=max_len)
    test_ds = SNLIDataset(test_rows, max_len=max_len)

    elmo_encoder: Optional[CharIdEncoder] = None
    bilm = None
    elmo_layers = None
    elmo_dim = 0
    if use_elmo:
        if not bilm_ckpt.is_file():
            raise FileNotFoundError(f"ELMo checkpoint not found: {bilm_ckpt}")
        bilm, elmo_layers, elmo_dim, options = load_pretrained_char_bilm_from_checkpoint(
            bilm_ckpt, map_location="cpu"
        )
        elmo_encoder = CharIdEncoder(
            max_chars_per_token=int(options["char_cnn"]["max_characters_per_token"]),
            n_characters=int(options["char_cnn"]["n_characters"]),
            vocab_file=vocab_file if vocab_file.is_file() else None,
        )
        print(f"ELMo ready: layers={elmo_layers}, dim={elmo_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collate_fn = make_collate_fn(vocab=vocab, max_len=max_len, encoder=elmo_encoder)
    criterion = nn.CrossEntropyLoss()
    glove_weight: Optional[Tensor] = None
    if glove_path.is_file():
        glove_weight = load_glove_embeddings(glove_path, vocab, glove_dim)
    else:
        print(f"[warn] GloVe file not found: {glove_path}. Using random init.")

    final_dev_accs: List[float] = []
    final_test_accs: List[float] = []
    for seed in seeds:
        print(f"\n===== Seed {seed} =====")
        set_random_seed(seed)
        loader_gen = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            generator=loader_gen,
        )
        dev_loader = DataLoader(
            dev_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = ESIMWithELMoInput(
            vocab_size=len(vocab),
            glove_dim=glove_dim,
            hidden_size=hidden_size,
            num_classes=3,
            pad_id=0,
            dropout=dropout,
            use_elmo=use_elmo,
            bilm=bilm,
            elmo_layers=elmo_layers,
            elmo_dim=elmo_dim,
        ).to(device)

        if glove_weight is not None:
            model.word_embed.weight.data.copy_(glove_weight)
            print("Initialized word embedding from GloVe.")

        optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        best_dev_acc = 0.0
        best_ckpt_path = metrics_path.parent / f"snli_esim_glove_elmo_seed{seed}_best.pt"
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer=optim)
            with torch.no_grad():
                dv_loss, dv_acc = run_epoch(model, dev_loader, criterion, device, optimizer=None)
            if dv_acc > best_dev_acc:
                best_dev_acc = dv_acc
                torch.save(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "use_elmo": use_elmo,
                        "best_dev_acc": best_dev_acc,
                        "model_state_dict": model.state_dict(),
                    },
                    best_ckpt_path,
                )
            print(
                f"seed {seed} | epoch {epoch:02d} | train loss={tr_loss:.4f} acc={tr_acc:.4f} "
                f"| dev loss={dv_loss:.4f} acc={dv_acc:.4f} | best dev acc={best_dev_acc:.4f}"
            )
            append_metrics_csv(
                metrics_path,
                seed=seed,
                epoch=epoch,
                train_loss=tr_loss,
                train_acc=tr_acc,
                dev_loss=dv_loss,
                dev_acc=dv_acc,
                use_elmo=use_elmo,
            )
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        with torch.no_grad():
            ts_loss, ts_acc = run_epoch(model, test_loader, criterion, device, optimizer=None)
        print(
            f"seed {seed} | best dev ckpt epoch={ckpt['epoch']} acc={ckpt['best_dev_acc']:.4f} "
            f"| test loss={ts_loss:.4f} acc={ts_acc:.4f}"
        )
        final_dev_accs.append(best_dev_acc)
        final_test_accs.append(ts_acc)

    if final_dev_accs:
        mean_acc = sum(final_dev_accs) / len(final_dev_accs)
        print(
            f"\nBest dev acc per seed {seeds}: mean={mean_acc:.4f}, values={final_dev_accs}"
        )
    if final_test_accs:
        mean_test_acc = sum(final_test_accs) / len(final_test_accs)
        print(f"Test acc from best-dev checkpoints: mean={mean_test_acc:.4f}, values={final_test_accs}")

    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    train()
