#!/usr/bin/env python3
"""
SQuAD v1.1 baseline:
GloVe word embedding -> BiLSTM -> attention -> span predictor.
"""

from __future__ import annotations

import csv
import random
import re
import string
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
WORD_SPAN_PATTERN = re.compile(r"\S+")
KST = timezone(timedelta(hours=9))
DEFAULT_METRICS_CSV = Path(__file__).resolve().parent / "baseline_eval_metrics.csv"
DEFAULT_GLOVE_CACHE = Path(__file__).resolve().parent / ".glove_cache"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize_text(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


def token_mask_from_ids(token_ids: Tensor, pad_id: int) -> Tensor:
    return (token_ids != pad_id).to(dtype=torch.float32, device=token_ids.device)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def map_answer_char_span_to_token_span(
    context: str,
    answer_start: int,
    answer_text: str,
    max_context_tokens: int,
) -> Optional[Tuple[int, int, List[str]]]:
    spans = [(m.start(), m.end()) for m in WORD_SPAN_PATTERN.finditer(context)]
    tokens = [context[s:e] for s, e in spans]
    if len(tokens) == 0:
        return None

    answer_end = answer_start + len(answer_text)
    start_token_idx = None
    end_token_idx = None
    for i, (s, e) in enumerate(spans):
        if start_token_idx is None and s <= answer_start < e:
            start_token_idx = i
        if s < answer_end <= e:
            end_token_idx = i
            break
        if answer_start <= s and e <= answer_end:
            if start_token_idx is None:
                start_token_idx = i
            end_token_idx = i

    if start_token_idx is None or end_token_idx is None:
        return None
    if start_token_idx >= max_context_tokens or end_token_idx >= max_context_tokens:
        return None
    return start_token_idx, end_token_idx, tokens[:max_context_tokens]


@dataclass
class SquadExample:
    question_tokens: List[str]
    context_tokens: List[str]
    start_idx: int
    end_idx: int
    answers: List[str]


class SquadWordDataset(Dataset):
    def __init__(self, examples: List[SquadExample], max_question_tokens: int, max_context_tokens: int) -> None:
        self.examples = examples
        self.max_question_tokens = max_question_tokens
        self.max_context_tokens = max_context_tokens

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SquadExample:
        ex = self.examples[idx]
        return SquadExample(
            question_tokens=ex.question_tokens[: self.max_question_tokens],
            context_tokens=ex.context_tokens[: self.max_context_tokens],
            start_idx=ex.start_idx,
            end_idx=ex.end_idx,
            answers=ex.answers,
        )


class Vocabulary:
    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.id_to_token: List[str] = [PAD_TOKEN, UNK_TOKEN]

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def add_token(self, token: str) -> None:
        t = token.lower()
        if t not in self.token_to_id:
            self.token_to_id[t] = len(self.id_to_token)
            self.id_to_token.append(t)

    def build_from_examples(self, examples: List[SquadExample]) -> None:
        for ex in examples:
            for t in ex.question_tokens:
                self.add_token(t)
            for t in ex.context_tokens:
                self.add_token(t)

    def encode(self, tokens: List[str]) -> Tensor:
        ids = [self.token_to_id.get(t.lower(), self.unk_id) for t in tokens]
        return torch.tensor(ids, dtype=torch.long)


def make_squad_collate_fn(vocab: Vocabulary):
    def _collate(batch: List[SquadExample]) -> Dict[str, object]:
        q_encoded = [vocab.encode(ex.question_tokens) for ex in batch]
        c_encoded = [vocab.encode(ex.context_tokens) for ex in batch]
        max_q = max(int(x.size(0)) for x in q_encoded)
        max_c = max(int(x.size(0)) for x in c_encoded)
        q_out = torch.full((len(batch), max_q), vocab.pad_id, dtype=torch.long)
        c_out = torch.full((len(batch), max_c), vocab.pad_id, dtype=torch.long)
        start = torch.zeros(len(batch), dtype=torch.long)
        end = torch.zeros(len(batch), dtype=torch.long)
        answers: List[List[str]] = []
        context_tokens: List[List[str]] = []

        for i, ex in enumerate(batch):
            q_out[i, : q_encoded[i].size(0)] = q_encoded[i]
            c_out[i, : c_encoded[i].size(0)] = c_encoded[i]
            start[i] = int(ex.start_idx)
            end[i] = int(ex.end_idx)
            answers.append(ex.answers)
            context_tokens.append(ex.context_tokens)

        return {
            "question_ids": q_out,
            "context_ids": c_out,
            "start_positions": start,
            "end_positions": end,
            "answers": answers,
            "context_tokens": context_tokens,
        }

    return _collate


def ensure_glove_txt(glove_cache_dir: Path, glove_dim: int) -> Path:
    glove_cache_dir.mkdir(parents=True, exist_ok=True)
    txt_path = glove_cache_dir / f"glove.6B.{glove_dim}d.txt"
    if txt_path.is_file():
        return txt_path

    zip_path = glove_cache_dir / "glove.6B.zip"
    if not zip_path.is_file():
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        print(f"Downloading GloVe from {url}")
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        member = f"glove.6B.{glove_dim}d.txt"
        if member not in zf.namelist():
            raise FileNotFoundError(f"{member} not found in {zip_path}")
        zf.extract(member, glove_cache_dir)
    return txt_path


def load_glove_embedding_matrix_from_txt(vocab: Vocabulary, glove_dim: int, glove_cache_dir: Path) -> Tensor:
    txt_path = ensure_glove_txt(glove_cache_dir, glove_dim)
    matrix = torch.empty(len(vocab.id_to_token), glove_dim)
    nn.init.normal_(matrix, mean=0.0, std=0.05)
    matrix[vocab.pad_id].zero_()

    needed = set(vocab.id_to_token) - {PAD_TOKEN, UNK_TOKEN}
    hit = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != glove_dim + 1:
                continue
            token = parts[0]
            if token not in needed:
                continue
            vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
            matrix[vocab.token_to_id[token]] = vec
            hit += 1
            needed.remove(token)
            if not needed:
                break
    coverage = hit / max(1, len(vocab.id_to_token) - 2)
    print(f"GloVe(txt) coverage: {hit}/{max(1, len(vocab.id_to_token)-2)} ({coverage:.2%})")
    return matrix


def load_glove_embedding_matrix(vocab: Vocabulary, glove_dim: int, glove_cache_dir: Path) -> Tensor:
    try:
        from torchtext.vocab import GloVe
    except Exception as exc:
        print(f"[warn] torchtext GloVe load failed: {exc}")
        print("[info] Falling back to raw GloVe txt loader.")
        return load_glove_embedding_matrix_from_txt(vocab, glove_dim, glove_cache_dir)

    glove_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        glove = GloVe(name="6B", dim=glove_dim, cache=str(glove_cache_dir))
        matrix = torch.empty(len(vocab.id_to_token), glove_dim)
        nn.init.normal_(matrix, mean=0.0, std=0.05)
        matrix[vocab.pad_id].zero_()
        hit = 0
        for idx, tok in enumerate(vocab.id_to_token):
            if tok in (PAD_TOKEN, UNK_TOKEN):
                continue
            glove_idx = glove.stoi.get(tok)
            if glove_idx is not None:
                matrix[idx] = glove.vectors[glove_idx]
                hit += 1
        print(f"GloVe(torchtext) coverage: {hit}/{max(1, len(vocab.id_to_token)-2)}")
        return matrix
    except Exception as exc:
        print(f"[warn] torchtext GloVe runtime failed: {exc}")
        print("[info] Falling back to raw GloVe txt loader.")
        return load_glove_embedding_matrix_from_txt(vocab, glove_dim, glove_cache_dir)


class SQuADGloveBaselineQA(nn.Module):
    def __init__(self, embedding_weights: Tensor, hidden_dim: int = 256, dropout: float = 0.2, pad_id: int = 0) -> None:
        super().__init__()
        embed_dim = int(embedding_weights.size(1))
        self.pad_id = pad_id
        self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False, padding_idx=pad_id)
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        enc_dim = hidden_dim * 2
        self.q_proj = nn.Linear(enc_dim, enc_dim, bias=False)
        self.c_proj = nn.Linear(enc_dim, enc_dim, bias=False)
        self.fuse = nn.Sequential(nn.Linear(enc_dim * 4, enc_dim), nn.ReLU(), nn.Dropout(dropout))
        self.modeling = nn.LSTM(
            input_size=enc_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.start_head = nn.Linear(enc_dim * 2, 1)
        self.end_head = nn.Linear(enc_dim * 2, 1)

    def _encode(self, token_ids: Tensor) -> Tensor:
        x = self.embedding(token_ids)
        out, _ = self.encoder(x)
        return out

    def forward(self, question_ids: Tensor, context_ids: Tensor) -> Tuple[Tensor, Tensor]:
        q_h = self._encode(question_ids)
        c_h = self._encode(context_ids)
        q_mask = token_mask_from_ids(question_ids, self.pad_id)
        c_mask = token_mask_from_ids(context_ids, self.pad_id)
        q_proj = self.q_proj(q_h)
        c_proj = self.c_proj(c_h)
        sim = torch.bmm(c_proj, q_proj.transpose(1, 2)) / (float(c_h.size(-1)) ** 0.5)
        sim = sim.masked_fill(q_mask.unsqueeze(1) == 0, -1e9)
        c2q = torch.bmm(torch.softmax(sim, dim=-1), q_h)
        q2c_scores = sim.max(dim=-1).values.masked_fill(c_mask == 0, -1e9)
        q2c = torch.bmm(torch.softmax(q2c_scores, dim=-1).unsqueeze(1), c_h).expand(-1, c_h.size(1), -1)
        fused = torch.cat([c_h, c2q, c_h * c2q, c_h * q2c], dim=-1)
        fused = self.fuse(fused)
        modeled, _ = self.modeling(fused)
        span_features = torch.cat([fused, modeled], dim=-1)
        start_logits = self.start_head(span_features).squeeze(-1)
        end_logits = self.end_head(span_features).squeeze(-1)
        mask_bool = c_mask <= 0.0
        start_logits = start_logits.masked_fill(mask_bool, -1e9)
        end_logits = end_logits.masked_fill(mask_bool, -1e9)
        return start_logits, end_logits


def decode_span_answer(context_tokens: List[str], start_idx: int, end_idx: int) -> str:
    if start_idx < 0 or end_idx < 0 or start_idx >= len(context_tokens):
        return ""
    if end_idx < start_idx:
        end_idx = start_idx
    end_idx = min(end_idx, len(context_tokens) - 1)
    return " ".join(context_tokens[start_idx : end_idx + 1]).strip()


def read_squad_examples(
    split_name: str,
    max_question_tokens: int,
    max_context_tokens: int,
    max_examples: Optional[int] = None,
) -> List[SquadExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("`pip install datasets` 후 다시 실행해 주세요.") from exc

    ds = load_dataset("squad", split=split_name)
    examples: List[SquadExample] = []
    for row in ds:
        context = str(row["context"])
        question = str(row["question"])
        answers_obj = row["answers"]
        starts = answers_obj["answer_start"]
        texts = answers_obj["text"]
        if not starts or not texts:
            continue
        mapped = map_answer_char_span_to_token_span(context, int(starts[0]), str(texts[0]), max_context_tokens)
        if mapped is None:
            continue
        s_idx, e_idx, context_tokens = mapped
        q_tokens = tokenize_text(question)[:max_question_tokens]
        if not q_tokens:
            continue
        examples.append(
            SquadExample(
                question_tokens=q_tokens,
                context_tokens=context_tokens,
                start_idx=s_idx,
                end_idx=e_idx,
                answers=[str(a) for a in texts],
            )
        )
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def append_epoch_metrics_csv(
    csv_path: Path | str,
    seed: int,
    epoch: int,
    train_loss: float,
    dev_loss: float,
    dev_em: float,
    dev_f1: float,
    test_loss: float,
    test_em: float,
    test_f1: float,
) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not path.is_file() or path.stat().st_size == 0
    cols = (
        "timestamp",
        "seed",
        "epoch",
        "train_loss",
        "dev_loss",
        "dev_em",
        "dev_f1",
        "test_loss",
        "test_em",
        "test_f1",
    )
    row = {
        "timestamp": datetime.now(KST).isoformat(),
        "seed": seed,
        "epoch": epoch,
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "dev_em": dev_em,
        "dev_f1": dev_f1,
        "test_loss": test_loss,
        "test_em": test_em,
        "test_f1": test_f1,
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        if need_header:
            w.writeheader()
        w.writerow(row)


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer], device: torch.device) -> float:
    ce = nn.CrossEntropyLoss()
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        q = batch["question_ids"].to(device)
        c = batch["context_ids"].to(device)
        y_s = batch["start_positions"].to(device)
        y_e = batch["end_positions"].to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        s_logits, e_logits = model(q, c)
        loss = 0.5 * (ce(s_logits, y_s) + ce(e_logits, y_e))
        if train_mode:
            loss.backward()
            optimizer.step()
        bs = int(q.size(0))
        total_loss += float(loss.detach().cpu()) * bs
        total_count += bs
    if total_count == 0:
        return 0.0
    return total_loss / total_count


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    ce = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_count = 0
    em_sum = 0.0
    f1_sum = 0.0
    for batch in loader:
        q = batch["question_ids"].to(device)
        c = batch["context_ids"].to(device)
        y_s = batch["start_positions"].to(device)
        y_e = batch["end_positions"].to(device)
        context_tokens: List[List[str]] = batch["context_tokens"]
        answers: List[List[str]] = batch["answers"]
        s_logits, e_logits = model(q, c)
        loss = 0.5 * (ce(s_logits, y_s) + ce(e_logits, y_e))
        s_pred = s_logits.argmax(dim=-1).tolist()
        e_pred = e_logits.argmax(dim=-1).tolist()
        for i in range(len(context_tokens)):
            pred_text = decode_span_answer(context_tokens[i], s_pred[i], e_pred[i])
            em_sum += max(exact_match_score(pred_text, gt) for gt in answers[i])
            f1_sum += max(f1_score(pred_text, gt) for gt in answers[i])
        bs = int(q.size(0))
        total_loss += float(loss.detach().cpu()) * bs
        total_count += bs
    if total_count == 0:
        return 0.0, 0.0, 0.0
    return total_loss / total_count, em_sum / total_count, f1_sum / total_count


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_squad_baseline(
    output_csv: Path = DEFAULT_METRICS_CSV,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    hidden_dim: int = 256,
    glove_dim: int = 300,
    max_question_tokens: int = 48,
    max_context_tokens: int = 400,
    max_train_examples: Optional[int] = 12000,
    max_dev_examples: Optional[int] = 2000,
    max_test_examples: Optional[int] = 2000,
    test_split_name: str = "validation",
    glove_cache_dir: Path = DEFAULT_GLOVE_CACHE,
    device: str = "cpu",
    seeds: Tuple[int, ...] = (13, 17, 23),
) -> None:
    dev = torch.device(device)
    print("--- Loading SQuAD dataset ---")
    train_examples = read_squad_examples("train", max_question_tokens, max_context_tokens, max_train_examples)
    dev_examples = read_squad_examples("validation", max_question_tokens, max_context_tokens, max_dev_examples)
    test_examples = read_squad_examples(test_split_name, max_question_tokens, max_context_tokens, max_test_examples)
    print(f"Loaded SQuAD examples: train={len(train_examples)}, dev={len(dev_examples)}, test={len(test_examples)}")

    vocab = Vocabulary()
    vocab.build_from_examples(train_examples)
    embedding_weights = load_glove_embedding_matrix(vocab, glove_dim=glove_dim, glove_cache_dir=glove_cache_dir)

    train_ds = SquadWordDataset(train_examples, max_question_tokens, max_context_tokens)
    dev_ds = SquadWordDataset(dev_examples, max_question_tokens, max_context_tokens)
    test_ds = SquadWordDataset(test_examples, max_question_tokens, max_context_tokens)
    collate_fn = make_squad_collate_fn(vocab=vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print("--- Training SQuAD GloVe baseline (multi-seed) ---")
    for seed in seeds:
        set_global_seed(seed)
        print(f"\n=== Seed {seed} ===")
        model = SQuADGloveBaselineQA(embedding_weights.clone(), hidden_dim=hidden_dim, pad_id=vocab.pad_id).to(dev)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        for epoch in range(1, epochs + 1):
            train_loss = run_epoch(model, train_loader, optimizer=optimizer, device=dev)
            dev_loss, dev_em, dev_f1 = evaluate(model, dev_loader, device=dev)
            test_loss, test_em, test_f1 = evaluate(model, test_loader, device=dev)
            print(
                f"seed {seed} | epoch {epoch:02d} | train_loss={train_loss:.4f} "
                f"| dev_loss={dev_loss:.4f} dev_EM={dev_em:.4f} dev_F1={dev_f1:.4f} "
                f"| test_loss={test_loss:.4f} test_EM={test_em:.4f} test_F1={test_f1:.4f}"
            )
            append_epoch_metrics_csv(
                output_csv,
                seed,
                epoch,
                train_loss,
                dev_loss,
                dev_em,
                dev_f1,
                test_loss,
                test_em,
                test_f1,
            )
    print(f"Metrics saved to: {output_csv}")


if __name__ == "__main__":
    OUTPUT_CSV = DEFAULT_METRICS_CSV
    EPOCHS = 5
    BATCH_SIZE = 16
    LR = 1e-4
    HIDDEN_DIM = 256
    GLOVE_DIM = 300
    MAX_QUESTION_TOKENS = 48
    MAX_CONTEXT_TOKENS = 400
    MAX_TRAIN_EXAMPLES = None
    MAX_DEV_EXAMPLES = 2000
    MAX_TEST_EXAMPLES = 2000
    TEST_SPLIT_NAME = "validation"
    GLOVE_CACHE_DIR = DEFAULT_GLOVE_CACHE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEEDS = (13, 17, 23)

    train_squad_baseline(
        output_csv=OUTPUT_CSV,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        hidden_dim=HIDDEN_DIM,
        glove_dim=GLOVE_DIM,
        max_question_tokens=MAX_QUESTION_TOKENS,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_train_examples=MAX_TRAIN_EXAMPLES,
        max_dev_examples=MAX_DEV_EXAMPLES,
        max_test_examples=MAX_TEST_EXAMPLES,
        test_split_name=TEST_SPLIT_NAME,
        glove_cache_dir=GLOVE_CACHE_DIR,
        device=DEVICE,
        seeds=SEEDS,
    )
