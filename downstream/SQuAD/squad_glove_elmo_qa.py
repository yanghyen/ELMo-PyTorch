#!/usr/bin/env python3
"""
SQuAD v1.1 with GloVe + ELMo:
- Input: concat([GloVe, ELMo])
- Optional output-side ELMo reinjection
- Multi-seed training and single CSV logging
"""

from __future__ import annotations

import csv
import random
import re
import string
import sys
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
WORD_SPAN_PATTERN = re.compile(r"\S+")
KST = timezone(timedelta(hours=9))
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

_THIS_FILE = Path(__file__).resolve()
_BILM_TF_ROOT = next((p for p in _THIS_FILE.parents if p.name == "bilm-tf"), _THIS_FILE.parents[2])
_REPO_ROOT = _BILM_TF_ROOT.parent
DEFAULT_BILM_CHECKPOINT = _REPO_ROOT / "bilm-tf" / "checkpoints" / "bilm" / "final_model.pt"
DEFAULT_VOCAB_FILE = _BILM_TF_ROOT / "bilm" / "data" / "pretrain" / "elmo" / "vocab.txt"
DEFAULT_METRICS_CSV = Path(__file__).resolve().parent / "glove_elmo_eval_metrics.csv"
DEFAULT_GLOVE_CACHE = Path(__file__).resolve().parent / ".glove_cache"


def _ensure_bilm_tf_on_path() -> None:
    root = str(_BILM_TF_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def tokenize_text(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


def token_mask_from_ids(token_ids: Tensor, pad_id: int) -> Tensor:
    return (token_ids != pad_id).to(dtype=torch.float32, device=token_ids.device)


def token_mask_from_char_ids(char_ids: Tensor) -> Tensor:
    return (char_ids.sum(dim=-1) > 0).to(dtype=torch.float32, device=char_ids.device)


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


class CharBiLMStackAdapter(nn.Module):
    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, char_ids: Tensor) -> List[Tensor]:
        out = self.lm(char_ids)
        if isinstance(out, dict):
            emb = out["lm_embeddings"]
            return [emb[:, i, :, :] for i in range(emb.size(1))]
        if isinstance(out, list):
            return out
        raise TypeError("biLM must return dict with lm_embeddings or list of layers")


def load_pretrained_char_bilm_from_checkpoint(
    checkpoint_path: Path | str,
    map_location: str | torch.device = "cpu",
) -> Tuple[CharBiLMStackAdapter, int, int, Dict[str, Any]]:
    _ensure_bilm_tf_on_path()
    from bilm.src.simple_language_model import SimpleLanguageModel

    device_obj = torch.device(map_location)
    ckpt = torch.load(Path(checkpoint_path), map_location=device_obj)
    options = ckpt["options"]
    state = ckpt["model_state_dict"]
    vocab_size = int(state["output_projection.weight"].shape[0])
    core = SimpleLanguageModel(options, vocab_size)
    core.load_state_dict(state, strict=True)
    core.to(device_obj)
    core.eval()
    adapter = CharBiLMStackAdapter(core)
    with torch.no_grad():
        max_c = int(options["char_cnn"]["max_characters_per_token"])
        probe = torch.zeros(1, 4, max_c, dtype=torch.long, device=device_obj)
        layers = adapter(probe)
    num_layers = len(layers)
    hidden_dim = int(options["lstm"]["dim"]) * 2
    return adapter, num_layers, hidden_dim, options


class ELMoEmbedding(nn.Module):
    def __init__(self, bilm: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.bilm = bilm
        self.num_layers = num_layers
        for p in self.bilm.parameters():
            p.requires_grad = False
        self.bilm.eval()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, char_ids: Tensor) -> Tensor:
        with torch.no_grad():
            layers = self.bilm(char_ids)
        w = F.softmax(self.layer_logits, dim=0)
        out = sum(w[i] * layers[i] for i in range(self.num_layers))
        return self.gamma * out


class CharIdEncoder:
    def __init__(self, max_chars_per_token: int, n_characters: int, vocab_file: Optional[Path]) -> None:
        self.max_chars_per_token = max_chars_per_token
        self.n_characters = n_characters
        self._vocab = None
        if vocab_file is not None:
            _ensure_bilm_tf_on_path()
            from bilm.src.dataset.data import UnicodeCharsVocabulary

            self._vocab = UnicodeCharsVocabulary(str(vocab_file), max_chars_per_token)

    def _fallback_word_to_char_ids(self, word: str) -> Tensor:
        bow_char, eow_char, pad_char = 258, 259, 260
        code = torch.full((self.max_chars_per_token,), pad_char, dtype=torch.long)
        b = word.encode("utf-8", "ignore")[: max(0, self.max_chars_per_token - 2)]
        code[0] = bow_char
        for i, byte in enumerate(b, start=1):
            code[i] = int(byte)
        if len(b) + 1 < self.max_chars_per_token:
            code[len(b) + 1] = eow_char
        return code.clamp(min=0, max=self.n_characters - 1)

    def encode_tokens(self, tokens: List[str]) -> Tensor:
        if self._vocab is not None:
            arr = self._vocab.encode_chars(tokens, split=False)
            return torch.from_numpy(arr).long()
        bos_char, eos_char = 256, 257
        bos = torch.full((self.max_chars_per_token,), 260, dtype=torch.long)
        eos = torch.full((self.max_chars_per_token,), 260, dtype=torch.long)
        bos[0], bos[1], bos[2] = 258, bos_char, 259
        eos[0], eos[1], eos[2] = 258, eos_char, 259
        pieces = [bos] + [self._fallback_word_to_char_ids(t) for t in tokens] + [eos]
        return torch.stack(pieces, dim=0)


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

    def build_from_examples(self, examples: List["SquadExample"]) -> None:
        for ex in examples:
            for t in ex.question_tokens:
                self.add_token(t)
            for t in ex.context_tokens:
                self.add_token(t)

    def encode(self, tokens: List[str]) -> Tensor:
        return torch.tensor([self.token_to_id.get(t.lower(), self.unk_id) for t in tokens], dtype=torch.long)


def ensure_glove_txt(glove_cache_dir: Path, glove_dim: int) -> Path:
    glove_cache_dir.mkdir(parents=True, exist_ok=True)
    txt_path = glove_cache_dir / f"glove.6B.{glove_dim}d.txt"
    if txt_path.is_file():
        return txt_path
    zip_path = glove_cache_dir / "glove.6B.zip"
    if not zip_path.is_file():
        urllib.request.urlretrieve("https://nlp.stanford.edu/data/glove.6B.zip", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        member = f"glove.6B.{glove_dim}d.txt"
        if member not in zf.namelist():
            raise FileNotFoundError(f"{member} not found in {zip_path}")
        zf.extract(member, glove_cache_dir)
    return txt_path


def load_glove_embedding_matrix(vocab: Vocabulary, glove_dim: int, glove_cache_dir: Path) -> Tensor:
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
            matrix[vocab.token_to_id[token]] = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
            hit += 1
            needed.remove(token)
            if not needed:
                break
    print(f"GloVe coverage: {hit}/{max(1, len(vocab.id_to_token)-2)}")
    return matrix


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


class SquadDataset(Dataset):
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


def make_collate_fn(vocab: Vocabulary, char_encoder: CharIdEncoder):
    def _collate(batch: List[SquadExample]) -> Dict[str, Any]:
        q_word = [vocab.encode(ex.question_tokens) for ex in batch]
        c_word = [vocab.encode(ex.context_tokens) for ex in batch]
        q_char = [char_encoder.encode_tokens(ex.question_tokens) for ex in batch]
        c_char = [char_encoder.encode_tokens(ex.context_tokens) for ex in batch]

        max_qw = max(int(x.size(0)) for x in q_word)
        max_cw = max(int(x.size(0)) for x in c_word)
        max_qc = max(int(x.size(0)) for x in q_char)
        max_cc = max(int(x.size(0)) for x in c_char)
        char_len = int(q_char[0].size(1))

        q_word_out = torch.full((len(batch), max_qw), vocab.pad_id, dtype=torch.long)
        c_word_out = torch.full((len(batch), max_cw), vocab.pad_id, dtype=torch.long)
        q_char_out = torch.zeros(len(batch), max_qc, char_len, dtype=torch.long)
        c_char_out = torch.zeros(len(batch), max_cc, char_len, dtype=torch.long)
        start = torch.zeros(len(batch), dtype=torch.long)
        end = torch.zeros(len(batch), dtype=torch.long)
        answers: List[List[str]] = []
        context_tokens: List[List[str]] = []

        for i, ex in enumerate(batch):
            q_word_out[i, : q_word[i].size(0)] = q_word[i]
            c_word_out[i, : c_word[i].size(0)] = c_word[i]
            q_char_out[i, : q_char[i].size(0), :] = q_char[i]
            c_char_out[i, : c_char[i].size(0), :] = c_char[i]
            start[i] = int(ex.start_idx)
            end[i] = int(ex.end_idx)
            answers.append(ex.answers)
            context_tokens.append(ex.context_tokens)

        return {
            "question_word_ids": q_word_out,
            "context_word_ids": c_word_out,
            "question_char_ids": q_char_out,
            "context_char_ids": c_char_out,
            "start_positions": start,
            "end_positions": end,
            "answers": answers,
            "context_tokens": context_tokens,
        }

    return _collate


class SquadGloveELMoQA(nn.Module):
    def __init__(
        self,
        embedding_weights: Tensor,
        bilm: nn.Module,
        num_layers: int,
        elmo_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        pad_id: int = 0,
        use_output_elmo: bool = False,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.use_output_elmo = use_output_elmo
        glove_dim = int(embedding_weights.size(1))
        self.glove = nn.Embedding.from_pretrained(embedding_weights, freeze=False, padding_idx=pad_id)
        self.elmo = ELMoEmbedding(bilm, num_layers=num_layers)

        self.in_proj = nn.Linear(glove_dim + elmo_dim, hidden_dim * 2)
        enc_dim = hidden_dim * 2
        self.encoder = nn.LSTM(
            input_size=enc_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
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
        self.output_elmo_proj = nn.Linear(elmo_dim, enc_dim)
        self.start_head = nn.Linear(enc_dim * 2, 1)
        self.end_head = nn.Linear(enc_dim * 2, 1)

    def _encode(self, word_ids: Tensor, char_ids: Tensor) -> Tensor:
        x_glove = self.glove(word_ids)
        x_elmo = self.elmo(char_ids)
        if x_elmo.size(1) == x_glove.size(1) + 2:
            x_elmo = x_elmo[:, 1:-1, :]
        elif x_elmo.size(1) != x_glove.size(1):
            min_len = min(x_elmo.size(1), x_glove.size(1))
            x_elmo = x_elmo[:, :min_len, :]
            x_glove = x_glove[:, :min_len, :]

        x = torch.cat([x_glove, x_elmo], dim=-1)
        x = self.in_proj(x)
        out, _ = self.encoder(x)
        return out

    def forward(
        self,
        question_word_ids: Tensor,
        context_word_ids: Tensor,
        question_char_ids: Tensor,
        context_char_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        q_h = self._encode(question_word_ids, question_char_ids)
        c_h = self._encode(context_word_ids, context_char_ids)
        q_mask = token_mask_from_ids(question_word_ids, self.pad_id)
        c_mask = token_mask_from_ids(context_word_ids, self.pad_id)

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

        if self.use_output_elmo:
            if c_elmo.size(1) == span_features.size(1) + 2:
                c_elmo = c_elmo[:, 1:-1, :]
            c_elmo_proj = self.output_elmo_proj(c_elmo)
            span_features = span_features + torch.cat([c_elmo_proj, c_elmo_proj], dim=-1)

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
        starts = row["answers"]["answer_start"]
        texts = row["answers"]["text"]
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
        qw = batch["question_word_ids"].to(device)
        cw = batch["context_word_ids"].to(device)
        qc = batch["question_char_ids"].to(device)
        cc = batch["context_char_ids"].to(device)
        y_s = batch["start_positions"].to(device)
        y_e = batch["end_positions"].to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        s_logits, e_logits = model(qw, cw, qc, cc)
        loss = 0.5 * (ce(s_logits, y_s) + ce(e_logits, y_e))
        if train_mode:
            loss.backward()
            optimizer.step()
        bs = int(qw.size(0))
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
        qw = batch["question_word_ids"].to(device)
        cw = batch["context_word_ids"].to(device)
        qc = batch["question_char_ids"].to(device)
        cc = batch["context_char_ids"].to(device)
        y_s = batch["start_positions"].to(device)
        y_e = batch["end_positions"].to(device)
        context_tokens: List[List[str]] = batch["context_tokens"]
        answers: List[List[str]] = batch["answers"]
        s_logits, e_logits = model(qw, cw, qc, cc)
        loss = 0.5 * (ce(s_logits, y_s) + ce(e_logits, y_e))
        s_pred = s_logits.argmax(dim=-1).tolist()
        e_pred = e_logits.argmax(dim=-1).tolist()
        for i in range(len(context_tokens)):
            pred = decode_span_answer(context_tokens[i], s_pred[i], e_pred[i])
            em_sum += max(exact_match_score(pred, gt) for gt in answers[i])
            f1_sum += max(f1_score(pred, gt) for gt in answers[i])
        bs = int(qw.size(0))
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


def train_squad_glove_elmo(
    checkpoint_path: Path = DEFAULT_BILM_CHECKPOINT,
    vocab_file: Optional[Path] = DEFAULT_VOCAB_FILE,
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
    seeds: Tuple[int, ...] = (13, 17, 23),
    use_output_elmo: bool = False,
    device: str = "cpu",
) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if vocab_file is not None and not vocab_file.is_file():
        raise FileNotFoundError(f"vocab_file not found: {vocab_file}")

    dev = torch.device(device)
    print("--- Loading SQuAD dataset ---")
    train_examples = read_squad_examples("train", max_question_tokens, max_context_tokens, max_train_examples)
    dev_examples = read_squad_examples("validation", max_question_tokens, max_context_tokens, max_dev_examples)
    test_examples = read_squad_examples(test_split_name, max_question_tokens, max_context_tokens, max_test_examples)
    print(f"Loaded SQuAD examples: train={len(train_examples)}, dev={len(dev_examples)}, test={len(test_examples)}")

    vocab = Vocabulary()
    vocab.build_from_examples(train_examples)
    glove_matrix = load_glove_embedding_matrix(vocab, glove_dim=glove_dim, glove_cache_dir=glove_cache_dir)

    bilm, num_layers, elmo_dim, options = load_pretrained_char_bilm_from_checkpoint(checkpoint_path, map_location=dev)
    char_encoder = CharIdEncoder(
        max_chars_per_token=int(options["char_cnn"]["max_characters_per_token"]),
        n_characters=int(options["char_cnn"]["n_characters"]),
        vocab_file=vocab_file,
    )

    train_ds = SquadDataset(train_examples, max_question_tokens, max_context_tokens)
    dev_ds = SquadDataset(dev_examples, max_question_tokens, max_context_tokens)
    test_ds = SquadDataset(test_examples, max_question_tokens, max_context_tokens)
    collate_fn = make_collate_fn(vocab=vocab, char_encoder=char_encoder)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print("--- Training SQuAD GloVe+ELMo (multi-seed) ---")
    for seed in seeds:
        set_global_seed(seed)
        print(f"\n=== Seed {seed} ===")
        seed_bilm, _, _, _ = load_pretrained_char_bilm_from_checkpoint(checkpoint_path, map_location=dev)
        seed_bilm.to(dev)
        model = SquadGloveELMoQA(
            embedding_weights=glove_matrix.clone(),
            bilm=seed_bilm,
            num_layers=num_layers,
            elmo_dim=elmo_dim,
            hidden_dim=hidden_dim,
            pad_id=vocab.pad_id,
            use_output_elmo=use_output_elmo,
        ).to(dev)
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
    CHECKPOINT_PATH = DEFAULT_BILM_CHECKPOINT
    VOCAB_FILE = DEFAULT_VOCAB_FILE
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
    SEEDS = (13, 17, 23)
    USE_OUTPUT_ELMO = False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_squad_glove_elmo(
        checkpoint_path=CHECKPOINT_PATH,
        vocab_file=VOCAB_FILE,
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
        seeds=SEEDS,
        use_output_elmo=USE_OUTPUT_ELMO,
        device=DEVICE,
    )
