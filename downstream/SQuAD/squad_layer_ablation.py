#!/usr/bin/env python3
"""
SQuAD layer ablation script (Table-2 style):
- baseline (no ELMo)
- ELMo last layer only
- ELMo all layers (lambda sweep)

Outputs per-setting dev metrics to CSV.
"""

from __future__ import annotations

import csv
import random
import re
import string
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
WORD_SPAN_PATTERN = re.compile(r"\S+")
KST = timezone(timedelta(hours=9))

_THIS_FILE = Path(__file__).resolve()
_BILM_TF_ROOT = next((p for p in _THIS_FILE.parents if p.name == "bilm-tf"), _THIS_FILE.parents[2])
_REPO_ROOT = _BILM_TF_ROOT.parent
DEFAULT_BILM_CHECKPOINT = _REPO_ROOT / "bilm-tf" / "checkpoints" / "bilm" / "final_model.pt"
DEFAULT_VOCAB_FILE = _BILM_TF_ROOT / "bilm" / "data" / "pretrain" / "elmo" / "vocab.txt"
DEFAULT_OUTPUT_CSV = Path(__file__).resolve().parent / "layer_ablation_metrics.csv"


def _ensure_bilm_tf_on_path() -> None:
    root = str(_BILM_TF_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def tokenize_text(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


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
    def __init__(
        self,
        bilm: nn.Module,
        num_layers: int,
        mode: str = "all_layers",
        layer_l2_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        self.bilm = bilm
        self.num_layers = num_layers
        self.mode = mode
        self.layer_l2_lambda = layer_l2_lambda
        for p in self.bilm.parameters():
            p.requires_grad = False
        self.bilm.eval()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, char_ids: Tensor) -> Tensor:
        with torch.no_grad():
            layers = self.bilm(char_ids)
        if self.mode == "last_only":
            out = layers[-1]
        elif self.mode == "all_layers":
            w = F.softmax(self.layer_logits, dim=0)
            out = sum(w[i] * layers[i] for i in range(self.num_layers))
        else:
            raise ValueError(f"Unknown ELMo mode: {self.mode}")
        return self.gamma * out

    def regularization_loss(self) -> Tensor:
        if self.mode != "all_layers" or self.layer_l2_lambda <= 0.0:
            return torch.tensor(0.0, device=self.gamma.device)
        return self.layer_l2_lambda * torch.sum(self.layer_logits ** 2)


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

    def encode_tokens(self, tokens: Sequence[str]) -> Tensor:
        toks = list(tokens)
        if self._vocab is not None:
            arr = self._vocab.encode_chars(toks, split=False)
            return torch.from_numpy(arr).long()
        bos_char, eos_char = 256, 257
        bos = torch.full((self.max_chars_per_token,), 260, dtype=torch.long)
        eos = torch.full((self.max_chars_per_token,), 260, dtype=torch.long)
        bos[0], bos[1], bos[2] = 258, bos_char, 259
        eos[0], eos[1], eos[2] = 258, eos_char, 259
        pieces = [bos] + [self._fallback_word_to_char_ids(t) for t in toks] + [eos]
        return torch.stack(pieces, dim=0)


def map_answer_char_span_to_token_span(
    context: str, answer_start: int, answer_text: str, max_context_tokens: int
) -> Optional[Tuple[int, int, List[str]]]:
    spans = [(m.start(), m.end()) for m in WORD_SPAN_PATTERN.finditer(context)]
    tokens = [context[s:e] for s, e in spans]
    if not tokens:
        return None
    answer_end = answer_start + len(answer_text)
    s_idx = None
    e_idx = None
    for i, (s, e) in enumerate(spans):
        if s_idx is None and s <= answer_start < e:
            s_idx = i
        if s < answer_end <= e:
            e_idx = i
            break
        if answer_start <= s and e <= answer_end:
            if s_idx is None:
                s_idx = i
            e_idx = i
    if s_idx is None or e_idx is None:
        return None
    if s_idx >= max_context_tokens or e_idx >= max_context_tokens:
        return None
    return s_idx + 1, e_idx + 1, tokens[:max_context_tokens]


@dataclass
class SquadExample:
    question_tokens: List[str]
    context_tokens: List[str]
    start_idx: int
    end_idx: int
    answers: List[str]


class SquadCharDataset(Dataset):
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


def make_squad_collate_fn(encoder: CharIdEncoder):
    def _collate(batch: List[SquadExample]) -> Dict[str, Any]:
        q_encoded = [encoder.encode_tokens(ex.question_tokens) for ex in batch]
        c_encoded = [encoder.encode_tokens(ex.context_tokens) for ex in batch]
        max_q = max(int(x.size(0)) for x in q_encoded)
        max_c = max(int(x.size(0)) for x in c_encoded)
        char_len = int(q_encoded[0].size(1))
        q_out = torch.zeros(len(batch), max_q, char_len, dtype=torch.long)
        c_out = torch.zeros(len(batch), max_c, char_len, dtype=torch.long)
        start = torch.zeros(len(batch), dtype=torch.long)
        end = torch.zeros(len(batch), dtype=torch.long)
        answers: List[List[str]] = []
        context_tokens: List[List[str]] = []
        for i, ex in enumerate(batch):
            q_out[i, : q_encoded[i].size(0), :] = q_encoded[i]
            c_out[i, : c_encoded[i].size(0), :] = c_encoded[i]
            start[i] = int(ex.start_idx)
            end[i] = int(ex.end_idx)
            answers.append(ex.answers)
            context_tokens.append(ex.context_tokens)
        return {
            "question_char_ids": q_out,
            "context_char_ids": c_out,
            "start_positions": start,
            "end_positions": end,
            "answers": answers,
            "context_tokens": context_tokens,
        }

    return _collate


class BaselineQA(nn.Module):
    def __init__(self, max_characters_per_token: int, hidden_dim: int, n_characters: int = 261) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_characters, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.modeling = nn.LSTM(
            input_size=hidden_dim * 4,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.start_head = nn.Linear(hidden_dim, 1)
        self.end_head = nn.Linear(hidden_dim, 1)
        self.max_characters_per_token = max_characters_per_token

    def _encode_char_ids(self, char_ids: Tensor) -> Tensor:
        # (B,T,C) -> (B,T,H), simple average over character embeddings
        x = self.embed(char_ids)
        x = x.mean(dim=2)
        return torch.tanh(self.proj(x))

    def forward(self, question_char_ids: Tensor, context_char_ids: Tensor) -> Tuple[Tensor, Tensor]:
        q_h = self._encode_char_ids(question_char_ids)
        c_h = self._encode_char_ids(context_char_ids)
        q_mask = token_mask_from_char_ids(question_char_ids)
        c_mask = token_mask_from_char_ids(context_char_ids)
        q_proj = self.q_proj(q_h)
        c_proj = self.c_proj(c_h)
        sim = torch.bmm(c_proj, q_proj.transpose(1, 2)) / (float(c_h.size(-1)) ** 0.5)
        sim = sim.masked_fill(q_mask.unsqueeze(1) == 0, -1e9)
        c2q = torch.bmm(torch.softmax(sim, dim=-1), q_h)
        q2c_scores = sim.max(dim=-1).values.masked_fill(c_mask == 0, -1e9)
        q2c = torch.bmm(torch.softmax(q2c_scores, dim=-1).unsqueeze(1), c_h).expand(-1, c_h.size(1), -1)
        fused = torch.cat([c_h, c2q, c_h * c2q, c_h * q2c], dim=-1)
        modeled, _ = self.modeling(fused)
        start_logits = self.start_head(modeled).squeeze(-1).masked_fill(c_mask <= 0, -1e9)
        end_logits = self.end_head(modeled).squeeze(-1).masked_fill(c_mask <= 0, -1e9)
        return start_logits, end_logits


class ELMoQA(nn.Module):
    def __init__(self, bilm: nn.Module, hidden_dim: int, num_layers: int, mode: str, layer_l2_lambda: float) -> None:
        super().__init__()
        self.elmo = ELMoEmbedding(bilm, num_layers, mode=mode, layer_l2_lambda=layer_l2_lambda)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.modeling = nn.LSTM(
            input_size=hidden_dim * 4,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.start_head = nn.Linear(hidden_dim, 1)
        self.end_head = nn.Linear(hidden_dim, 1)

    def forward(self, question_char_ids: Tensor, context_char_ids: Tensor) -> Tuple[Tensor, Tensor]:
        q_h = self.elmo(question_char_ids)
        c_h = self.elmo(context_char_ids)
        q_mask = token_mask_from_char_ids(question_char_ids)
        c_mask = token_mask_from_char_ids(context_char_ids)
        q_proj = self.q_proj(q_h)
        c_proj = self.c_proj(c_h)
        sim = torch.bmm(c_proj, q_proj.transpose(1, 2)) / (float(c_h.size(-1)) ** 0.5)
        sim = sim.masked_fill(q_mask.unsqueeze(1) == 0, -1e9)
        c2q = torch.bmm(torch.softmax(sim, dim=-1), q_h)
        q2c_scores = sim.max(dim=-1).values.masked_fill(c_mask == 0, -1e9)
        q2c = torch.bmm(torch.softmax(q2c_scores, dim=-1).unsqueeze(1), c_h).expand(-1, c_h.size(1), -1)
        fused = torch.cat([c_h, c2q, c_h * c2q, c_h * q2c], dim=-1)
        modeled, _ = self.modeling(fused)
        start_logits = self.start_head(modeled).squeeze(-1).masked_fill(c_mask <= 0, -1e9)
        end_logits = self.end_head(modeled).squeeze(-1).masked_fill(c_mask <= 0, -1e9)
        return start_logits, end_logits

    def regularization_loss(self) -> Tensor:
        return self.elmo.regularization_loss()


def decode_span_answer(context_tokens: List[str], start_idx: int, end_idx: int) -> str:
    if start_idx < 0 or start_idx >= len(context_tokens):
        return ""
    if end_idx < start_idx:
        end_idx = start_idx
    end_idx = min(end_idx, len(context_tokens) - 1)
    return " ".join(context_tokens[start_idx : end_idx + 1]).strip()


def read_squad_examples(
    split_name: str, max_question_tokens: int, max_context_tokens: int, max_examples: Optional[int]
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
        q_toks = tokenize_text(question)[:max_question_tokens]
        if not q_toks:
            continue
        examples.append(
            SquadExample(
                question_tokens=q_toks,
                context_tokens=context_tokens,
                start_idx=s_idx,
                end_idx=e_idx,
                answers=[str(a) for a in texts],
            )
        )
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_count = 0
    em_sum = 0.0
    f1_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            q = batch["question_char_ids"].to(device)
            c = batch["context_char_ids"].to(device)
            y_s = batch["start_positions"].to(device)
            y_e = batch["end_positions"].to(device)
            s_logits, e_logits = model(q, c)
            loss = 0.5 * (ce(s_logits, y_s) + ce(e_logits, y_e))
            s_pred = s_logits.argmax(dim=-1).tolist()
            e_pred = e_logits.argmax(dim=-1).tolist()
            for i, context_tokens in enumerate(batch["context_tokens"]):
                pred = decode_span_answer(context_tokens, s_pred[i] - 1, e_pred[i] - 1)
                gts = batch["answers"][i]
                em_sum += max(exact_match_score(pred, gt) for gt in gts)
                f1_sum += max(f1_score(pred, gt) for gt in gts)
            bs = int(q.size(0))
            total_loss += float(loss.detach().cpu()) * bs
            total_count += bs
    if total_count == 0:
        return 0.0, 0.0, 0.0
    return total_loss / total_count, em_sum / total_count, f1_sum / total_count


def train_one_setting(
    setting_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Tuple[float, float]:
    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            q = batch["question_char_ids"].to(device)
            c = batch["context_char_ids"].to(device)
            y_s = batch["start_positions"].to(device)
            y_e = batch["end_positions"].to(device)
            optimizer.zero_grad(set_to_none=True)
            s_logits, e_logits = model(q, c)
            loss = 0.5 * (ce(s_logits, y_s) + ce(e_logits, y_e))
            reg = getattr(model, "regularization_loss", None)
            if callable(reg):
                loss = loss + reg()
            loss.backward()
            optimizer.step()
    _, dev_em, dev_f1 = evaluate(model, dev_loader, device)
    print(f"{setting_name}: dev_EM={dev_em:.4f}, dev_F1={dev_f1:.4f}")
    return dev_em, dev_f1


def append_result(
    csv_path: Path,
    setting: str,
    lambda_value: Optional[float],
    dev_em: float,
    dev_f1: float,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    cols = ("timestamp", "task", "setting", "lambda", "dev_em", "dev_f1")
    row = {
        "timestamp": datetime.now(KST).isoformat(),
        "task": "SQuAD",
        "setting": setting,
        "lambda": "" if lambda_value is None else lambda_value,
        "dev_em": dev_em,
        "dev_f1": dev_f1,
    }
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if need_header:
            w.writeheader()
        w.writerow(row)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_squad_layer_ablation(
    checkpoint_path: Path = DEFAULT_BILM_CHECKPOINT,
    vocab_file: Optional[Path] = DEFAULT_VOCAB_FILE,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-4,
    max_question_tokens: int = 48,
    max_context_tokens: int = 400,
    max_train_examples: Optional[int] = 12000,
    max_dev_examples: Optional[int] = 2000,
    lambdas: Tuple[float, ...] = (1.0, 0.001),
    device: str = "cpu",
    seed: int = 13,
) -> None:
    set_global_seed(seed)
    dev = torch.device(device)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    bilm, num_layers, hidden_dim, options = load_pretrained_char_bilm_from_checkpoint(checkpoint_path, dev)
    encoder = CharIdEncoder(
        max_chars_per_token=int(options["char_cnn"]["max_characters_per_token"]),
        n_characters=int(options["char_cnn"]["n_characters"]),
        vocab_file=vocab_file,
    )
    train_examples = read_squad_examples("train", max_question_tokens, max_context_tokens, max_train_examples)
    dev_examples = read_squad_examples("validation", max_question_tokens, max_context_tokens, max_dev_examples)
    train_ds = SquadCharDataset(train_examples, max_question_tokens, max_context_tokens)
    dev_ds = SquadCharDataset(dev_examples, max_question_tokens, max_context_tokens)
    collate_fn = make_squad_collate_fn(encoder)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    baseline = BaselineQA(
        max_characters_per_token=int(options["char_cnn"]["max_characters_per_token"]),
        hidden_dim=hidden_dim,
        n_characters=int(options["char_cnn"]["n_characters"]),
    ).to(dev)
    em, f1 = train_one_setting("baseline", baseline, train_loader, dev_loader, dev, epochs, lr)
    append_result(output_csv, "baseline", None, em, f1)

    last_only = ELMoQA(
        bilm=load_pretrained_char_bilm_from_checkpoint(checkpoint_path, dev)[0].to(dev),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        mode="last_only",
        layer_l2_lambda=0.0,
    ).to(dev)
    em, f1 = train_one_setting("last_only", last_only, train_loader, dev_loader, dev, epochs, lr)
    append_result(output_csv, "last_only", None, em, f1)

    for lam in lambdas:
        all_layers = ELMoQA(
            bilm=load_pretrained_char_bilm_from_checkpoint(checkpoint_path, dev)[0].to(dev),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mode="all_layers",
            layer_l2_lambda=float(lam),
        ).to(dev)
        em, f1 = train_one_setting(f"all_layers_lambda_{lam}", all_layers, train_loader, dev_loader, dev, epochs, lr)
        append_result(output_csv, "all_layers", float(lam), em, f1)

    print(f"Saved ablation metrics to: {output_csv}")


if __name__ == "__main__":
    CHECKPOINT_PATH = DEFAULT_BILM_CHECKPOINT
    VOCAB_FILE = DEFAULT_VOCAB_FILE
    OUTPUT_CSV = DEFAULT_OUTPUT_CSV
    EPOCHS = 3
    BATCH_SIZE = 16
    LR = 1e-4
    MAX_QUESTION_TOKENS = 48
    MAX_CONTEXT_TOKENS = 400
    MAX_TRAIN_EXAMPLES = 12000
    MAX_DEV_EXAMPLES = 2000
    LAMBDAS = (1.0, 0.001)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 13

    run_squad_layer_ablation(
        checkpoint_path=CHECKPOINT_PATH,
        vocab_file=VOCAB_FILE,
        output_csv=OUTPUT_CSV,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        max_question_tokens=MAX_QUESTION_TOKENS,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_train_examples=MAX_TRAIN_EXAMPLES,
        max_dev_examples=MAX_DEV_EXAMPLES,
        lambdas=LAMBDAS,
        device=DEVICE,
        seed=SEED,
    )
