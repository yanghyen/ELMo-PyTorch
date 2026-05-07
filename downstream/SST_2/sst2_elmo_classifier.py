#!/usr/bin/env python3
"""
SST-2 style sentiment classification with a frozen pretrained ELMo-style biLM.

Loads the PyTorch biLM checkpoint produced by ``bilm-tf`` training
(``SimpleLanguageModel``). Default checkpoint path (under this repo):

    bilm-tf/checkpoints/bilm/final_moedl.pt

That model exposes **two** ``lm_embeddings`` slices (skip-duplicate vs
concatenated states), so ``num_layers=2`` for ``ELMoEmbedding``.

Sequence pooling defaults to **attention** over valid time steps; padding is
masked using all-zero character slots (same idea as the biLM) or ``pad_token_id``
for the baseline. Classifier head is a small MLP (Linear–ReLU–Dropout–Linear).
``batch_accuracy`` / ``eval_batch_metrics`` support quick eval.
Eval lines can be appended via ``append_eval_metrics_csv``.

Usage:
    PYTHONPATH=/path/to/ELMo_repo/bilm-tf python downstream/SST_2/sst2_elmo_classifier.py

Or run from repo root; this file prepends ``<repo>/bilm-tf`` to ``sys.path``.
"""

from __future__ import annotations

import csv
import re
import sys
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


PoolingMode = Literal["attention", "mean", "max"]
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
KST = timezone(timedelta(hours=9))


def token_mask_from_char_ids(char_ids: Tensor) -> Tensor:
    """
    (batch, seq_len, char_len) -> (batch, seq_len) float {0,1}.
    Matches biLM convention: padded word slots are all-zero characters.
    """
    return (char_ids.sum(dim=-1) > 0).to(dtype=torch.float32, device=char_ids.device)


def token_mask_from_token_ids(token_ids: Tensor, pad_id: int) -> Tensor:
    """(batch, seq_len) -> (batch, seq_len) float mask, 1 where token != pad_id."""
    return (token_ids != pad_id).to(dtype=torch.float32, device=token_ids.device)


def pool_sequence(
    h: Tensor,
    mask: Tensor,
    mode: PoolingMode,
    attn_linear: Optional[nn.Module],
) -> Tensor:
    """
    h: (B, T, D), mask: (B, T) with 1 = valid, 0 = pad.
    """
    if mask.shape != h.shape[:2]:
        raise ValueError("mask must be (B, T) matching h[:, :, 0]")
    m = mask.unsqueeze(-1)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1e-6)

    if mode == "mean":
        return (h * m).sum(dim=1) / denom

    if mode == "max":
        h_masked = h.masked_fill(m == 0, -1e4)
        return h_masked.max(dim=1).values

    if mode == "attention":
        if attn_linear is None:
            raise ValueError("attention pooling requires attn_linear")
        scores = attn_linear(h)  # (B, T, H_attn)
        if scores.dim() == 2:
            scores = scores.unsqueeze(-1)
        if scores.dim() != 3:
            raise ValueError("attention scorer must produce (B, T) or (B, T, H_attn)")
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=1)  # (B, T, H_attn)
        attn = torch.nan_to_num(attn, nan=0.0)
        # Weighted sum per head, then mean over heads (multi-head scalar attention)
        pooled_heads = (attn.unsqueeze(-1) * h.unsqueeze(2)).sum(dim=1)  # (B, H_attn, D)
        return pooled_heads.mean(dim=1)  # (B, D)

    raise ValueError(f"Unknown pooling mode: {mode}")


@torch.no_grad()
def batch_accuracy(logits: Tensor, labels: Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().cpu())


# ---------------------------------------------------------------------------
# Repo paths + checkpoint loading (bilm-tf SimpleLanguageModel)
# ---------------------------------------------------------------------------

# Resolve bilm-tf root robustly whether this file sits under
# "<workspace>/downstream/..." or "<workspace>/bilm-tf/downstream/...".
_THIS_FILE = Path(__file__).resolve()
_BILM_TF_ROOT = next(
    (p for p in _THIS_FILE.parents if p.name == "bilm-tf"),
    _THIS_FILE.parents[2],
)
_REPO_ROOT = _BILM_TF_ROOT.parent
# ``/home/ssai/Workspace/ELMo_repo/bilm-tf/checkpoints/bilm/...``).
DEFAULT_BILM_CHECKPOINT = (
    _REPO_ROOT / "bilm-tf" / "checkpoints" / "bilm" / "final_model.pt"
)
DEFAULT_VOCAB_FILE = (
    _BILM_TF_ROOT / "bilm" / "data" / "pretrain" / "elmo" / "vocab.txt"
)
# 평가 결과 append 저장 (데모 / 스크립트 실행 시)
DEFAULT_EVAL_CSV = Path(__file__).resolve().parent / "elmo_eval_metrics.csv"
DEFAULT_DEMO_EVAL_CSV = Path(__file__).resolve().parent / "elmo_demo_eval_metrics.csv"
EVAL_CSV_COLUMNS = ("timestamp", "phase", "loss", "accuracy")
SST2_ZIP_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"


def _ensure_bilm_tf_on_path() -> None:
    root = str(_BILM_TF_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


class CharBiLMStackAdapter(nn.Module):
    """
    Wraps ``SimpleLanguageModel`` (dict output) to match the downstream contract:
    ``forward(char_ids) -> List[Tensor]`` with one tensor per ``lm_embeddings`` slice.
    """

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, char_ids: Tensor) -> List[Tensor]:
        out = self.lm(char_ids)
        if isinstance(out, dict):
            emb = out["lm_embeddings"]
            return [emb[:, i, :, :] for i in range(emb.size(1))]
        if not isinstance(out, list):
            raise TypeError("biLM must return dict with lm_embeddings or a list of layers")
        return out


def load_pretrained_char_bilm_from_checkpoint(
    checkpoint_path: Path | str,
    map_location: str | torch.device = "cpu",
) -> Tuple[CharBiLMStackAdapter, int, int, Dict[str, Any]]:
    """
    Load ``checkpoint_step_*.pt`` from bilm-tf training.

    Returns:
        adapter: ``CharBiLMStackAdapter`` around ``SimpleLanguageModel``
        num_layers: length of the layer list (2 for current checkpoints)
        embedding_dim: last dimension of each layer tensor (here ``2 * lstm.dim``)
        options: training options dict (e.g. ``max_characters_per_token``)
    """
    _ensure_bilm_tf_on_path()
    from bilm.src.simple_language_model import SimpleLanguageModel

    device_obj = torch.device(map_location)
    path = Path(checkpoint_path)
    ckpt = torch.load(path, map_location=device_obj)
    options = ckpt["options"]
    state = ckpt["model_state_dict"]
    vocab_size = int(state["output_projection.weight"].shape[0])

    core = SimpleLanguageModel(options, vocab_size)
    core.load_state_dict(state, strict=True)
    core.to(device_obj)
    core.eval()

    adapter = CharBiLMStackAdapter(core)
    lstm_dim = int(options["lstm"]["dim"])
    embedding_dim = 2 * lstm_dim

    with torch.no_grad():
        max_c = int(options["char_cnn"]["max_characters_per_token"])
        n_char = int(options["char_cnn"]["n_characters"])
        probe = torch.zeros(1, 3, max_c, dtype=torch.long, device=device_obj)
        layers = adapter(probe)
    num_layers = len(layers)
    for li, t in enumerate(layers):
        if t.shape[-1] != embedding_dim:
            raise ValueError(
                f"Layer {li} embedding dim {t.shape[-1]} != expected {embedding_dim}"
            )
    return adapter, num_layers, embedding_dim, options


# ---------------------------------------------------------------------------
# Dummy biLM (fallback demo when checkpoint is missing)
# ---------------------------------------------------------------------------


class DummyCharBiLM(nn.Module):
    """Minimal stand-in: char_ids (B, T, C) -> list of ``n_layers`` tensors (B, T, H)."""

    def __init__(
        self,
        hidden_dim: int = 32,
        char_vocab_size: int = 128,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.projs = nn.ModuleList(
            nn.Linear(char_vocab_size, hidden_dim) for _ in range(n_layers)
        )

    def forward(self, char_ids: Tensor) -> List[Tensor]:
        b, t, c = char_ids.shape
        x = char_ids.float().mean(dim=-1, keepdim=True)
        x = x.expand(-1, -1, self.projs[0].in_features)
        h = torch.tanh(self.projs[0](x))
        out = [h]
        for i in range(1, self.n_layers):
            h = torch.tanh(self.projs[i](h))
            out.append(h)
        return out


# ---------------------------------------------------------------------------
# ELMo embedding: learned layer weights + gamma (biLM frozen)
# ---------------------------------------------------------------------------


LayerMode = Union[Literal["weighted"], int]


class ELMoEmbedding(nn.Module):
    """
    Wraps a pretrained biLM and produces a single sequence of vectors per layer
    choice:

    * ``layer_mode="weighted"`` (default): softmax over trainable scalars ``s_i``
      (one per biLM layer), then multiply by ``gamma``.
    * ``layer_mode=k`` (int): use only layer ``k`` (still scaled by ``gamma``).
      Indices are ``0 .. num_layers-1``. The ``s_i`` parameters get
      ``requires_grad=False`` in this mode.

    **Where the biLM is frozen:** every parameter of ``bilm`` has
    ``requires_grad=False`` in ``__init__``, and ``bilm.eval()`` is called.

    **Switching single-layer mode:** ``model.elmo.set_layer_mode(0)`` (or ``1``, …).
    """

    def __init__(
        self,
        bilm: nn.Module,
        num_layers: int,
        layer_mode: LayerMode = "weighted",
    ) -> None:
        super().__init__()
        self.bilm = bilm
        self.num_layers = num_layers
        self.layer_mode: LayerMode = layer_mode

        for p in self.bilm.parameters():
            p.requires_grad = False
        self.bilm.eval()

        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.gamma = nn.Parameter(torch.tensor(1.0))

        self._apply_layer_mode_to_logits()

    def set_layer_mode(self, mode: LayerMode) -> None:
        self.layer_mode = mode
        self._apply_layer_mode_to_logits()

    def _apply_layer_mode_to_logits(self) -> None:
        if self.layer_mode == "weighted":
            self.layer_logits.requires_grad = True
        else:
            self.layer_logits.requires_grad = False

    def forward(self, char_ids: Tensor) -> Tensor:
        with torch.no_grad():
            layers = self.bilm(char_ids)
        if len(layers) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} biLM layers, got {len(layers)}"
            )

        if self.layer_mode == "weighted":
            w = F.softmax(self.layer_logits, dim=0)
            out = sum(w[i] * layers[i] for i in range(self.num_layers))
        else:
            idx = int(self.layer_mode)
            if not 0 <= idx < self.num_layers:
                raise IndexError(
                    f"layer_mode {idx} out of range for num_layers={self.num_layers}"
                )
            out = layers[idx]

        return self.gamma * out


# ---------------------------------------------------------------------------
# SST classifier (ELMo path)
# ---------------------------------------------------------------------------


class SSTClassifier(nn.Module):
    """
    char_ids (B, T, C) -> ELMo -> masked pool over T -> MLP -> logits (B, 2).

    Pooling: ``attention`` (default, learnable per-position weights), ``mean``,
    or ``max``. Padding is inferred from char ids (all-zero word = pad), same
    as the pretrained biLM mask.
    """

    def __init__(
        self,
        bilm: nn.Module,
        hidden_dim: int,
        num_layers: int,
        layer_mode: LayerMode = "weighted",
        pooling: PoolingMode = "attention",
        classifier_dropout: float = 0.2,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.pooling: PoolingMode = pooling
        self.elmo = ELMoEmbedding(bilm, num_layers=num_layers, layer_mode=layer_mode)
        self.attn_scorer: Optional[nn.Linear] = (
            nn.Linear(hidden_dim, attn_heads) if pooling == "attention" else None
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, char_ids: Tensor, token_mask: Optional[Tensor] = None) -> Tensor:
        h = self.elmo(char_ids)
        if token_mask is None:
            mask = token_mask_from_char_ids(char_ids)
        else:
            mask = token_mask.to(dtype=h.dtype, device=h.device)
        pooled = pool_sequence(h, mask, self.pooling, self.attn_scorer)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Baseline: token embedding + mean pool + linear (no biLM)
# ---------------------------------------------------------------------------


class BaselineEmbeddingClassifier(nn.Module):
    """
    Token ids (B, T) -> nn.Embedding -> masked pool -> small MLP -> (B, num_labels).

    ``pad_token_id`` positions are ignored in pooling (default ``0``).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_labels: int = 2,
        pad_token_id: int = 0,
        pooling: PoolingMode = "attention",
        classifier_dropout: float = 0.2,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.pooling: PoolingMode = pooling
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attn_scorer: Optional[nn.Linear] = (
            nn.Linear(embed_dim, attn_heads) if pooling == "attention" else None
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(embed_dim, num_labels),
        )

    def forward(self, token_ids: Tensor, token_mask: Optional[Tensor] = None) -> Tensor:
        e = self.embedding(token_ids)
        if token_mask is None:
            mask = token_mask_from_token_ids(token_ids, self.pad_token_id)
        else:
            mask = token_mask.to(dtype=e.dtype, device=e.device)
        pooled = pool_sequence(e, mask, self.pooling, self.attn_scorer)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def trainable_parameters(module: nn.Module) -> List[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def simple_train_step(
    model: nn.Module,
    batch_x: Tensor,
    batch_y: Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def eval_batch_metrics(
    model: nn.Module,
    batch_x: Tensor,
    batch_y: Tensor,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """Returns (loss, accuracy) on a single batch."""
    model.eval()
    logits = model(batch_x)
    loss = float(criterion(logits, batch_y).detach().cpu())
    acc = batch_accuracy(logits, batch_y)
    return loss, acc


def append_eval_metrics_csv(
    csv_path: Path | str,
    phase: str,
    loss: float,
    accuracy: float,
) -> Path:
    """
    평가 한 줄을 CSV에 append. 파일이 없거나 비어 있으면 헤더를 씁니다.
    Returns the resolved path written to.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not path.is_file() or path.stat().st_size == 0
    row = {
        "timestamp": datetime.now(KST).isoformat(),
        "phase": phase,
        "loss": loss,
        "accuracy": accuracy,
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(EVAL_CSV_COLUMNS))
        if need_header:
            w.writeheader()
        w.writerow(row)
    return path


def append_epoch_metrics_csv(
    csv_path: Path | str,
    epoch: int,
    train_loss: float,
    train_acc: float,
    dev_loss: float,
    dev_acc: float,
) -> Path:
    """
    Epoch-level metrics for real SST-2 training.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not path.is_file() or path.stat().st_size == 0
    cols = ("timestamp", "epoch", "train_loss", "train_acc", "dev_loss", "dev_acc")
    row = {
        "timestamp": datetime.now(KST).isoformat(),
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "dev_loss": dev_loss,
        "dev_acc": dev_acc,
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        if need_header:
            w.writeheader()
        w.writerow(row)
    return path


def ensure_sst2_dataset(data_dir: Path) -> Path:
    """
    Ensure GLUE SST-2 train/dev files exist locally.
    If missing, download SST-2.zip and extract into data_dir's parent.
    """
    train_tsv = data_dir / "train.tsv"
    dev_tsv = data_dir / "dev.tsv"
    if train_tsv.is_file() and dev_tsv.is_file():
        return data_dir

    parent = data_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    zip_path = parent / "SST-2.zip"
    print(f"SST-2 not found at {data_dir}. Downloading from {SST2_ZIP_URL}")
    urllib.request.urlretrieve(SST2_ZIP_URL, zip_path)
    print(f"Downloaded: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(parent)
    print(f"Extracted SST-2 archive under: {parent}")

    extracted_default = parent / "SST-2"
    if extracted_default.is_dir():
        data_dir = extracted_default
        train_tsv = data_dir / "train.tsv"
        dev_tsv = data_dir / "dev.tsv"
        print(f"Using extracted dataset dir: {data_dir}")

    if not train_tsv.is_file() or not dev_tsv.is_file():
        raise FileNotFoundError(
            f"Downloaded archive but train/dev not found at {train_tsv} / {dev_tsv}"
        )
    return data_dir


def read_sst2_tsv(path: Path, has_labels: bool) -> List[Tuple[str, int]]:
    """
    Read GLUE SST-2 tsv.
    - train/dev: columns include sentence + label
    - test: sentence only (label set to -1 when has_labels=False)
    """
    rows: List[Tuple[str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            sent = r.get("sentence", "")
            if not sent:
                continue
            if has_labels:
                label = int(r["label"])
            else:
                label = -1
            rows.append((sent, label))
    return rows


def tokenize_text(sentence: str) -> List[str]:
    """
    Regex tokenizer that keeps punctuation as separate tokens.
    """
    return TOKEN_PATTERN.findall(sentence)


class CharIdEncoder:
    """
    Character id encoder for downstream SST-2.

    Priority:
    1) If `vocab_file` is provided, use ``UnicodeCharsVocabulary.encode_chars``
       from bilm (recommended; matches pretraining char conversion pipeline).
    2) Else fallback to utf-8 byte based encoding with ELMo-like special ids.
       (reasonable fallback, but still weaker than using original vocab file)
    """

    def __init__(self, max_chars_per_token: int, n_characters: int, vocab_file: Optional[Path]):
        self.max_chars_per_token = max_chars_per_token
        self.n_characters = n_characters
        self.vocab_file = vocab_file
        self._vocab = None
        if vocab_file is not None:
            _ensure_bilm_tf_on_path()
            from bilm.src.dataset.data import UnicodeCharsVocabulary

            self._vocab = UnicodeCharsVocabulary(str(vocab_file), max_chars_per_token)

    def _fallback_word_to_char_ids(self, word: str) -> Tensor:
        # ELMo-compatible layout (0..260 special range expected by n_characters=261)
        bow_char, eow_char, pad_char = 258, 259, 260
        code = torch.full((self.max_chars_per_token,), pad_char, dtype=torch.long)
        b = word.encode("utf-8", "ignore")[: max(0, self.max_chars_per_token - 2)]
        code[0] = bow_char
        for i, byte in enumerate(b, start=1):
            code[i] = int(byte)
        if len(b) + 1 < self.max_chars_per_token:
            code[len(b) + 1] = eow_char
        return code.clamp(min=0, max=self.n_characters - 1)

    def encode_tokens(self, tokens: List[str], max_tokens: Optional[int] = None) -> Tensor:
        toks = tokens if max_tokens is None else tokens[:max_tokens]
        if self._vocab is not None:
            arr = self._vocab.encode_chars(toks, split=False)  # includes BOS/EOS
            return torch.from_numpy(arr).long()

        bos_char, eos_char = 256, 257
        bos = torch.full((self.max_chars_per_token,), 260, dtype=torch.long)
        eos = torch.full((self.max_chars_per_token,), 260, dtype=torch.long)
        bos[0], bos[1], bos[2] = 258, bos_char, 259
        eos[0], eos[1], eos[2] = 258, eos_char, 259
        pieces = [bos]
        for t in toks:
            pieces.append(self._fallback_word_to_char_ids(t))
        pieces.append(eos)
        return torch.stack(pieces, dim=0)


class SST2CharDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        max_tokens: int,
    ) -> None:
        self.samples = samples
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[str], Tensor]:
        sentence, label = self.samples[idx]
        tokens = tokenize_text(sentence)[: self.max_tokens]
        y = torch.tensor(label, dtype=torch.long)
        return tokens, y


def make_sst2_collate_fn(encoder: CharIdEncoder):
    def _collate(batch: List[Tuple[List[str], Tensor]]) -> Tuple[Tensor, Tensor]:
        tokens_list = [b[0] for b in batch]
        labels = torch.stack([b[1] for b in batch], dim=0)
        encoded = [encoder.encode_tokens(toks) for toks in tokens_list]
        max_len = max(int(x.size(0)) for x in encoded)
        char_len = int(encoded[0].size(1))
        out = torch.zeros(len(encoded), max_len, char_len, dtype=torch.long)
        for i, x in enumerate(encoded):
            out[i, : x.size(0), :] = x
        return out, labels

    return _collate


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        if train_mode:
            loss.backward()
            optimizer.step()

        bs = int(batch_y.size(0))
        total_loss += float(loss.detach().cpu()) * bs
        total_correct += float((logits.argmax(dim=-1) == batch_y).float().sum().cpu())
        total_count += bs

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def simple_training_loop(
    model: nn.Module,
    char_ids: Tensor,
    labels: Tensor,
    num_epochs: int = 2,
    lr: float = 1e-3,
) -> None:
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        loss = simple_train_step(model, char_ids, labels, optimizer, criterion)
        print(f"epoch {epoch + 1}: loss={loss:.4f}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _demo() -> None:
    device = torch.device("cpu")
    torch.manual_seed(0)

    batch_size, seq_len = 4, 12
    num_steps = 3

    ckpt_path = DEFAULT_BILM_CHECKPOINT
    if ckpt_path.is_file():
        print(f"--- Loading biLM from {ckpt_path} ---")
        bilm, num_layers, hidden_dim, options = load_pretrained_char_bilm_from_checkpoint(
            ckpt_path, map_location=device
        )
        bilm.to(device)
        char_len = int(options["char_cnn"]["max_characters_per_token"])
        char_vocab = int(options["char_cnn"]["n_characters"])
        char_ids = torch.randint(
            0, char_vocab, (batch_size, seq_len, char_len), device=device
        )
        model = SSTClassifier(
            bilm, hidden_dim=hidden_dim, num_layers=num_layers, layer_mode="weighted"
        ).to(device)
    else:
        print(f"--- Checkpoint not found at {ckpt_path}; using DummyCharBiLM ---")
        char_len = 16
        char_vocab = 128
        hidden_dim = 32
        num_layers = 3
        bilm = DummyCharBiLM(
            hidden_dim=hidden_dim, char_vocab_size=char_vocab, n_layers=num_layers
        ).to(device)
        char_ids = torch.randint(
            0, char_vocab, (batch_size, seq_len, char_len), device=device
        )
        model = SSTClassifier(
            bilm, hidden_dim=hidden_dim, num_layers=num_layers, layer_mode="weighted"
        ).to(device)

    # Last 3 word slots: all-zero chars => excluded by token_mask_from_char_ids
    char_ids[:, -3:, :] = 0

    labels = torch.randint(0, 2, (batch_size,), device=device)

    bilm_trainable = sum(p.numel() for p in bilm.parameters() if p.requires_grad)
    assert bilm_trainable == 0, "biLM must stay frozen"

    opt = torch.optim.Adam(trainable_parameters(model), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    print("--- ELMo SSTClassifier (weighted layers, attention pool + MLP head) ---")
    for step in range(num_steps):
        loss = simple_train_step(model, char_ids, labels, opt, ce)
        print(f"step {step + 1}: loss={loss:.4f}")
    ev_loss, ev_acc = eval_batch_metrics(model, char_ids, labels, ce)
    print(f"eval: loss={ev_loss:.4f}  acc={ev_acc:.4f}")
    append_eval_metrics_csv(
        DEFAULT_DEMO_EVAL_CSV, "elmo_weighted", ev_loss, ev_acc
    )

    last_idx = num_layers - 1
    model.elmo.set_layer_mode(last_idx)
    print(f"\n--- Single-layer mode (layer {last_idx} only) ---")
    opt2 = torch.optim.Adam(trainable_parameters(model), lr=1e-3)
    for step in range(num_steps):
        loss = simple_train_step(model, char_ids, labels, opt2, ce)
        print(f"step {step + 1}: loss={loss:.4f}")
    ev_loss, ev_acc = eval_batch_metrics(model, char_ids, labels, ce)
    print(f"eval: loss={ev_loss:.4f}  acc={ev_acc:.4f}")
    append_eval_metrics_csv(
        DEFAULT_DEMO_EVAL_CSV, f"elmo_single_layer_{last_idx}", ev_loss, ev_acc
    )

    print("\n--- Baseline nn.Embedding classifier ---")
    vocab_size = 5000
    embed_dim = 64
    baseline = BaselineEmbeddingClassifier(vocab_size, embed_dim).to(device)
    opt_b = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    token_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    token_ids[:, -2:] = 0  # pad_token_id=0, masked in pool
    for step in range(num_steps):
        loss = simple_train_step(baseline, token_ids, labels, opt_b, ce)
        print(f"step {step + 1}: loss={loss:.4f}")
    ev_loss, ev_acc = eval_batch_metrics(baseline, token_ids, labels, ce)
    print(f"eval: loss={ev_loss:.4f}  acc={ev_acc:.4f}")
    append_eval_metrics_csv(DEFAULT_DEMO_EVAL_CSV, "baseline_embedding", ev_loss, ev_acc)

    print(f"\nEval metrics appended to: {DEFAULT_DEMO_EVAL_CSV}")
    print("Demo finished.")


def train_on_real_sst2(
    data_dir: Path,
    checkpoint_path: Path = DEFAULT_BILM_CHECKPOINT,
    output_csv: Path = DEFAULT_EVAL_CSV,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_tokens: int = 64,
    device: str = "cpu",
    vocab_file: Optional[Path] = None,
) -> None:
    """
    Train/evaluate on real GLUE SST-2 tsv files.
    Expected files in data_dir: train.tsv, dev.tsv
    """
    data_dir = ensure_sst2_dataset(data_dir)
    train_tsv = data_dir / "train.tsv"
    dev_tsv = data_dir / "dev.tsv"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dev = torch.device(device)
    print(f"--- Loading biLM from {checkpoint_path} ---")
    bilm, num_layers, hidden_dim, options = load_pretrained_char_bilm_from_checkpoint(
        checkpoint_path, map_location=dev
    )
    bilm.to(dev)

    max_chars = int(options["char_cnn"]["max_characters_per_token"])
    n_characters = int(options["char_cnn"]["n_characters"])
    if vocab_file is None:
        print(
            "[warn] --vocab_file 미지정: fallback byte char encoding 사용 "
            "(정확 재현을 위해서는 pretrain vocab_file 권장)"
        )
    elif not vocab_file.is_file():
        raise FileNotFoundError(f"vocab_file not found: {vocab_file}")
    else:
        print(f"Using vocab_file: {vocab_file}")

    encoder = CharIdEncoder(
        max_chars_per_token=max_chars,
        n_characters=n_characters,
        vocab_file=vocab_file,
    )
    model = SSTClassifier(
        bilm=bilm,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        layer_mode="weighted",
        pooling="attention",
    ).to(dev)

    train_samples = read_sst2_tsv(train_tsv, has_labels=True)
    dev_samples = read_sst2_tsv(dev_tsv, has_labels=True)
    print(f"Loaded SST-2: train={len(train_samples)}, dev={len(dev_samples)}")

    train_ds = SST2CharDataset(train_samples, max_tokens=max_tokens)
    dev_ds = SST2CharDataset(dev_samples, max_tokens=max_tokens)
    collate_fn = make_sst2_collate_fn(encoder)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    optimizer = torch.optim.Adam(trainable_parameters(model), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("--- Real SST-2 training ---")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, dev)
        with torch.no_grad():
            dv_loss, dv_acc = run_epoch(model, dev_loader, None, criterion, dev)
        print(
            f"epoch {epoch:02d} | train loss={tr_loss:.4f} acc={tr_acc:.4f} "
            f"| dev loss={dv_loss:.4f} acc={dv_acc:.4f}"
        )
        append_epoch_metrics_csv(
            output_csv, epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, dev_loss=dv_loss, dev_acc=dv_acc
        )
    print(f"Epoch metrics appended to: {output_csv}")


if __name__ == "__main__":
    # Edit these values directly instead of using argparse.
    RUN_MODE = "train"  # "demo" or "train"
    DATA_DIR = Path(__file__).resolve().parent / "SST-2"
    CHECKPOINT_PATH = DEFAULT_BILM_CHECKPOINT
    OUTPUT_CSV = DEFAULT_EVAL_CSV
    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 1e-3
    MAX_TOKENS = 64
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VOCAB_FILE: Optional[Path] = DEFAULT_VOCAB_FILE

    if RUN_MODE == "demo":
        _demo()
    else:
        train_on_real_sst2(
            data_dir=DATA_DIR,
            checkpoint_path=CHECKPOINT_PATH,
            output_csv=OUTPUT_CSV,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            max_tokens=MAX_TOKENS,
            device=DEVICE,
            vocab_file=VOCAB_FILE,
        )
