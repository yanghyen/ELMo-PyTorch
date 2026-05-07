#!/usr/bin/env python3
"""
SST-5 sentiment classification with a frozen pretrained biLM + ELMo-style layer mix.

Loads the PyTorch biLM checkpoint from bilm-tf training (``SimpleLanguageModel``).
No dependency on ``downstream/SST_2/sst2_elmo_classifier.py``.

Expected dataset format (TSV):
    - train.tsv / dev.tsv under downstream/SST_5/SST-5
    - columns: sentence(or text) + label
    - label range: 0..4
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Tokenization & pooling (biLM / ELMo classifier path)
# ---------------------------------------------------------------------------

PoolingMode = Literal["attention", "mean", "max"]
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def token_mask_from_char_ids(char_ids: Tensor) -> Tensor:
    """(batch, seq_len, char_len) -> (batch, seq_len) float {0,1}."""
    return (char_ids.sum(dim=-1) > 0).to(dtype=torch.float32, device=char_ids.device)


def pool_sequence(
    h: Tensor,
    mask: Tensor,
    mode: PoolingMode,
    attn_linear: Optional[nn.Module],
) -> Tensor:
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
        scores = attn_linear(h)
        if scores.dim() == 2:
            scores = scores.unsqueeze(-1)
        if scores.dim() != 3:
            raise ValueError("attention scorer must produce (B, T) or (B, T, H_attn)")
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=1)
        attn = torch.nan_to_num(attn, nan=0.0)
        pooled_heads = (attn.unsqueeze(-1) * h.unsqueeze(2)).sum(dim=1)
        return pooled_heads.mean(dim=1)

    raise ValueError(f"Unknown pooling mode: {mode}")


def tokenize_text(sentence: str) -> List[str]:
    return TOKEN_PATTERN.findall(sentence)


# ---------------------------------------------------------------------------
# Repo paths + checkpoint loading (bilm-tf SimpleLanguageModel)
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_BILM_TF_ROOT = next(
    (p for p in _THIS_FILE.parents if p.name == "bilm-tf"),
    _THIS_FILE.parents[2],
)
_REPO_ROOT = _BILM_TF_ROOT.parent
DEFAULT_BILM_CHECKPOINT = (
    _REPO_ROOT / "bilm-tf" / "checkpoints" / "bilm" / "final_model.pt"
)
DEFAULT_GLOVE_PATH = (
    _REPO_ROOT / "bilm-tf" / "downstream" / "SQuAD" / ".glove_cache" / "glove.6B.300d.txt"
)
DEFAULT_VOCAB_FILE = (
    _BILM_TF_ROOT / "bilm" / "data" / "pretrain" / "elmo" / "vocab.txt"
)


def _ensure_bilm_tf_on_path() -> None:
    root = str(_BILM_TF_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


class CharBiLMStackAdapter(nn.Module):
    """``SimpleLanguageModel`` dict output -> ``forward(char_ids) -> List[Tensor]``."""

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
# ELMo-style layer mix + SST classifier
# ---------------------------------------------------------------------------

LayerMode = Union[Literal["weighted"], int]


class ELMoEmbedding(nn.Module):
    """Frozen biLM + trainable softmax over layers + gamma."""

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
        # Keep biLM in eval mode even when parent model is set to train().
        self.bilm.eval()
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


class SSTClassifier(nn.Module):
    """char_ids -> ELMo mix -> pool -> MLP -> logits (default 2 classes; SST-5 replaces last layer)."""

    def __init__(
        self,
        bilm: nn.Module,
        hidden_dim: int,
        num_layers: int,
        vocab_size: int,
        glove_dim: int = 300,
        glove_weight: Optional[Tensor] = None,
        layer_mode: LayerMode = "weighted",
        pooling: PoolingMode = "attention",
        classifier_dropout: float = 0.2,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.pooling: PoolingMode = pooling
        self.elmo = ELMoEmbedding(bilm, num_layers=num_layers, layer_mode=layer_mode)
        self.word_embed = nn.Embedding(vocab_size, glove_dim, padding_idx=0)
        if glove_weight is not None:
            self.word_embed.weight.data.copy_(glove_weight)
        self.fuse = nn.Linear(hidden_dim + glove_dim, hidden_dim)
        self.attn_scorer: Optional[nn.Linear] = (
            nn.Linear(hidden_dim, attn_heads) if pooling == "attention" else None
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        word_ids: Tensor,
        char_ids: Tensor,
        token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        elmo_h = self.elmo(char_ids)
        glove_h = self.word_embed(word_ids)
        h = self.fuse(torch.cat([glove_h, elmo_h], dim=-1))
        if token_mask is None:
            # Exclude padding and special BOS/EOS from sentence pooling.
            bos_id = 2
            eos_id = 3
            mask = ((word_ids != 0) & (word_ids != bos_id) & (word_ids != eos_id)).to(
                dtype=h.dtype, device=h.device
            )
        else:
            mask = token_mask.to(dtype=h.dtype, device=h.device)
        pooled = pool_sequence(h, mask, self.pooling, self.attn_scorer)
        return self.classifier(pooled)


class WordVocab:
    def __init__(self) -> None:
        self.token_to_id = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.id_to_token = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def build(self, token_lists: List[List[str]]) -> None:
        for tokens in token_lists:
            for tok in tokens:
                if tok not in self.token_to_id:
                    self.token_to_id[tok] = len(self.id_to_token)
                    self.id_to_token.append(tok)

    def encode_with_special_tokens(self, tokens: List[str]) -> List[int]:
        ids = [self.token_to_id["<bos>"]]
        ids.extend(self.token_to_id.get(t, 1) for t in tokens)
        ids.append(self.token_to_id["<eos>"])
        return ids

    def __len__(self) -> int:
        return len(self.id_to_token)


def load_glove_embeddings(glove_path: Path, vocab: WordVocab, embed_dim: int) -> Tensor:
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


class CharIdEncoder:
    """Character ids for words; prefers ``UnicodeCharsVocabulary`` when ``vocab_file`` is set."""

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
            arr = self._vocab.encode_chars(toks, split=False)
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


def trainable_parameters(module: nn.Module) -> List[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


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

    for batch_word_ids, batch_char_ids, batch_y in loader:
        batch_word_ids = batch_word_ids.to(device)
        batch_char_ids = batch_char_ids.to(device)
        batch_y = batch_y.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        logits = model(batch_word_ids, batch_char_ids)
        loss = criterion(logits, batch_y)

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm=1.0)
            optimizer.step()

        bs = int(batch_y.size(0))
        total_loss += float(loss.detach().cpu()) * bs
        total_correct += float((logits.argmax(dim=-1) == batch_y).float().sum().cpu())
        total_count += bs

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


# ---------------------------------------------------------------------------
# SST-5 dataset & training
# ---------------------------------------------------------------------------

NUM_LABELS = 5


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
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("sentence\tlabel\n")
            for row in split:
                sentence = str(row.get("text", "")).strip()
                if not sentence:
                    continue
                label = int(row["label"])
                if not 0 <= label < NUM_LABELS:
                    raise ValueError(f"Invalid label {label} in split {split_name}")
                sentence = sentence.replace("\t", " ").replace("\n", " ")
                f.write(f"{sentence}\t{label}\n")

    _write_split_tsv("train", train_tsv)
    _write_split_tsv("validation", dev_tsv)
    _write_split_tsv("test", test_tsv)
    print(f"Prepared SST-5 at: {data_dir}")
    return data_dir


def read_sst5_tsv(path: Path) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines:
        return rows
    header = lines[0].split("\t")
    try:
        sent_idx = header.index("sentence")
    except ValueError:
        sent_idx = header.index("text") if "text" in header else 0
    label_idx = header.index("label")
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= max(sent_idx, label_idx):
            continue
        sentence = parts[sent_idx].strip()
        if not sentence:
            continue
        label = int(parts[label_idx])
        if not 0 <= label < NUM_LABELS:
            raise ValueError(f"Invalid SST-5 label {label} in {path}")
        rows.append((sentence, label))
    return rows


class SST5CharDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        max_tokens: int,
    ) -> None:
        self.samples = samples
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sentence, label = self.samples[idx]
        tokens = tokenize_text(sentence)[: self.max_tokens]
        return tokens, torch.tensor(label, dtype=torch.long)


def make_sst5_collate_fn(encoder: CharIdEncoder, word_vocab: WordVocab):
    def _collate(batch):
        tokens_list = [b[0] for b in batch]
        labels = torch.stack([b[1] for b in batch], dim=0)
        encoded_char = [encoder.encode_tokens(toks) for toks in tokens_list]
        encoded_word = [word_vocab.encode_with_special_tokens(toks) for toks in tokens_list]
        max_len = max(int(x.size(0)) for x in encoded_char)
        char_len = int(encoded_char[0].size(1))
        char_out = torch.zeros(len(encoded_char), max_len, char_len, dtype=torch.long)
        word_out = torch.zeros(len(encoded_word), max_len, dtype=torch.long)

        for i, x in enumerate(encoded_char):
            char_out[i, : x.size(0), :] = x
        for i, x in enumerate(encoded_word):
            L = min(len(x), max_len)
            word_out[i, :L] = torch.tensor(x[:L], dtype=torch.long)

        return word_out, char_out, labels

    return _collate


def build_sst5_model(
    bilm: nn.Module,
    hidden_dim: int,
    num_layers: int,
    vocab_size: int,
    glove_weight: Optional[Tensor],
    glove_dim: int,
) -> SSTClassifier:
    model = SSTClassifier(
        bilm=bilm,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        vocab_size=vocab_size,
        glove_dim=glove_dim,
        glove_weight=glove_weight,
        layer_mode="weighted",
        pooling="attention",
    )
    model.classifier[-1] = nn.Linear(hidden_dim, NUM_LABELS)
    return model


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_on_real_sst5(
    data_dir: Path,
    checkpoint_path: Path = DEFAULT_BILM_CHECKPOINT,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 3e-4,
    max_tokens: int = 64,
    device: str = "cpu",
    vocab_file: Optional[Path] = DEFAULT_VOCAB_FILE,
    seeds: Tuple[int, ...] = (17, 21, 23),
    early_stopping_patience: int = 3,
    glove_path: Path = DEFAULT_GLOVE_PATH,
    glove_dim: int = 300,
) -> None:
    data_dir = ensure_sst5_dataset(data_dir)
    train_tsv = data_dir / "train.tsv"
    dev_tsv = data_dir / "dev.tsv"
    test_tsv = data_dir / "test.tsv"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dev = torch.device(device)

    print(f"--- Loading biLM options from {checkpoint_path} ---")
    _, _, _, options = load_pretrained_char_bilm_from_checkpoint(
        checkpoint_path, map_location=dev
    )
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
    train_samples = read_sst5_tsv(train_tsv)
    dev_samples = read_sst5_tsv(dev_tsv)
    test_samples = read_sst5_tsv(test_tsv)
    print(
        f"Loaded SST-5: train={len(train_samples)}, dev={len(dev_samples)}, test={len(test_samples)}"
    )
    word_vocab = WordVocab()
    word_vocab.build([tokenize_text(s)[:max_tokens] for s, _ in train_samples])
    glove_weight: Optional[Tensor] = None
    if glove_path.is_file():
        glove_weight = load_glove_embeddings(glove_path, word_vocab, glove_dim)
    else:
        print(f"[warn] GloVe file not found: {glove_path}. Using random init.")

    train_ds = SST5CharDataset(train_samples, max_tokens=max_tokens)
    dev_ds = SST5CharDataset(dev_samples, max_tokens=max_tokens)
    test_ds = SST5CharDataset(test_samples, max_tokens=max_tokens)
    collate_fn = make_sst5_collate_fn(encoder, word_vocab)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    print("--- Real SST-5 training (multi-seed + early stopping) ---")
    final_test_accs: List[float] = []
    for seed in seeds:
        set_global_seed(seed)
        print(f"\n=== Seed {seed} ===")

        print(f"--- Loading biLM from {checkpoint_path} ---")
        bilm, num_layers, hidden_dim, _ = load_pretrained_char_bilm_from_checkpoint(
            checkpoint_path, map_location=dev
        )
        bilm.to(dev)
        model = build_sst5_model(
            bilm=bilm,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            vocab_size=len(word_vocab),
            glove_weight=glove_weight,
            glove_dim=glove_dim,
        ).to(dev)
        optimizer = torch.optim.AdamW(trainable_parameters(model), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        best_dev_loss = float("inf")
        best_dev_acc = 0.0
        best_epoch = 0
        patience_count = 0
        best_state_dict: Optional[Dict[str, Tensor]] = None

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, dev)
            with torch.no_grad():
                dv_loss, dv_acc = run_epoch(model, dev_loader, None, criterion, dev)
            print(
                f"seed {seed} | epoch {epoch:02d} | "
                f"train loss={tr_loss:.4f} acc={tr_acc:.4f} "
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
                best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                patience_count += 1
                if patience_count >= early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch} (seed={seed}, "
                        f"best_epoch={best_epoch}, best_dev_loss={best_dev_loss:.4f})"
                    )
                    break

        if best_state_dict is None:
            raise RuntimeError(f"No best model captured for seed {seed}")
        model.load_state_dict(best_state_dict)
        with torch.no_grad():
            ts_loss, ts_acc = run_epoch(model, test_loader, None, criterion, dev)
        final_test_accs.append(ts_acc)
        print(
            f"seed {seed} | best-dev epoch={best_epoch} "
            f"| test loss={ts_loss:.4f} acc={ts_acc:.4f}"
        )
        print(
            f"Seed {seed} done | best_epoch={best_epoch} "
            f"best_dev_loss={best_dev_loss:.4f} best_dev_acc={best_dev_acc:.4f}"
        )
    if final_test_accs:
        mean_test_acc = sum(final_test_accs) / len(final_test_accs)
        print(
            f"Test acc from best-dev checkpoints over seeds {seeds}: "
            f"mean={mean_test_acc:.4f}, values={final_test_accs}"
        )
    print("File saving disabled: metrics/checkpoints are not written.")


if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent / "SST-5"
    CHECKPOINT_PATH = DEFAULT_BILM_CHECKPOINT
    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 3e-4
    MAX_TOKENS = 64
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VOCAB_FILE: Optional[Path] = DEFAULT_VOCAB_FILE
    SEEDS = (10, 11, 12, 13, 14)
    EARLY_STOPPING_PATIENCE = 3
    GLOVE_PATH = DEFAULT_GLOVE_PATH
    GLOVE_DIM = 300

    train_on_real_sst5(
        data_dir=DATA_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        max_tokens=MAX_TOKENS,
        device=DEVICE,
        vocab_file=VOCAB_FILE,
        seeds=SEEDS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        glove_path=GLOVE_PATH,
        glove_dim=GLOVE_DIM,
    )
