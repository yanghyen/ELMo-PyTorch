#!/usr/bin/env python3
"""
CoNLL-2003 NER with:
  1) Baseline: [GloVe ; CharCNN] -> BiLSTM -> CRF
  2) ELMo:     [GloVe ; CharCNN ; ELMo] -> BiLSTM -> CRF

Example:
    python downstream/NER/conll2003_ner_bilstm_crf.py --model baseline
    python downstream/NER/conll2003_ner_bilstm_crf.py --model elmo \
      --bilm-checkpoint checkpoints/bilm/final_model.pt
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_conll2003() -> Dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "This script requires `datasets`. Install via `pip install datasets`."
        ) from exc
    return load_dataset("conll2003")


class WordVocab:
    def __init__(self) -> None:
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.itos = ["<pad>", "<unk>"]

    def build(self, token_lists: Sequence[Sequence[str]], min_freq: int = 1) -> None:
        freq: Dict[str, int] = {}
        for toks in token_lists:
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
        for tok, c in freq.items():
            if c >= min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(t, 1) for t in tokens]

    def __len__(self) -> int:
        return len(self.itos)


class CharVocab:
    def __init__(self) -> None:
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.itos = ["<pad>", "<unk>"]

    def build(self, token_lists: Sequence[Sequence[str]]) -> None:
        for toks in token_lists:
            for tok in toks:
                for ch in tok:
                    if ch not in self.stoi:
                        self.stoi[ch] = len(self.itos)
                        self.itos.append(ch)

    def encode_token(self, token: str, max_word_len: int) -> List[int]:
        ids = [self.stoi.get(ch, 1) for ch in token[:max_word_len]]
        if len(ids) < max_word_len:
            ids += [0] * (max_word_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.itos)


def load_glove_matrix(vocab: WordVocab, dim: int = 100) -> Tensor:
    emb = torch.empty(len(vocab), dim).uniform_(-0.05, 0.05)
    emb[0].zero_()
    try:
        from torchtext.vocab import GloVe

        glove = GloVe(name="6B", dim=dim)
        hit = 0
        for i, tok in enumerate(vocab.itos):
            if tok in glove.stoi:
                emb[i] = glove[tok]
                hit += 1
        print(f"GloVe loaded: {hit}/{len(vocab)} tokens covered")
    except Exception as exc:
        print(f"[WARN] Failed to load GloVe ({exc}). Using random word embeddings.")
    return emb


class NERDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


class CharIdEncoder:
    def __init__(self, max_chars_per_token: int, n_characters: int, vocab_file: Optional[Path]):
        self.max_chars_per_token = max_chars_per_token
        self.n_characters = n_characters
        self._vocab = None
        if vocab_file is not None and vocab_file.is_file():
            if str(Path.cwd()) not in sys.path:
                sys.path.insert(0, str(Path.cwd()))
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

    def encode_tokens_with_bos_eos(self, tokens: Sequence[str]) -> Tensor:
        if self._vocab is not None:
            arr = self._vocab.encode_chars(list(tokens), split=False)
            return torch.from_numpy(arr).long()
        bos_char, eos_char = 256, 257
        bos = torch.full((self.max_chars_per_token,), 260, dtype=torch.long)
        eos = torch.full((self.max_chars_per_token,), 260, dtype=torch.long)
        bos[0], bos[1], bos[2] = 258, bos_char, 259
        eos[0], eos[1], eos[2] = 258, eos_char, 259
        pieces = [bos]
        for t in tokens:
            pieces.append(self._fallback_word_to_char_ids(t))
        pieces.append(eos)
        return torch.stack(pieces, dim=0)


class CRF(nn.Module):
    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        log_numerator = self._score_sentence(emissions, tags, mask)
        log_denominator = self._log_partition(emissions, mask)
        return torch.mean(log_denominator - log_numerator)

    def _score_sentence(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        bsz, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0, :].gather(1, tags[:, 0:1]).squeeze(1)
        for t in range(1, seq_len):
            valid = mask[:, t]
            emit_t = emissions[:, t, :].gather(1, tags[:, t:t + 1]).squeeze(1)
            trans_t = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + (emit_t + trans_t) * valid
        lengths = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def _log_partition(self, emissions: Tensor, mask: Tensor) -> Tensor:
        score = self.start_transitions + emissions[:, 0, :]
        seq_len = emissions.size(1)
        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, t, :].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def decode(self, emissions: Tensor, mask: Tensor) -> List[List[int]]:
        bsz, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0, :]
        history: List[Tensor] = []
        for t in range(1, seq_len):
            next_score = score.unsqueeze(2) + self.transitions
            best_score, best_tag = next_score.max(dim=1)
            best_score = best_score + emissions[:, t, :]
            score = torch.where(mask[:, t].unsqueeze(1), best_score, score)
            history.append(best_tag)
        score = score + self.end_transitions
        best_last_score, best_last_tag = score.max(dim=1)
        _ = best_last_score
        seq_ends = mask.long().sum(dim=1) - 1

        paths: List[List[int]] = []
        for i in range(bsz):
            end = int(seq_ends[i].item())
            last_tag = int(best_last_tag[i].item())
            path = [last_tag]
            for hist_t in reversed(history[:end]):
                last_tag = int(hist_t[i][last_tag].item())
                path.append(last_tag)
            path.reverse()
            paths.append(path)
        return paths


class CharCNNEncoder(nn.Module):
    def __init__(self, char_vocab_size: int, char_emb_dim: int, out_dim: int, kernel_sizes: Tuple[int, ...] = (3, 4, 5), num_filters: int = 50):
        super().__init__()
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(char_emb_dim, num_filters, k, padding=k // 2) for k in kernel_sizes]
        )
        self.proj = nn.Linear(num_filters * len(kernel_sizes), out_dim)

    def forward(self, char_ids: Tensor) -> Tensor:
        bsz, seq_len, word_len = char_ids.shape
        x = self.char_emb(char_ids).view(bsz * seq_len, word_len, -1).transpose(1, 2)
        conv_outs = [F.relu(conv(x)).max(dim=-1).values for conv in self.convs]
        cat = torch.cat(conv_outs, dim=-1)
        out = self.proj(cat)
        return out.view(bsz, seq_len, -1)


class ELMoEmbedding(nn.Module):
    def __init__(self, bilm: nn.Module, num_layers: int):
        super().__init__()
        self.bilm = bilm
        self.num_layers = num_layers
        for p in self.bilm.parameters():
            p.requires_grad = False
        self.bilm.eval()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, char_ids_with_bos_eos: Tensor) -> Tensor:
        with torch.no_grad():
            out = self.bilm(char_ids_with_bos_eos)
        if isinstance(out, dict):
            layers = [out["lm_embeddings"][:, i] for i in range(out["lm_embeddings"].size(1))]
        else:
            layers = out
        weights = F.softmax(self.layer_logits, dim=0)
        mixed = sum(weights[i] * layers[i] for i in range(self.num_layers))
        return self.gamma * mixed


class NERTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        char_vocab_size: int,
        num_tags: int,
        word_emb_matrix: Tensor,
        hidden_dim: int = 256,
        char_emb_dim: int = 30,
        char_out_dim: int = 128,
        dropout: float = 0.2,
        elmo_module: Optional[ELMoEmbedding] = None,
        elmo_dim: int = 0,
    ) -> None:
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, word_emb_matrix.size(1), padding_idx=0)
        self.word_emb.weight.data.copy_(word_emb_matrix)
        self.char_encoder = CharCNNEncoder(
            char_vocab_size=char_vocab_size,
            char_emb_dim=char_emb_dim,
            out_dim=char_out_dim,
        )
        self.elmo_module = elmo_module
        enc_in = word_emb_matrix.size(1) + char_out_dim + (elmo_dim if elmo_module is not None else 0)
        self.encoder = nn.LSTM(
            input_size=enc_in,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags)

    def emissions(
        self,
        word_ids: Tensor,
        char_ids: Tensor,
        elmo_char_ids: Optional[Tensor] = None,
    ) -> Tensor:
        word = self.word_emb(word_ids)
        ch = self.char_encoder(char_ids)
        feats = [word, ch]
        if self.elmo_module is not None:
            if elmo_char_ids is None:
                raise ValueError("ELMo mode requires elmo_char_ids in batch")
            elmo_all = self.elmo_module(elmo_char_ids)
            elmo_tok = elmo_all[:, 1:-1, :]
            feats.append(elmo_tok)
        x = self.dropout(torch.cat(feats, dim=-1))
        enc, _ = self.encoder(x)
        enc = self.dropout(enc)
        return self.classifier(enc)

    def loss(
        self,
        word_ids: Tensor,
        char_ids: Tensor,
        tags: Tensor,
        mask: Tensor,
        elmo_char_ids: Optional[Tensor] = None,
    ) -> Tensor:
        emissions = self.emissions(word_ids, char_ids, elmo_char_ids=elmo_char_ids)
        return self.crf(emissions, tags, mask)

    def decode(
        self,
        word_ids: Tensor,
        char_ids: Tensor,
        mask: Tensor,
        elmo_char_ids: Optional[Tensor] = None,
    ) -> List[List[int]]:
        emissions = self.emissions(word_ids, char_ids, elmo_char_ids=elmo_char_ids)
        return self.crf.decode(emissions, mask)


def spans_from_bio(tag_names: Sequence[str]) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
    start = -1
    ent_type = ""
    for i, t in enumerate(tag_names):
        if t.startswith("B-"):
            if start != -1:
                spans.append((ent_type, start, i - 1))
            ent_type = t[2:]
            start = i
        elif t.startswith("I-") and start != -1 and t[2:] == ent_type:
            continue
        else:
            if start != -1:
                spans.append((ent_type, start, i - 1))
                start = -1
                ent_type = ""
    if start != -1:
        spans.append((ent_type, start, len(tag_names) - 1))
    return spans


def ner_f1(golds: List[List[int]], preds: List[List[int]], id2tag: Sequence[str]) -> Tuple[float, float, float]:
    tp = 0
    fp = 0
    fn = 0
    for g, p in zip(golds, preds):
        g_sp = set(spans_from_bio([id2tag[i] for i in g]))
        p_sp = set(spans_from_bio([id2tag[i] for i in p]))
        tp += len(g_sp & p_sp)
        fp += len(p_sp - g_sp)
        fn += len(g_sp - p_sp)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return prec, rec, f1


def make_collate_fn(
    word_vocab: WordVocab,
    char_vocab: CharVocab,
    max_word_len: int,
    elmo_encoder: Optional[CharIdEncoder],
):
    def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Tensor]:
        max_len = max(len(x["tokens"]) for x in batch)
        bsz = len(batch)
        word_ids = torch.zeros(bsz, max_len, dtype=torch.long)
        char_ids = torch.zeros(bsz, max_len, max_word_len, dtype=torch.long)
        tags = torch.zeros(bsz, max_len, dtype=torch.long)
        mask = torch.zeros(bsz, max_len, dtype=torch.bool)
        elmo_ids = None
        if elmo_encoder is not None:
            max_len_elmo = max_len + 2
            elmo_ids = torch.zeros(
                bsz, max_len_elmo, elmo_encoder.max_chars_per_token, dtype=torch.long
            )

        for i, row in enumerate(batch):
            toks = row["tokens"]
            lab = row["ner_tags"]
            L = len(toks)
            mask[i, :L] = True
            word_ids[i, :L] = torch.tensor(word_vocab.encode(toks), dtype=torch.long)
            tags[i, :L] = torch.tensor(lab, dtype=torch.long)
            chars = [char_vocab.encode_token(t, max_word_len) for t in toks]
            char_ids[i, :L] = torch.tensor(chars, dtype=torch.long)
            if elmo_encoder is not None and elmo_ids is not None:
                e = elmo_encoder.encode_tokens_with_bos_eos(toks)
                elmo_ids[i, : e.size(0)] = e

        out = {"word_ids": word_ids, "char_ids": char_ids, "tags": tags, "mask": mask}
        if elmo_ids is not None:
            out["elmo_char_ids"] = elmo_ids
        return out

    return collate


def run_epoch(
    model: NERTagger,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        word_ids = batch["word_ids"].to(device)
        char_ids = batch["char_ids"].to(device)
        tags = batch["tags"].to(device)
        mask = batch["mask"].to(device)
        elmo_ids = batch.get("elmo_char_ids")
        if elmo_ids is not None:
            elmo_ids = elmo_ids.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        loss = model.loss(word_ids, char_ids, tags, mask, elmo_char_ids=elmo_ids)
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        bs = word_ids.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        total_count += bs
    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(
    model: NERTagger,
    loader: DataLoader,
    id2tag: Sequence[str],
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    gold_all: List[List[int]] = []
    pred_all: List[List[int]] = []
    for batch in loader:
        word_ids = batch["word_ids"].to(device)
        char_ids = batch["char_ids"].to(device)
        tags = batch["tags"].to(device)
        mask = batch["mask"].to(device)
        elmo_ids = batch.get("elmo_char_ids")
        if elmo_ids is not None:
            elmo_ids = elmo_ids.to(device)

        pred_paths = model.decode(word_ids, char_ids, mask, elmo_char_ids=elmo_ids)
        lengths = mask.long().sum(dim=1).tolist()
        for i, L in enumerate(lengths):
            g = tags[i, :L].tolist()
            p = pred_paths[i][:L]
            gold_all.append(g)
            pred_all.append(p)
    return ner_f1(gold_all, pred_all, id2tag)


def load_bilm_for_elmo(checkpoint_path: Path) -> Tuple[ELMoEmbedding, int, int, Dict[str, Any]]:
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    from bilm.src.simple_language_model import SimpleLanguageModel

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    options = ckpt["options"]
    state = ckpt["model_state_dict"]
    vocab_size = int(state["output_projection.weight"].shape[0])

    core = SimpleLanguageModel(options, vocab_size)
    core.load_state_dict(state, strict=True)
    core.eval()

    with torch.no_grad():
        max_c = int(options["char_cnn"]["max_characters_per_token"])
        probe = torch.zeros(1, 3, max_c, dtype=torch.long)
        out = core(probe)
        if isinstance(out, dict):
            n_layers = int(out["lm_embeddings"].size(1))
            emb_dim = int(out["lm_embeddings"].size(-1))
        else:
            n_layers = len(out)
            emb_dim = int(out[0].size(-1))
    return ELMoEmbedding(core, num_layers=n_layers), n_layers, emb_dim, options


def append_epoch_metrics_csv(
    csv_path: Path,
    model_name: str,
    seed: int,
    epoch: int,
    train_loss: float,
    dev_p: float,
    dev_r: float,
    dev_f1: float,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(
                [
                    "model",
                    "seed",
                    "epoch",
                    "train_loss",
                    "dev_precision",
                    "dev_recall",
                    "dev_f1",
                ]
            )
        w.writerow([model_name, seed, epoch, train_loss, dev_p, dev_r, dev_f1])


def append_final_metrics_csv(
    csv_path: Path,
    model_name: str,
    seed: int,
    best_dev_f1: float,
    test_p: float,
    test_r: float,
    test_f1: float,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(
                [
                    "model",
                    "seed",
                    "best_dev_f1",
                    "test_precision",
                    "test_recall",
                    "test_f1",
                ]
            )
        w.writerow([model_name, seed, best_dev_f1, test_p, test_r, test_f1])


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "elmo"], default="baseline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--char-out-dim", type=int, default=128)
    parser.add_argument("--char-emb-dim", type=int, default=30)
    parser.add_argument("--max-word-len", type=int, default=20)
    parser.add_argument("--glove-dim", type=int, default=100)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[13, 17, 23, 29, 37],
        help="반복 실행할 seed 목록 (기본 5개)",
    )
    parser.add_argument(
        "--bilm-checkpoint",
        type=Path,
        default=Path("checkpoints/bilm/final_model.pt"),
    )
    parser.add_argument(
        "--elmo-vocab-file",
        type=Path,
        default=Path("bilm/data/pretrain/elmo/vocab.txt"),
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="epoch별 dev 성능을 append할 CSV 경로 (기본: 모델별 파일명)",
    )
    parser.add_argument(
        "--final-metrics-csv",
        type=Path,
        default=script_dir / "final_eval_metrics.csv",
        help="run별 최종 test 성능 CSV 경로",
    )
    args = parser.parse_args()
    if args.metrics_csv is None:
        args.metrics_csv = script_dir / f"{args.model}_eval_metrics.csv"

    ds = ensure_conll2003()
    train_rows = ds["train"]
    dev_rows = ds["validation"]
    test_rows = ds["test"]
    id2tag = ds["train"].features["ner_tags"].feature.names
    num_tags = len(id2tag)

    train_tokens = [r["tokens"] for r in train_rows]
    word_vocab = WordVocab()
    word_vocab.build(train_tokens, min_freq=1)
    char_vocab = CharVocab()
    char_vocab.build(train_tokens)
    word_emb_matrix = load_glove_matrix(word_vocab, dim=args.glove_dim)

    elmo_module = None
    elmo_dim = 0
    elmo_encoder = None
    if args.model == "elmo":
        if not args.bilm_checkpoint.is_file():
            raise FileNotFoundError(
                f"ELMo checkpoint not found: {args.bilm_checkpoint}. "
                "Train or place the biLM checkpoint first."
            )
        elmo_module, n_layers, elmo_dim, bilm_opt = load_bilm_for_elmo(args.bilm_checkpoint)
        max_chars = int(bilm_opt["char_cnn"]["max_characters_per_token"])
        n_characters = int(bilm_opt["char_cnn"]["n_characters"])
        elmo_encoder = CharIdEncoder(max_chars, n_characters, args.elmo_vocab_file)
        print(f"Using ELMo ({n_layers} layers, dim={elmo_dim})")

    collate = make_collate_fn(word_vocab, char_vocab, args.max_word_len, elmo_encoder)
    train_loader = DataLoader(
        NERDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        NERDataset(dev_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        NERDataset(test_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        print(f"\n=== Running seed {seed} ({args.model}) ===")
        set_seed(seed)

        model = NERTagger(
            vocab_size=len(word_vocab),
            char_vocab_size=len(char_vocab),
            num_tags=num_tags,
            word_emb_matrix=word_emb_matrix,
            hidden_dim=args.hidden_dim,
            char_emb_dim=args.char_emb_dim,
            char_out_dim=args.char_out_dim,
            elmo_module=elmo_module,
            elmo_dim=elmo_dim,
        )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_dev_f1 = -1.0
        best_state = None
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, optimizer, device)
            dev_p, dev_r, dev_f1 = evaluate(model, dev_loader, id2tag, device)
            append_epoch_metrics_csv(
                csv_path=args.metrics_csv,
                model_name=args.model,
                seed=seed,
                epoch=epoch,
                train_loss=train_loss,
                dev_p=dev_p,
                dev_r=dev_r,
                dev_f1=dev_f1,
            )
            print(
                f"[{args.model}][seed={seed}] epoch={epoch} train_loss={train_loss:.4f} "
                f"dev_p={dev_p:.4f} dev_r={dev_r:.4f} dev_f1={dev_f1:.4f}"
            )
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        test_p, test_r, test_f1 = evaluate(model, test_loader, id2tag, device)
        append_final_metrics_csv(
            csv_path=args.final_metrics_csv,
            model_name=args.model,
            seed=seed,
            best_dev_f1=best_dev_f1,
            test_p=test_p,
            test_r=test_r,
            test_f1=test_f1,
        )
        print(
            f"[{args.model}][seed={seed}] TEST: "
            f"precision={test_p:.4f} recall={test_r:.4f} f1={test_f1:.4f}"
        )


if __name__ == "__main__":
    main()
