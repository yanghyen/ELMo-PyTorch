#!/usr/bin/env python3
"""
Deep Semantic Role Labeling (SRL)

Baseline:
  GloVe + predicate-indicator -> stacked BiLSTM -> softmax tagging

ELMo variant:
  [GloVe ; ELMo] + predicate-indicator -> stacked BiLSTM -> softmax tagging

Default seeds are fixed to 3 runs: [13, 17, 23].
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

_THIS_FILE = Path(__file__).resolve()
_BILM_TF_ROOT = next((p for p in _THIS_FILE.parents if p.name == "bilm-tf"), _THIS_FILE.parents[2])
if str(_BILM_TF_ROOT) not in sys.path:
    sys.path.insert(0, str(_BILM_TF_ROOT))

from downstream.SST_2.sst2_elmo_classifier import (  # noqa: E402
    CharIdEncoder,
    ELMoEmbedding,
    load_pretrained_char_bilm_from_checkpoint,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


@dataclass
class SRLRow:
    tokens: List[str]
    pred_ind: List[int]
    tags: List[str]


class SRLDataset(Dataset):
    def __init__(self, rows: Sequence[SRLRow]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> SRLRow:
        return self.rows[idx]


def _extract_frames_from_example(example: Dict[str, Any]) -> List[Tuple[List[str], List[int], List[str]]]:
    tokens = (
        example.get("words")
        or example.get("tokens")
        or example.get("sentence")
        or []
    )
    if not tokens:
        return []

    outputs: List[Tuple[List[str], List[int], List[str]]] = []
    if "srl_frames" in example and isinstance(example["srl_frames"], list):
        for fr in example["srl_frames"]:
            tags = []
            if isinstance(fr, dict):
                tags = fr.get("tags", []) or fr.get("frames", [])
            if not tags or len(tags) != len(tokens):
                continue
            verb_idx = int(fr.get("verb_index", -1)) if isinstance(fr, dict) else -1
            if verb_idx < 0:
                # OntoNotes SRL often marks predicate position directly in frame labels.
                for i, tag in enumerate(tags):
                    if tag == "B-V" or tag == "I-V" or tag == "V":
                        verb_idx = i
                        break
            if verb_idx < 0 and isinstance(fr, dict):
                verb_token = str(fr.get("verb", ""))
                if verb_token:
                    for i, tok in enumerate(tokens):
                        if tok == verb_token:
                            verb_idx = i
                            break
            pred_ind = [1 if i == verb_idx else 0 for i in range(len(tokens))]
            outputs.append((list(tokens), pred_ind, list(tags)))

    elif "verb" in example and "srl_tags" in example:
        tags = example.get("srl_tags", [])
        if tags and len(tags) == len(tokens):
            verb_token = str(example.get("verb", ""))
            pred_ind = [1 if t == verb_token else 0 for t in tokens]
            if sum(pred_ind) == 0 and "verb_index" in example:
                idx = int(example["verb_index"])
                if 0 <= idx < len(tokens):
                    pred_ind[idx] = 1
            outputs.append((list(tokens), pred_ind, list(tags)))

    return outputs


def load_conll2005_srl(
    dataset_name: str = "conll2005",
    dataset_config: Optional[str] = "wsj",
) -> Tuple[List[SRLRow], List[SRLRow], List[SRLRow]]:
    try:
        from datasets import load_dataset
        from datasets.exceptions import DatasetNotFoundError
    except ImportError as exc:
        raise ImportError("Install datasets first: pip install datasets") from exc

    load_candidates: List[Tuple[str, Optional[str]]] = [(dataset_name, dataset_config)]
    if (dataset_name, dataset_config) != ("ontonotes/conll2012_ontonotesv5", "english_v12"):
        load_candidates.append(("ontonotes/conll2012_ontonotesv5", "english_v12"))

    ds = None
    errors: List[str] = []
    for cand_name, cand_config in load_candidates:
        try:
            if cand_config is None:
                ds = load_dataset(cand_name)
            else:
                ds = load_dataset(cand_name, cand_config)
            print(f"Loaded SRL dataset: {cand_name}" + (f" ({cand_config})" if cand_config else ""))
            break
        except DatasetNotFoundError as exc:
            errors.append(f"{cand_name} ({cand_config}): {exc}")
        except Exception as exc:  # pragma: no cover - depends on remote HF state
            errors.append(f"{cand_name} ({cand_config}): {exc}")

    if ds is None:
        details = "\n".join(f"- {msg}" for msg in errors)
        raise RuntimeError(
            "Failed to load an SRL dataset from Hugging Face Hub.\n"
            "Try one of the following:\n"
            "1) update datasets: pip install -U datasets\n"
            "2) pass --srl-dataset and --srl-config to an available dataset\n"
            f"Attempted:\n{details}"
        )

    train_rows: List[SRLRow] = []
    dev_rows: List[SRLRow] = []
    test_rows: List[SRLRow] = []

    for split_name, sink in [("train", train_rows), ("validation", dev_rows), ("test", test_rows)]:
        split = ds[split_name]
        for ex in split:
            if "sentences" in ex and isinstance(ex["sentences"], list):
                for sent in ex["sentences"]:
                    if not isinstance(sent, dict):
                        continue
                    for toks, pred_ind, tags in _extract_frames_from_example(sent):
                        sink.append(SRLRow(tokens=toks, pred_ind=pred_ind, tags=tags))
            else:
                for toks, pred_ind, tags in _extract_frames_from_example(ex):
                    sink.append(SRLRow(tokens=toks, pred_ind=pred_ind, tags=tags))

    if not train_rows or not dev_rows or not test_rows:
        raise RuntimeError("Failed to parse SRL rows from CoNLL-2005 dataset.")
    return train_rows, dev_rows, test_rows


def build_tag_vocab(rows: Sequence[SRLRow]) -> Tuple[Dict[str, int], List[str]]:
    tags = sorted({t for r in rows for t in r.tags})
    tag2id = {t: i for i, t in enumerate(tags)}
    return tag2id, tags


def load_glove_matrix(vocab: WordVocab, dim: int = 100) -> Tensor:
    candidates: List[Path] = [
        _BILM_TF_ROOT / "bilm" / "data" / "pretrain" / "glove" / f"glove.6B.{dim}d.txt",
        _BILM_TF_ROOT / "data" / "glove" / f"glove.6B.{dim}d.txt",
        _BILM_TF_ROOT / "glove.6B" / f"glove.6B.{dim}d.txt",
        _BILM_TF_ROOT / f"glove.6B.{dim}d.txt",
        Path.home() / ".vector_cache" / f"glove.6B.{dim}d.txt",
        Path("glove.6B") / f"glove.6B.{dim}d.txt",
        Path(f"glove.6B.{dim}d.txt"),
    ]
    search_roots: List[Path] = [
        _BILM_TF_ROOT,
        _BILM_TF_ROOT / "downstream",
        Path.home() / ".vector_cache",
    ]
    for root in search_roots:
        if root.is_dir():
            candidates.extend(sorted(root.rglob("glove.6B.*d.txt")))

    glove_path = next((p for p in candidates if p.is_file()), None)
    if glove_path is None:
        emb = torch.empty(len(vocab), dim).uniform_(-0.05, 0.05)
        emb[0].zero_()
        print(
            "[WARN] GloVe file not found. "
            "Place glove.6B.<dim>d.txt in one of the default paths. Using random embeddings."
        )
        return emb

    actual_dim = dim
    try:
        with open(glove_path, "r", encoding="utf-8") as probe_f:
            for line in probe_f:
                parts = line.rstrip().split(" ")
                if len(parts) > 2:
                    actual_dim = len(parts) - 1
                    break
    except Exception:
        pass
    if actual_dim != dim:
        print(f"[INFO] Requested GloVe dim={dim} but found dim={actual_dim} at {glove_path}. Using found dim.")

    emb = torch.empty(len(vocab), actual_dim).uniform_(-0.05, 0.05)
    emb[0].zero_()
    vocab_lookup = {tok: i for i, tok in enumerate(vocab.itos)}
    hit = 0
    try:
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) <= actual_dim:
                    continue
                token = parts[0]
                vec_str = parts[1:]
                if len(vec_str) != actual_dim:
                    continue
                idx = vocab_lookup.get(token)
                if idx is None:
                    continue
                emb[idx] = torch.tensor([float(x) for x in vec_str], dtype=emb.dtype)
                hit += 1
        print(f"GloVe loaded from {glove_path}: {hit}/{len(vocab)} tokens covered")
    except Exception as exc:
        print(f"[WARN] Failed to parse GloVe file ({exc}). Using random embeddings.")
    return emb


class DeepSRLTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        word_emb_matrix: Tensor,
        pred_emb_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3,
        elmo_module: Optional[ELMoEmbedding] = None,
        elmo_dim: int = 0,
    ) -> None:
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, word_emb_matrix.size(1), padding_idx=0)
        self.word_emb.weight.data.copy_(word_emb_matrix)
        self.pred_emb = nn.Embedding(2, pred_emb_dim)
        self.elmo_module = elmo_module

        in_dim = word_emb_matrix.size(1) + pred_emb_dim + (elmo_dim if elmo_module is not None else 0)
        self.encoder = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_tags)

    def forward(
        self,
        word_ids: Tensor,
        pred_ind: Tensor,
        elmo_char_ids: Optional[Tensor] = None,
    ) -> Tensor:
        word = self.word_emb(word_ids)
        pred = self.pred_emb(pred_ind)
        feats = [word, pred]
        if self.elmo_module is not None:
            if elmo_char_ids is None:
                raise ValueError("ELMo mode requires elmo_char_ids")
            elmo_all = self.elmo_module(elmo_char_ids)
            feats.insert(1, elmo_all[:, 1:-1, :])
        x = torch.cat(feats, dim=-1)
        x = self.dropout(x)
        h, _ = self.encoder(x)
        h = self.dropout(h)
        return self.classifier(h)


def make_collate_fn(
    word_vocab: WordVocab,
    tag2id: Dict[str, int],
    elmo_encoder: Optional[CharIdEncoder],
):
    def collate(batch: Sequence[SRLRow]) -> Dict[str, Tensor]:
        bsz = len(batch)
        max_len = max(len(r.tokens) for r in batch)
        word_ids = torch.zeros(bsz, max_len, dtype=torch.long)
        pred_ind = torch.zeros(bsz, max_len, dtype=torch.long)
        tag_ids = torch.zeros(bsz, max_len, dtype=torch.long)
        mask = torch.zeros(bsz, max_len, dtype=torch.bool)

        elmo_ids = None
        if elmo_encoder is not None:
            elmo_ids = torch.zeros(
                bsz, max_len + 2, elmo_encoder.max_chars_per_token, dtype=torch.long
            )

        for i, row in enumerate(batch):
            L = len(row.tokens)
            mask[i, :L] = True
            word_ids[i, :L] = torch.tensor(word_vocab.encode(row.tokens), dtype=torch.long)
            pred_ind[i, :L] = torch.tensor(row.pred_ind, dtype=torch.long)
            tag_ids[i, :L] = torch.tensor([tag2id[t] for t in row.tags], dtype=torch.long)
            if elmo_encoder is not None and elmo_ids is not None:
                elmo_ids[i, : L + 2] = elmo_encoder.encode_tokens(row.tokens)

        out = {"word_ids": word_ids, "pred_ind": pred_ind, "tag_ids": tag_ids, "mask": mask}
        if elmo_ids is not None:
            out["elmo_char_ids"] = elmo_ids
        return out

    return collate


def run_epoch(
    model: DeepSRLTagger,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_tok = 0

    for batch in loader:
        word_ids = batch["word_ids"].to(device)
        pred_ind = batch["pred_ind"].to(device)
        tag_ids = batch["tag_ids"].to(device)
        mask = batch["mask"].to(device)
        elmo_ids = batch.get("elmo_char_ids")
        if elmo_ids is not None:
            elmo_ids = elmo_ids.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        logits = model(word_ids, pred_ind, elmo_char_ids=elmo_ids)
        loss = criterion(logits[mask], tag_ids[mask])

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        n_tok = int(mask.sum().item())
        total_loss += float(loss.detach().cpu()) * n_tok
        total_tok += n_tok

    return total_loss / max(1, total_tok)


@torch.no_grad()
def span_f1_score(
    model: DeepSRLTagger,
    loader: DataLoader,
    device: torch.device,
    id2tag: Sequence[str],
) -> float:
    def bio_spans(tags: Sequence[str]) -> set[Tuple[str, int, int]]:
        spans: set[Tuple[str, int, int]] = set()
        start = -1
        cur_label = ""
        for i, tag in enumerate(tags):
            if tag == "O":
                if start != -1:
                    spans.add((cur_label, start, i - 1))
                    start = -1
                    cur_label = ""
                continue
            if "-" in tag:
                prefix, label = tag.split("-", 1)
            else:
                prefix, label = "B", tag
            if prefix == "B":
                if start != -1:
                    spans.add((cur_label, start, i - 1))
                start, cur_label = i, label
            elif prefix == "I":
                if start == -1 or cur_label != label:
                    if start != -1:
                        spans.add((cur_label, start, i - 1))
                    start, cur_label = i, label
            else:
                if start != -1:
                    spans.add((cur_label, start, i - 1))
                start, cur_label = i, label
        if start != -1:
            spans.add((cur_label, start, len(tags) - 1))
        return spans

    model.eval()
    tp = 0
    fp = 0
    fn = 0
    for batch in loader:
        word_ids = batch["word_ids"].to(device)
        pred_ind = batch["pred_ind"].to(device)
        tag_ids = batch["tag_ids"].to(device)
        mask = batch["mask"].to(device)
        elmo_ids = batch.get("elmo_char_ids")
        if elmo_ids is not None:
            elmo_ids = elmo_ids.to(device)
        logits = model(word_ids, pred_ind, elmo_char_ids=elmo_ids)
        pred_ids = logits.argmax(dim=-1).cpu()
        gold_ids = tag_ids.cpu()
        mask_cpu = mask.cpu()

        for i in range(pred_ids.size(0)):
            seq_len = int(mask_cpu[i].sum().item())
            pred_tags = [id2tag[int(x)] for x in pred_ids[i, :seq_len]]
            gold_tags = [id2tag[int(x)] for x in gold_ids[i, :seq_len]]
            pred_spans = bio_spans(pred_tags)
            gold_spans = bio_spans(gold_tags)
            tp += len(pred_spans & gold_spans)
            fp += len(pred_spans - gold_spans)
            fn += len(gold_spans - pred_spans)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def append_metrics_csv(
    csv_path: Path,
    phase: str,
    model_name: str,
    seed: int,
    epoch: int,
    train_loss: float,
    dev_f1: float,
    test_f1: Optional[float],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["phase", "model", "seed", "epoch", "train_loss", "dev_f1", "test_f1"])
        w.writerow([phase, model_name, seed, epoch, train_loss, dev_f1, "" if test_f1 is None else test_f1])


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "elmo"], default="elmo")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--glove-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4, help="stacked BiLSTM depth")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pred-emb-dim", type=int, default=16)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--srl-dataset", type=str, default="conll2005")
    parser.add_argument("--srl-config", type=str, default="wsj")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[13, 17, 23],
        help="반복 실행할 seed 3개",
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
        default=script_dir / "srl_deep_glove_elmo_metrics.csv",
    )
    args = parser.parse_args()

    train_rows, dev_rows, test_rows = load_conll2005_srl(
        dataset_name=args.srl_dataset,
        dataset_config=args.srl_config,
    )
    print(f"Loaded SRL rows: train={len(train_rows)}, dev={len(dev_rows)}, test={len(test_rows)}")

    word_vocab = WordVocab()
    word_vocab.build([r.tokens for r in train_rows], min_freq=args.min_freq)
    tag2id, id2tag = build_tag_vocab(train_rows)
    glove = load_glove_matrix(word_vocab, dim=args.glove_dim)

    elmo_module = None
    elmo_dim = 0
    elmo_encoder = None
    if args.model == "elmo":
        if not args.bilm_checkpoint.is_file():
            raise FileNotFoundError(f"ELMo checkpoint not found: {args.bilm_checkpoint}")
        bilm, n_layers, elmo_dim, options = load_pretrained_char_bilm_from_checkpoint(
            args.bilm_checkpoint, map_location="cpu"
        )
        elmo_module = ELMoEmbedding(bilm=bilm, num_layers=n_layers)
        elmo_encoder = CharIdEncoder(
            max_chars_per_token=int(options["char_cnn"]["max_characters_per_token"]),
            n_characters=int(options["char_cnn"]["n_characters"]),
            vocab_file=args.elmo_vocab_file if args.elmo_vocab_file.is_file() else None,
        )
        print(f"Using ELMo: layers={n_layers}, dim={elmo_dim}")

    collate = make_collate_fn(word_vocab, tag2id, elmo_encoder)
    train_ds = SRLDataset(train_rows)
    dev_ds = SRLDataset(dev_rows)
    test_ds = SRLDataset(test_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    for seed in args.seeds:
        print(f"\n=== Seed {seed} | model={args.model} ===")
        set_seed(seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(seed),
        )
        dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        model = DeepSRLTagger(
            vocab_size=len(word_vocab),
            num_tags=len(tag2id),
            word_emb_matrix=glove,
            pred_emb_dim=args.pred_emb_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            elmo_module=elmo_module,
            elmo_dim=elmo_dim,
        ).to(device)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        best_dev = float("-inf")
        best_epoch = 0
        best_train_loss = 0.0
        best_state = None

        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
            dev_f1 = span_f1_score(model, dev_loader, device, id2tag)
            if dev_f1 > best_dev:
                best_dev = dev_f1
                best_epoch = epoch
                best_train_loss = train_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            append_metrics_csv(
                csv_path=args.metrics_csv,
                phase="train_epoch",
                model_name=args.model,
                seed=seed,
                epoch=epoch,
                train_loss=train_loss,
                dev_f1=dev_f1,
                test_f1=None,
            )
            print(
                f"[{args.model}] seed={seed} epoch={epoch} "
                f"train_loss={train_loss:.4f} dev_f1={dev_f1:.4f}"
            )

        if best_state is None:
            raise RuntimeError("No best model state was selected from training epochs.")
        model.load_state_dict(best_state)
        test_f1 = span_f1_score(model, test_loader, device, id2tag)
        append_metrics_csv(
            csv_path=args.metrics_csv,
            phase="best_eval",
            model_name=args.model,
            seed=seed,
            epoch=best_epoch,
            train_loss=best_train_loss,
            dev_f1=best_dev,
            test_f1=test_f1,
        )
        print(
            f"[{args.model}] seed={seed} best_epoch={best_epoch} "
            f"best_dev_f1={best_dev:.4f} test_f1={test_f1:.4f}"
        )


if __name__ == "__main__":
    main()
