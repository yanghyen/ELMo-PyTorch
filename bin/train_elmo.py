import argparse
import json
import os

import yaml

from bilm.src.training import train_elmo
from bilm.src.data import Vocabulary, UnicodeCharsVocabulary


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    return data


def _build_options_from_config(cfg, vocab_size):
    """Map config.yaml sections to LanguageModelTrainer options dict."""
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    m_char = model.get("char_cnn") or {}
    m_lstm = model.get("lstm") or {}

    use_char = bool(model.get("use_character_inputs", True))
    filters = m_char.get("filters")
    if not filters:
        filters = [
            [1, 32],
            [2, 32],
            [3, 64],
            [4, 128],
            [5, 256],
            [6, 512],
            [7, 1024],
        ]

    max_word_length = int(model.get("max_word_length", 50))

    options = {
        "bidirectional": True,
        "char_cnn": {
            "activation": m_char.get("activation", "relu"),
            "embedding": {"dim": int(m_char.get("embedding_dim", 16))},
            "filters": filters,
            "max_characters_per_token": max_word_length,
            "n_characters": int(m_char.get("n_characters", 261)),
            "n_highway": int(m_char.get("n_highway", 2)),
            "projection": {"dim": int(m_char.get("projection_dim", 512))},
        }
        if use_char
        else None,
        "dropout": float(training.get("dropout", 0.1)),
        "lstm": {
            "cell_clip": float(m_lstm.get("cell_clip", 3)),
            "dim": int(m_lstm.get("dim", 1024)),
            "n_layers": int(m_lstm.get("n_layers", 2)),
            "proj_clip": float(m_lstm.get("proj_clip", 3)),
            "projection_dim": int(m_lstm.get("projection_dim", 512)),
            "use_skip_connections": bool(m_lstm.get("use_skip_connections", True)),
        },
        "all_clip_norm_val": float(training.get("all_clip_norm_val", 10.0)),
        "clip_grad_norm": float(training.get("clip_grad_norm", 5.0)),
        "n_tokens_vocab": vocab_size,
        "batch_size": int(training.get("batch_size", 64)),
        "unroll_steps": int(
            training.get("num_steps", training.get("unroll_steps", 20))
        ),
        "n_negative_samples_batch": int(
            training.get("n_negative_samples_batch", 8192)
        ),
        "learning_rate": float(training.get("learning_rate", 1e-3)),
        "lr_decay_steps": int(training.get("lr_decay_steps", 1)),
        "lr_decay_rate": float(training.get("lr_decay_rate", 0.9)),
        "steps_per_epoch": int(training.get("steps_per_epoch", 1000)),
        "n_epochs": int(training.get("n_epochs", 10)),
        "max_steps": training.get("max_steps"),
        "save_every_steps": int(training.get("save_every_steps", 10000)),
        "log_every_steps": int(training.get("log_every_steps", 100)),
        "lr_scheduler_step_interval": training.get("lr_scheduler_step_interval"),
        "validation_every_steps": training.get("validation_every_steps"),
        "eval_max_batches": int(training.get("eval_max_batches", 50)),
    }
    return options


def _resolve_paths(cfg, args):
    data = cfg.get("data", {})
    save_dir = args.save_dir or data.get("save_dir")
    vocab_file = args.vocab_file or data.get("vocab_file")
    train_prefix = args.train_prefix or data.get("train_prefix")
    if args.test_prefix is not None:
        test_prefix = args.test_prefix
    else:
        test_prefix = data.get("test_prefix")
    if getattr(args, "valid_prefix", None) is not None:
        valid_prefix = args.valid_prefix
    else:
        valid_prefix = data.get("valid_prefix")
    return save_dir, vocab_file, train_prefix, test_prefix, valid_prefix


def main_with_config(cfg, args):
    save_dir, vocab_file, train_prefix, test_prefix, valid_prefix = _resolve_paths(
        cfg, args
    )
    for name, val in [
        ("save_dir", save_dir),
        ("vocab_file", vocab_file),
        ("train_prefix", train_prefix),
    ]:
        if not val:
            raise ValueError(
                f"Missing {name}: set in config.yaml under data: or pass CLI flag."
            )

    model_cfg = cfg.get("model", {})
    use_character_inputs = bool(model_cfg.get("use_character_inputs", True))
    max_word_length = int(model_cfg.get("max_word_length", 50))

    if use_character_inputs:
        vocab = UnicodeCharsVocabulary(vocab_file, max_word_length)
    else:
        vocab = Vocabulary(vocab_file, validate_file=True)

    options = _build_options_from_config(cfg, len(vocab))
    max_steps = options.get("max_steps")
    if max_steps is not None:
        max_steps = int(max_steps)
    n_epochs = int(options.get("n_epochs", 10))

    os.makedirs(save_dir, exist_ok=True)
    options_file = os.path.join(save_dir, "options.json")
    with open(options_file, "w", encoding="utf-8") as f:
        json.dump(options, f, indent=2)

    print("Training options:")
    print(json.dumps(options, indent=2))

    train_elmo(
        options=options,
        train_prefix=train_prefix,
        vocab_file=vocab_file,
        save_dir=save_dir,
        test_prefix=test_prefix,
        valid_prefix=valid_prefix,
        num_epochs=n_epochs,
        max_steps=max_steps,
    )

    print(f"Training completed! Model saved to {save_dir}")


def main_cli_only(args):
    """Legacy CLI without YAML (defaults match previous bin/train_elmo.py)."""
    if args.use_character_inputs:
        vocab = UnicodeCharsVocabulary(args.vocab_file, 50)
    else:
        vocab = Vocabulary(args.vocab_file, validate_file=True)

    options = {
        "bidirectional": True,
        "char_cnn": {
            "activation": "relu",
            "embedding": {"dim": 16},
            "filters": [
                [1, 32],
                [2, 32],
                [3, 64],
                [4, 128],
                [5, 256],
                [6, 512],
                [7, 1024],
            ],
            "max_characters_per_token": 50,
            "n_characters": 261,
            "n_highway": 2,
            "projection": {"dim": 512},
        }
        if args.use_character_inputs
        else None,
        "dropout": 0.1,
        "lstm": {
            "cell_clip": 3,
            "dim": 4096,
            "n_layers": 2,
            "proj_clip": 3,
            "projection_dim": 512,
            "use_skip_connections": True,
        },
        "all_clip_norm_val": 10.0,
        "clip_grad_norm": 10.0,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "n_tokens_vocab": len(vocab),
        "unroll_steps": 20,
        "n_negative_samples_batch": 8192,
        "learning_rate": args.learning_rate,
        "lr_decay_steps": 10,
        "lr_decay_rate": 0.9,
        "steps_per_epoch": args.steps_per_epoch,
        "save_every_steps": 10000,
        "log_every_steps": 100,
        "validation_every_steps": None,
        "eval_max_batches": 50,
    }

    os.makedirs(args.save_dir, exist_ok=True)
    options_file = os.path.join(args.save_dir, "options.json")
    with open(options_file, "w", encoding="utf-8") as f:
        json.dump(options, f, indent=2)

    print("Training options:")
    print(json.dumps(options, indent=2))

    train_elmo(
        options=options,
        train_prefix=args.train_prefix,
        vocab_file=args.vocab_file,
        save_dir=args.save_dir,
        test_prefix=args.test_prefix,
        valid_prefix=getattr(args, "valid_prefix", None),
        num_epochs=args.n_epochs,
        max_steps=None,
    )

    print(f"Training completed! Model saved to {args.save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train ELMo / BiLM with PyTorch (YAML config or CLI)."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config (e.g. config.yaml). When set, training.* and model.* are read from file; CLI can override paths.",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        help="Checkpoint directory (overrides config data.save_dir)",
    )
    parser.add_argument(
        "--vocab_file",
        default=None,
        help="Vocabulary file (overrides config data.vocab_file)",
    )
    parser.add_argument(
        "--train_prefix",
        default=None,
        help="Glob for training shards (overrides config data.train_prefix)",
    )
    parser.add_argument(
        "--test_prefix",
        default=None,
        help="Optional test shard glob (overrides config data.test_prefix)",
    )
    parser.add_argument(
        "--valid_prefix",
        default=None,
        help="Optional validation shard glob (overrides config data.valid_prefix)",
    )
    parser.add_argument(
        "--use_character_inputs",
        action="store_true",
        help="CLI-only mode: character inputs",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=1000,
        help="CLI-only mode: steps per epoch",
    )

    args = parser.parse_args()

    if args.config:
        cfg = _load_yaml(args.config)
        main_with_config(cfg, args)
    else:
        if not args.save_dir or not args.vocab_file or not args.train_prefix:
            parser.error(
                "Without --config, you must pass --save_dir, --vocab_file, and --train_prefix."
            )
        main_cli_only(args)


if __name__ == "__main__":
    main()
