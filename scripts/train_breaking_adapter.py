#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bas_vla.breaking.training import (
    TrainingConfig,
    attach_bow_features,
    build_vocab_from_records,
    save_training_artifacts,
    split_records,
    train_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the BAS-VLA breaking residual adapter from precomputed action records."
    )
    parser.add_argument("--records-path", type=Path, required=True, help="Path to a JSON list of cached records.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for model artifacts.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--delta-scale", type=float, default=0.35)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--lambda-consistency", type=float, default=0.5)
    parser.add_argument("--lambda-margin", type=float, default=1.0)
    parser.add_argument("--lambda-anchor", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("records-path must point to a JSON list of record dictionaries")
    return payload


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    records = load_records(args.records_path)
    vocab = build_vocab_from_records(records)
    records = attach_bow_features(records, vocab)
    by_split = split_records(records)
    train_records = by_split.get("train", [])
    val_records = by_split.get("val", [])

    if not train_records or not val_records:
        raise RuntimeError("records must contain both 'train' and 'val' splits")

    cfg = TrainingConfig(
        hidden_dim=args.hidden_dim,
        delta_scale=args.delta_scale,
        margin=args.margin,
        lambda_consistency=args.lambda_consistency,
        lambda_margin=args.lambda_margin,
        lambda_anchor=args.lambda_anchor,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    result = train_adapter(
        train_records=train_records,
        val_records=val_records,
        cfg=cfg,
        device=torch.device(args.device),
    )
    save_training_artifacts(args.output_dir, result, vocab, cfg)
    print(f"[bas-vla] saved artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
