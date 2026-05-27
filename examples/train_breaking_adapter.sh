#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "$REPO_ROOT/scripts/train_breaking_adapter.py" \
  --input-glob "$REPO_ROOT/cache/*.jsonl" \
  --output-dir "$REPO_ROOT/runs/example_breaking_adapter_train"
