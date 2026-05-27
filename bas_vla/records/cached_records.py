from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_KEYS = (
    "case_id",
    "family",
    "split",
    "clean_instruction",
    "control_instruction",
    "break_instruction",
    "clean_base_action",
    "control_base_action",
    "break_base_action",
    "expert_action",
)


def load_cached_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("records path must point to a JSON list")
    return payload


def _vector_dim(record: dict[str, Any], key: str) -> int:
    value = record.get(key, [])
    if not isinstance(value, list):
        return -1
    return len(value)


def summarize_cached_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    split_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    split_family_counter: dict[str, Counter[str]] = defaultdict(Counter)
    missing_keys: Counter[str] = Counter()
    action_dims: dict[str, Counter[int]] = {
        "clean_base_action": Counter(),
        "control_base_action": Counter(),
        "break_base_action": Counter(),
        "expert_action": Counter(),
    }

    for record in records:
        split = str(record.get("split", "missing"))
        family = str(record.get("family", "missing"))
        split_counter[split] += 1
        family_counter[family] += 1
        split_family_counter[split][family] += 1

        for key in REQUIRED_KEYS:
            if key not in record:
                missing_keys[key] += 1

        for key in action_dims:
            action_dims[key][_vector_dim(record, key)] += 1

    return {
        "num_records": len(records),
        "splits": dict(sorted(split_counter.items())),
        "families": dict(sorted(family_counter.items())),
        "split_family_counts": {
            split: dict(sorted(counter.items()))
            for split, counter in sorted(split_family_counter.items())
        },
        "missing_required_key_counts": dict(sorted(missing_keys.items())),
        "action_dims": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(action_dims.items())
        },
    }
