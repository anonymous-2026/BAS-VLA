#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path


def has_transformers() -> bool:
    try:
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


CHECKS = {
    "openpi": [
        "BAS_OPENPI_ROOT",
        "BAS_LIBERO_ROOT",
        "BAS_OPENPI_CHECKPOINT",
    ],
    "openvla_oft": [
        "BAS_OPENVLA_OFT_ROOT",
        "BAS_OPENVLA_OFT_CHECKPOINT",
    ],
    "grounded_preserving": [
        "BAS_GROUNDING_DINO_MODEL_ID",
        "BAS_SAM2_MODEL_ID",
        "BAS_GROUNDED_DEVICE",
    ],
}


def describe(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        return f"{name}: missing"
    path = Path(value)
    status = "exists" if path.exists() else "missing-path"
    return f"{name}: {path} [{status}]"


def main() -> int:
    print(f"[python]")
    print(f"transformers: {'available' if has_transformers() else 'missing'}")
    print()
    for section, names in CHECKS.items():
        print(f"[{section}]")
        for name in names:
            print(describe(name))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
