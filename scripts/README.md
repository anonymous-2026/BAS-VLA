# Scripts

This folder contains public CLI entrypoints that are already clean enough to preserve.

The current focus is on:

- public training utilities for BAS-VLA adapters
- public OpenPI + LIBERO evaluation utilities
- public OpenPI + LIBERO appearance-shift evaluation utilities
- public OpenVLA-OFT + LIBERO semantic-break evaluation utilities
- public shell launchers for semantic-break and appearance-shift campaigns
- stable summary and media helpers

Scripts are only preserved here when they satisfy all of the following:

1. officially supported or publicly verifiable model-benchmark alignment
2. no hardcoded machine-specific paths
3. no unsupported stat swapping or silent fallback behavior
4. clean command-line interface suitable for release

## Available scripts

- `train_breaking_adapter.py`
  - trains the BAS-VLA residual adapter from precomputed cached records
  - does not perform private model inference by itself
- `eval_openpi_libero.py`
  - runs local OpenPI checkpoint evaluation on LIBERO
  - supports instruction overrides, pair IDs, and image-shift presets
- `eval_openpi_appearance_libero.py`
  - runs local OpenPI checkpoint evaluation on LIBERO with appearance-shift presets
  - serves as the public appearance benchmark runner alongside the preserving auxiliary modules under `bas_vla/preserving/`
  - supports `--preserving-mode pres` for probe-gated weak fusion
- `eval_openvla_oft_libero.py`
  - runs baseline OpenVLA-OFT checkpoint evaluation on LIBERO semantic-break pairs
  - expects the official OpenVLA-OFT runtime through environment variables
  - supports `--preserving-mode default|pres|full`
- `run_openpi_campaign_worker.sh`
  - executes one clean/control/break condition set across multiple seeds
- `launch_openpi_main_campaign.sh`
  - launches the current public OpenPI main campaign example
- `run_openpi_appearance_pair.sh`
  - launches one clean-vs-shift appearance pair for a local OpenPI checkpoint
- `run_openpi_appearance_pair_formal.sh`
  - launches multi-seed clean-vs-shift appearance evaluation for one task row
- `run_openpi_appearance_visual_episode.sh`
  - reruns one selected episode with saved rollout media for appearance figures
- `launch_openpi_appearance_formal.sh`
  - launches multiple appearance rows from a TSV matrix
- `run_openpi_formal_worker.sh`
  - runs one instruction condition for a formal task triple
- `run_openpi_formal_task_triple.sh`
  - runs clean/control/break for one formal task row
- `launch_formal_task_triples.sh`
  - launches multiple formal task rows from a TSV matrix
- `launch_formal_task_matrix.sh`
  - launches single-condition formal rows from a TSV matrix
- `check_semantic_break_pairs.py`
  - validates a semantic break pair JSON file
  - reports family and suite coverage
- `summarize_cached_records.py`
  - summarizes cached record files before public training release
  - checks split coverage, family counts, and action vector dimensions
- `bootstrap_eval.py`
  - computes bootstrap summaries for JSON-based metrics
- `export_process_metrics.py`
  - exports curve-ready rows from paired condition summaries
- `make_pair_montage.py`
  - builds a reference-vs-variant image montage from episode frame folders
- `make_video_montage.py`
  - samples rollout videos into a paper-ready storyboard image

## Required environment variables for OpenPI + LIBERO scripts

- `BAS_OPENPI_ROOT`
  - path to the official OpenPI repository checkout
- `BAS_LIBERO_ROOT`
  - path to the LIBERO repository or installation root
- `BAS_OPENPI_CHECKPOINT`
  - path to the OpenPI checkpoint directory

Optional variables:

- `BAS_PYTHON_BIN`
- `BAS_LIBERO_SITE_PACKAGES`
- `LIBERO_CONFIG_PATH`
- `BAS_GROUNDING_DINO_MODEL_ID`
- `BAS_SAM2_MODEL_ID`
- `BAS_GROUNDED_DEVICE`

See `.env.example` for a minimal template.

The single repository dependency file is `requirements.txt`.
