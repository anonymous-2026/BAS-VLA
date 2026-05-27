# Protocol Scope

This repository only preserves code paths that can plausibly become part of a public release.

## Supported direction

The preferred direction for future public BAS-VLA entrypoints is:

- OpenVLA-OFT + LIBERO under an OFT-aligned protocol
- OpenPI + LIBERO under an OpenPI-aligned protocol

The current public evaluation and launcher scripts focus on:

- OpenPI + LIBERO-Object
- local checkpoint inference
- clean / control / break triples
- formal multi-seed rollout matrices

## Excluded direction

This repository intentionally does not preserve public-facing wrappers for:

- raw OpenVLA base on LIBERO with external statistics injection
- SpatialVLA on LIBERO through custom fallback-heavy backends

These paths are excluded because they are not stable enough for a public release.

## Repository consequence

The repository should preserve:

- method code that is independent of unsupported benchmark/model hacks
- configurations that can be carried into a public release
- notes that clarify which setup patterns are intentionally out of scope

The repository should not preserve:

- unsupported benchmark/model conclusions
- ad-hoc wrappers that have not passed a public-release quality bar
