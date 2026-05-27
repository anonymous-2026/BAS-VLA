# Integration Guidance

This document is a **recommendation**, not a rigid rulebook.

Its purpose is to keep future public BAS-VLA runners anchored to official or publicly verifiable model-benchmark recipes whenever possible.

## Preferred public pairings

### Tier 1

- OpenVLA-OFT + LIBERO
- OpenVLA suite-specific fine-tuned checkpoints + LIBERO
- OpenPI + LIBERO

These pairings are the best candidates for future public entrypoints because they have the strongest protocol support.

### Tier 2

- GR00T + LIBERO
- SmolVLA + LIBERO / MetaWorld
- VLA-0 + LIBERO

These are good expansion targets after the main release path is stable.

### Tier 3

- RoboCasa
- ManiSkill2
- MetaWorld
- RoboHive
- ProcTHOR / AI2-THOR
- CALVIN

These are useful future targets, but they should not drive the first public BAS-VLA release.

## Discouraged public pairings

- OpenVLA base raw checkpoint + LIBERO with external statistics injection
- SpatialVLA + custom LIBERO backend

These combinations should not be used as the basis for future public release scripts.

## Public runner principle

For any model-benchmark pair that may later receive a public runner:

1. verify the official or public recipe first
2. reproduce a clean anchor first
3. only then add BAS-VLA logic
4. only then wrap it behind a release-quality CLI
