from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

from .backends import GroundedBackendBundle, build_grounded_backend_bundle_from_env
from .deployment import (
    PreservingDeploymentResult,
    PreservingMode,
    deploy_default_action,
    deploy_full_action,
    deploy_preserving_action,
)
from .gate import PreservingGateConfig, PreservingGateInputs, PreservingGateScores, compute_preserving_gate
from .grounded_probe import GroundedProbeConfig, GroundedProbeOutput
from .probe import PreservingProbeOutput, StyleProbeConfig, StyleTriggerConfig
from .selector import PreservingProbeSelectorConfig, select_preserving_probe_output


def _as_float32_array(value: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


@dataclass(frozen=True)
class PreservingPipelineConfig:
    """Paper-aligned preserving auxiliary configuration.

    This helper keeps the public repository carrier-agnostic. It composes probe
    selection, gate computation, and weak-fusion deployment without depending on
    a specific runtime integration for feature extraction or action decoding.
    """

    mode: PreservingMode = "default"
    selector_config: PreservingProbeSelectorConfig = field(default_factory=PreservingProbeSelectorConfig)
    gate_config: PreservingGateConfig = field(default_factory=PreservingGateConfig)
    grounded_probe_config: GroundedProbeConfig = field(default_factory=GroundedProbeConfig)
    style_probe_config: StyleProbeConfig = field(default_factory=StyleProbeConfig)
    style_trigger_config: StyleTriggerConfig = field(default_factory=StyleTriggerConfig)


@dataclass(frozen=True)
class PreservingPipelineInputs:
    observation: Mapping[str, np.ndarray]
    instruction: str
    step_index: int
    visual_mid_gap: float
    semantic_late_gap: float
    action_gap: float
    base_action: np.ndarray | list[float] | None = None
    breaking_action: np.ndarray | list[float] | None = None
    probe_action: np.ndarray | list[float] | None = None
    phase_horizon_steps: int | None = None
    target_object_hint: str | None = None
    receptacle_hint: str | None = None


@dataclass(frozen=True)
class PreservingPipelineResult:
    selected_probe_output: GroundedProbeOutput | PreservingProbeOutput
    gate_inputs: PreservingGateInputs
    gate_scores: PreservingGateScores
    deployment: PreservingDeploymentResult
    backend_name: str
    used_probe_action_fallback: bool


def _resolve_anchor_action(mode: PreservingMode, inputs: PreservingPipelineInputs) -> np.ndarray:
    if mode == "default":
        if inputs.breaking_action is not None:
            return _as_float32_array(inputs.breaking_action)
        if inputs.base_action is not None:
            return _as_float32_array(inputs.base_action)
        raise ValueError("default mode requires either `breaking_action` or `base_action`.")

    if mode == "pres":
        if inputs.base_action is None:
            raise ValueError("pres mode requires `base_action`.")
        return _as_float32_array(inputs.base_action)

    if mode == "full":
        if inputs.breaking_action is None:
            raise ValueError("full mode requires `breaking_action`.")
        return _as_float32_array(inputs.breaking_action)

    raise ValueError(f"unsupported preserving mode: {mode}")


def run_preserving_pipeline(
    inputs: PreservingPipelineInputs,
    config: PreservingPipelineConfig | None = None,
    *,
    grounded_backend_bundle: GroundedBackendBundle | None = None,
) -> PreservingPipelineResult:
    if config is None:
        config = PreservingPipelineConfig()
    if grounded_backend_bundle is None:
        grounded_backend_bundle = build_grounded_backend_bundle_from_env()

    selected_probe_output = select_preserving_probe_output(
        inputs.observation,
        inputs.instruction,
        selector_config=config.selector_config,
        grounded_backend_bundle=grounded_backend_bundle,
        grounded_probe_config=config.grounded_probe_config,
        style_probe_config=config.style_probe_config,
        style_trigger_config=config.style_trigger_config,
        target_object_hint=inputs.target_object_hint,
        receptacle_hint=inputs.receptacle_hint,
    )

    gate_inputs = PreservingGateInputs(
        step_index=inputs.step_index,
        visual_mid_gap=inputs.visual_mid_gap,
        semantic_late_gap=inputs.semantic_late_gap,
        action_gap=inputs.action_gap,
        phase_horizon_steps=inputs.phase_horizon_steps,
    )
    gate_scores = compute_preserving_gate(gate_inputs, config.gate_config)
    if not selected_probe_output.enabled:
        gate_scores = replace(gate_scores, tau=0.0)

    anchor_action = _resolve_anchor_action(config.mode, inputs)

    if config.mode == "default":
        deployment = deploy_default_action(anchor_action)
        return PreservingPipelineResult(
            selected_probe_output=selected_probe_output,
            gate_inputs=gate_inputs,
            gate_scores=gate_scores,
            deployment=deployment,
            backend_name=grounded_backend_bundle.backend_name,
            used_probe_action_fallback=False,
        )

    use_fallback_probe_action = inputs.probe_action is None or not selected_probe_output.enabled
    effective_probe_action = anchor_action if use_fallback_probe_action else _as_float32_array(inputs.probe_action)

    if config.mode == "pres":
        deployment = deploy_preserving_action(
            anchor_action,
            effective_probe_action,
            gate_scores,
            config.gate_config,
        )
    else:
        deployment = deploy_full_action(
            anchor_action,
            effective_probe_action,
            gate_scores,
            config.gate_config,
        )

    return PreservingPipelineResult(
        selected_probe_output=selected_probe_output,
        gate_inputs=gate_inputs,
        gate_scores=gate_scores,
        deployment=deployment,
        backend_name=grounded_backend_bundle.backend_name,
        used_probe_action_fallback=use_fallback_probe_action,
    )
