"""Optional parameter/gradient audits for short distributed training smokes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class VisionGradientReport:
    candidate_count: int
    trainable_count: int
    gradient_count: int
    all_finite: bool
    grad_norm: float


@dataclass(frozen=True)
class VisionUpdateReport:
    updated: bool
    all_finite: bool
    max_delta: float


class VisionTrainingAudit:
    """Inspect visual tower gradients and a representative weight update."""

    _VISION_COMPONENTS = {"visual", "vision_tower", "image_encoder"}

    def __init__(self, model_parts: list[nn.Module], *, min_delta: float = 0.0):
        self.min_delta = float(min_delta)
        self._parameters: list[tuple[str, nn.Parameter]] = []
        seen: set[int] = set()
        for model in model_parts:
            for name, parameter in model.named_parameters():
                if id(parameter) in seen or not self._is_vision_parameter(name):
                    continue
                seen.add(id(parameter))
                self._parameters.append((name, parameter))

    @classmethod
    def _is_vision_parameter(cls, name: str) -> bool:
        return any(component in cls._VISION_COMPONENTS for component in name.split("."))

    @property
    def candidate_count(self) -> int:
        return len(self._parameters)

    @property
    def parameter_names(self) -> list[str]:
        return [name for name, parameter in self._parameters if parameter.requires_grad]

    @property
    def _representative_parameter(self) -> nn.Parameter | None:
        return next((p for _, p in self._parameters if p.requires_grad), None)

    def snapshot(self) -> torch.Tensor | None:
        """Clone one representative visual tensor, avoiding a tower-sized copy."""
        parameter = self._representative_parameter
        return parameter.detach().clone() if parameter is not None else None

    def inspect_gradients(self) -> VisionGradientReport:
        trainable = [parameter for _, parameter in self._parameters if parameter.requires_grad]
        gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
        all_finite = all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
        squared_norm = sum(float(gradient.detach().float().pow(2).sum().item()) for gradient in gradients)
        return VisionGradientReport(
            candidate_count=self.candidate_count,
            trainable_count=len(trainable),
            gradient_count=len(gradients),
            all_finite=all_finite,
            grad_norm=squared_norm**0.5,
        )

    def compare_update(self, before: torch.Tensor | None) -> VisionUpdateReport:
        parameter = self._representative_parameter
        if parameter is None:
            if before is not None:
                raise ValueError("snapshot exists although there is no trainable visual parameter")
            return VisionUpdateReport(updated=False, all_finite=True, max_delta=0.0)
        if before is None:
            raise ValueError("trainable visual parameter has no pre-step snapshot")
        delta = (parameter.detach().float() - before.to(device=parameter.device).float()).abs().max()
        all_finite = all(
            bool(torch.isfinite(p).all().item()) for _, p in self._parameters if p.requires_grad
        )
        max_delta = float(delta.item())
        return VisionUpdateReport(
            updated=max_delta > self.min_delta,
            all_finite=all_finite,
            max_delta=max_delta,
        )
