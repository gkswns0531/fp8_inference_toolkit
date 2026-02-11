"""Activation statistics collector for calibration.

Collects per-layer input channel-wise absmax statistics via forward hooks.
Used by SmoothQuant and potential future GPTQ to understand activation
distributions before quantization.
"""

from __future__ import annotations

import functools
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class LayerStats:
    """Per-layer activation and weight statistics."""

    input_absmax: torch.Tensor   # [K] per-channel max |activation|
    weight_absmax: torch.Tensor  # [K] per-column max |weight|
    num_samples: int


class ActivationStatsCollector:
    """Collects per-layer activation statistics via forward hooks.

    Attaches forward pre-hooks to all nn.Linear modules to track
    per-channel activation maximums across calibration samples.

    Usage:
        model = AutoModel.from_pretrained(...)
        collector = ActivationStatsCollector(model)
        for batch in calibration_data:
            model(batch)
        collector.save("calibration_stats.pt")
        collector.remove_hooks()
    """

    def __init__(self, model: nn.Module) -> None:
        self.stats: dict[str, LayerStats] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._register_hooks(model)

    def _register_hooks(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_pre_hook(
                    functools.partial(self._collect, name))
                self._hooks.append(hook)

    def _collect(
        self,
        name: str,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
    ) -> None:
        x = input[0]
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])  # [tokens, K]
        channel_max = x.detach().abs().amax(dim=0).float()  # [K]

        if name in self.stats:
            self.stats[name].input_absmax = torch.max(
                self.stats[name].input_absmax, channel_max)
            self.stats[name].num_samples += x.shape[0]
        else:
            w = module.weight.detach() if hasattr(module, 'weight') else None
            self.stats[name] = LayerStats(
                input_absmax=channel_max,
                weight_absmax=w.abs().amax(dim=0).float() if w is not None
                else torch.zeros_like(channel_max),
                num_samples=x.shape[0],
            )

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def save(self, path: str) -> None:
        """Save collected stats to a .pt file."""
        data = {}
        for k, v in self.stats.items():
            data[k] = {
                "input_absmax": v.input_absmax.cpu(),
                "weight_absmax": v.weight_absmax.cpu(),
                "num_samples": v.num_samples,
            }
        torch.save(data, path)

    @staticmethod
    def load(path: str) -> dict[str, dict]:
        """Load stats from a .pt file.

        Returns:
            Dict mapping layer names to stat dicts with keys:
            'input_absmax', 'weight_absmax', 'num_samples'
        """
        return torch.load(path, weights_only=False)
