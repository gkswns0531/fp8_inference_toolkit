"""Calibration infrastructure for INT4 quantization quality improvements.

Provides activation statistics collection for SmoothQuant and other
calibration-dependent quantization techniques.
"""

from calibration.collector import ActivationStatsCollector, LayerStats
from calibration.smoothquant import apply_smoothing_to_weight, compute_smoothing_factors

__all__ = [
    "ActivationStatsCollector",
    "LayerStats",
    "apply_smoothing_to_weight",
    "compute_smoothing_factors",
]
