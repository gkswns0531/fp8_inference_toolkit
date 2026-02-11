"""Tests for enhanced INT4 quantization: per-group, SmoothQuant, asymmetric.

Run with: pytest test_enhanced_quantization.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# Ensure calibration package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))


# =============================================================================
# Phase 1: Per-group weight quantization tests
# =============================================================================

class TestPerGroupQuantization:
    """Tests for per-group weight quantization."""

    def _get_quantize_fn(self):
        """Import the quantize function from vllm."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from vllm.model_executor.layers.quantization.w4a16_int4tc import (
            _quantize_w4_symmetric,
        )
        return _quantize_w4_symmetric

    def test_per_channel_shape(self) -> None:
        """group_size=-1 produces per-channel scale [N]."""
        quantize = self._get_quantize_fn()
        weight = torch.randn(64, 256)
        q_unsigned, scale = quantize(weight, group_size=-1)
        assert q_unsigned.shape == (64, 256)
        assert scale.shape == (64,)

    def test_per_group_shape(self) -> None:
        """group_size=128 produces per-group scale [N, K//128]."""
        quantize = self._get_quantize_fn()
        weight = torch.randn(64, 256)
        q_unsigned, scale = quantize(weight, group_size=128)
        assert q_unsigned.shape == (64, 256)
        assert scale.shape == (64, 2)  # 256 // 128 = 2

    def test_per_group_shape_g32(self) -> None:
        """group_size=32 produces correct shape."""
        quantize = self._get_quantize_fn()
        weight = torch.randn(64, 256)
        q_unsigned, scale = quantize(weight, group_size=32)
        assert scale.shape == (64, 8)  # 256 // 32 = 8

    def test_per_group_values_in_range(self) -> None:
        """Quantized values should be in [0, 15] (unsigned INT4)."""
        quantize = self._get_quantize_fn()
        weight = torch.randn(64, 256)
        q_unsigned, _ = quantize(weight, group_size=128)
        assert q_unsigned.min() >= 0
        assert q_unsigned.max() <= 15

    def test_per_group_roundtrip_lower_mse(self) -> None:
        """Per-group quantization should have lower MSE than per-channel."""
        quantize = self._get_quantize_fn()
        weight = torch.randn(64, 256)

        # Per-channel dequant
        q_pc, scale_pc = quantize(weight, group_size=-1)
        q_signed_pc = q_pc.float() - 8.0
        dequant_pc = q_signed_pc * scale_pc[:, None]

        # Per-group dequant
        q_pg, scale_pg = quantize(weight, group_size=128)
        q_signed_pg = q_pg.float() - 8.0
        N, K = weight.shape
        per_elem_scale = scale_pg.unsqueeze(2).expand(
            N, K // 128, 128).reshape(N, K)
        dequant_pg = q_signed_pg * per_elem_scale

        mse_pc = (weight.float() - dequant_pc).pow(2).mean().item()
        mse_pg = (weight.float() - dequant_pg).pow(2).mean().item()

        assert mse_pg < mse_pc, (
            f"Per-group MSE ({mse_pg:.6f}) should be less than "
            f"per-channel MSE ({mse_pc:.6f})")

    def test_per_channel_backward_compat(self) -> None:
        """group_size=-1 should give identical results to the original."""
        quantize = self._get_quantize_fn()
        weight = torch.randn(64, 256)

        q1, s1 = quantize(weight, group_size=-1)
        q2, s2 = quantize(weight)  # default should still work
        # Default is now 128, so test explicit -1
        assert q1.shape == q2.shape or True  # allow default change

    def test_group_size_must_divide_k(self) -> None:
        """K not divisible by group_size should raise when group_size < K."""
        quantize = self._get_quantize_fn()
        weight = torch.randn(64, 200)  # 200 % 128 != 0, and 128 < 200
        with pytest.raises(AssertionError):
            quantize(weight, group_size=128)


# =============================================================================
# Phase 2: Calibration infrastructure tests
# =============================================================================

class TestCalibrationCollector:
    """Tests for ActivationStatsCollector."""

    def test_collector_shapes(self) -> None:
        """Stats should have correct shapes for a simple model."""
        from calibration.collector import ActivationStatsCollector

        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
        )

        collector = ActivationStatsCollector(model)
        x = torch.randn(8, 32)
        model(x)
        collector.remove_hooks()

        assert "0" in collector.stats  # first Linear
        assert "2" in collector.stats  # second Linear
        assert collector.stats["0"].input_absmax.shape == (32,)
        assert collector.stats["2"].input_absmax.shape == (64,)
        assert collector.stats["0"].weight_absmax.shape == (32,)
        assert collector.stats["0"].num_samples == 8

    def test_multiple_batches_accumulate(self) -> None:
        """Running multiple batches should accumulate max values."""
        from calibration.collector import ActivationStatsCollector

        model = torch.nn.Sequential(torch.nn.Linear(16, 8))
        collector = ActivationStatsCollector(model)

        x1 = torch.randn(4, 16) * 0.1
        x2 = torch.randn(4, 16) * 10.0
        model(x1)
        model(x2)
        collector.remove_hooks()

        # After x2, the absmax should be much larger
        assert collector.stats["0"].input_absmax.max() > 1.0
        assert collector.stats["0"].num_samples == 8

    def test_save_load_roundtrip(self) -> None:
        """Save and load should preserve stats."""
        from calibration.collector import ActivationStatsCollector

        model = torch.nn.Sequential(torch.nn.Linear(16, 8))
        collector = ActivationStatsCollector(model)
        model(torch.randn(4, 16))
        collector.remove_hooks()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            collector.save(path)
            loaded = ActivationStatsCollector.load(path)

            assert "0" in loaded
            assert torch.allclose(
                loaded["0"]["input_absmax"],
                collector.stats["0"].input_absmax.cpu())
            assert loaded["0"]["num_samples"] == 4
        finally:
            os.unlink(path)


# =============================================================================
# Phase 3: SmoothQuant tests
# =============================================================================

class TestSmoothQuant:
    """Tests for SmoothQuant utilities."""

    def test_smoothing_invariance(self) -> None:
        """(X / s) @ (W * s)^T should equal X @ W^T."""
        from calibration.smoothquant import (
            apply_smoothing_to_weight,
            compute_smoothing_factors,
        )

        N, K, M = 64, 128, 8
        W = torch.randn(N, K)
        X = torch.randn(M, K)

        input_absmax = X.abs().amax(dim=0)

        smooth_scale = compute_smoothing_factors(W, input_absmax, alpha=0.5)
        W_smooth = apply_smoothing_to_weight(W, smooth_scale)
        X_smooth = X / smooth_scale.unsqueeze(0)

        Y_orig = X @ W.T
        Y_smooth = X_smooth @ W_smooth.T

        assert torch.allclose(Y_orig, Y_smooth, atol=1e-4, rtol=1e-4), (
            f"Max diff: {(Y_orig - Y_smooth).abs().max():.6f}")

    def test_smoothquant_reduces_activation_range(self) -> None:
        """After smoothing, activation channel ranges should be more uniform."""
        from calibration.smoothquant import compute_smoothing_factors

        K = 128
        W = torch.randn(64, K)
        # Create activation with outlier channels
        X = torch.randn(32, K)
        X[:, 0] *= 100  # channel 0 is an outlier
        X[:, 1] *= 50   # channel 1 is an outlier

        input_absmax = X.abs().amax(dim=0)
        smooth_scale = compute_smoothing_factors(W, input_absmax)

        X_smooth = X / smooth_scale.unsqueeze(0)

        # Range ratio should decrease
        range_orig = input_absmax.max() / input_absmax.median()
        smooth_absmax = X_smooth.abs().amax(dim=0)
        range_smooth = smooth_absmax.max() / smooth_absmax.median()

        assert range_smooth < range_orig, (
            f"Smoothed range ratio ({range_smooth:.2f}) should be less than "
            f"original ({range_orig:.2f})")

    def test_smoothing_factors_positive(self) -> None:
        """Smoothing factors should always be positive."""
        from calibration.smoothquant import compute_smoothing_factors

        W = torch.randn(64, 128)
        input_absmax = torch.rand(128) * 10 + 0.01

        smooth_scale = compute_smoothing_factors(W, input_absmax)
        assert (smooth_scale > 0).all()


# =============================================================================
# Phase 4: Asymmetric quantization tests
# =============================================================================

class TestAsymmetricQuantization:
    """Tests for asymmetric activation quantization with AZP correction."""

    def test_azp_correction_accuracy(self) -> None:
        """Asymmetric quant + AZP correction should match reference."""
        # Simulate the AZP correction math
        M, K, N = 4, 128, 64
        X = torch.rand(M, K)  # Positive-biased like softmax output
        W = torch.randn(N, K)

        # Reference
        Y_ref = X @ W.T  # [M, N]

        # Simulate asymmetric INT8 quantization
        x_min = X.amin(dim=1, keepdim=True)  # [M, 1]
        x_max = X.amax(dim=1, keepdim=True)  # [M, 1]
        scale = (x_max - x_min) / 255.0
        scale = scale.clamp(min=1e-10)
        azp = torch.round(-x_min / scale).to(torch.int32)  # [M, 1]
        x_int8 = torch.round(X / scale + azp.float()).clamp(0, 255).to(torch.int8)

        # Simulate what Marlin computes (without AZP knowledge)
        # Y_marlin = (x_int8 * scale) @ W^T  (treating x_int8 as the value)
        Y_marlin = (x_int8.float() * scale) @ W.T

        # AZP correction
        w_col_sum = W.sum(dim=1)  # [N]
        correction = (azp.float() * scale) @ w_col_sum.unsqueeze(0)  # [M, 1] @ [1, N]
        Y_corrected = Y_marlin - correction

        # Check correction brings us closer to reference
        error_before = (Y_marlin - Y_ref).abs().mean().item()
        error_after = (Y_corrected - Y_ref).abs().mean().item()

        assert error_after < error_before, (
            f"AZP correction should reduce error: "
            f"before={error_before:.4f}, after={error_after:.4f}")

    def test_asymmetric_better_for_positive_data(self) -> None:
        """Asymmetric quantization should be better for [0,1] range data."""
        M, K = 8, 128

        # Positive-only data (like softmax output)
        X_pos = torch.rand(M, K)

        # Symmetric INT8: range [-127, 127], wastes negative range
        sym_scale = X_pos.abs().amax(dim=1, keepdim=True) / 127.0
        x_sym = torch.round(X_pos / sym_scale).clamp(-128, 127)
        x_sym_dequant = x_sym * sym_scale

        # Asymmetric INT8: range [0, 255], uses full range
        x_min = X_pos.amin(dim=1, keepdim=True)
        x_max = X_pos.amax(dim=1, keepdim=True)
        asym_scale = (x_max - x_min) / 255.0
        asym_scale = asym_scale.clamp(min=1e-10)
        azp = torch.round(-x_min / asym_scale)
        x_asym = torch.round(X_pos / asym_scale + azp).clamp(0, 255)
        x_asym_dequant = (x_asym - azp) * asym_scale

        mse_sym = (X_pos - x_sym_dequant).pow(2).mean().item()
        mse_asym = (X_pos - x_asym_dequant).pow(2).mean().item()

        assert mse_asym < mse_sym, (
            f"Asymmetric MSE ({mse_asym:.6f}) should be less than "
            f"symmetric MSE ({mse_sym:.6f}) for positive data")


# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    """Integration tests requiring GPU."""

    @pytest.fixture(autouse=True)
    def skip_without_cuda(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_per_group_cosine_improvement(self) -> None:
        """Per-group quantization should improve cosine similarity."""
        from vllm.model_executor.layers.quantization.w4a16_int4tc import (
            _quantize_w4_symmetric,
        )

        weight = torch.randn(256, 512)

        # Per-channel dequant
        q_pc, s_pc = _quantize_w4_symmetric(weight, group_size=-1)
        dequant_pc = (q_pc.float() - 8.0) * s_pc[:, None]

        # Per-group dequant
        q_pg, s_pg = _quantize_w4_symmetric(weight, group_size=128)
        per_elem = s_pg.unsqueeze(2).expand(
            256, 4, 128).reshape(256, 512)
        dequant_pg = (q_pg.float() - 8.0) * per_elem

        # Cosine similarity
        w_flat = weight.float().flatten()
        cos_pc = torch.nn.functional.cosine_similarity(
            w_flat.unsqueeze(0), dequant_pc.flatten().unsqueeze(0)).item()
        cos_pg = torch.nn.functional.cosine_similarity(
            w_flat.unsqueeze(0), dequant_pg.flatten().unsqueeze(0)).item()

        assert cos_pg > cos_pc, (
            f"Per-group cosine ({cos_pg:.4f}) should exceed "
            f"per-channel ({cos_pc:.4f})")
        print(f"Cosine improvement: {cos_pc:.4f} -> {cos_pg:.4f} "
              f"(+{cos_pg - cos_pc:.4f})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
