from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import model.diana_h3p.layer2_backend as backend_module
from model.diana_h3p.layer2_backend import (
    NumpyLayer2Backend,
    PARITY_ATOL,
    PARITY_RTOL,
    TorchLayer2Backend,
    make_layer2_backend,
)


def _cases() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    diagonal_a = np.diag([0.4, 0.7, 1.1])
    diagonal_psi = np.diag([0.3, 0.5, 0.8])
    correlated_a = np.array(
        [[0.60, 0.18, -0.08], [0.18, 0.90, 0.21], [-0.08, 0.21, 0.55]],
        dtype=np.float64,
    )
    correlated_psi = np.array(
        [[0.35, -0.09, 0.04], [-0.09, 0.52, 0.13], [0.04, 0.13, 0.44]],
        dtype=np.float64,
    )
    rotation = np.array(
        [[1.0, 1.0, 0.0], [1.0, -1.0, 1.0], [0.0, 1.0, -1.0]],
        dtype=np.float64,
    )
    # H3P always applies its prespecified relative eigenvalue floor before the
    # backend. This remains ill-conditioned while representing a valid backend
    # input rather than a covariance the statistical layer would reject/floor.
    rotation, _ = np.linalg.qr(rotation)
    nearly_singular_a = rotation @ np.diag([0.8, 0.2, 1e-6]) @ rotation.T
    nearly_singular_psi = rotation @ np.diag([0.3, 0.1, 1.5e-6]) @ rotation.T
    residual = np.array([0.25, -0.12, 0.31], dtype=np.float64)
    return [
        (diagonal_a, diagonal_psi, residual),
        (correlated_a, correlated_psi, residual),
        (nearly_singular_a, nearly_singular_psi, residual),
    ]


def _batched_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cases = _cases()
    sigma_a = np.stack([case[0] for case in cases])
    psi = np.stack([case[1] for case in cases])
    residuals = np.stack([case[2] for case in cases])
    return sigma_a, psi, residuals, np.array([False, True, True])


def _assert_valid_posterior(means: np.ndarray, covariance: np.ndarray) -> None:
    assert means.shape == (3, 3)
    assert covariance.shape == (3, 3, 3)
    assert np.isfinite(means).all()
    assert np.isfinite(covariance).all()
    assert np.allclose(covariance, covariance.transpose(0, 2, 1))
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10


def test_numpy_diagonal_formula_and_exact_k0() -> None:
    sigma_a, psi, residual = _cases()[0]
    backend = NumpyLayer2Backend()
    means, covariance = backend.posterior_batch(
        sigma_a,
        psi,
        np.stack([residual, residual, residual]),
        np.array([False, True, True]),
    )
    assert np.array_equal(means[0], np.zeros(3))
    assert np.array_equal(covariance[0], sigma_a)
    gain = np.diag(sigma_a) / (np.diag(sigma_a) + np.diag(psi))
    assert np.allclose(means[1:], gain * residual, rtol=1e-12, atol=1e-12)
    expected_covariance = np.diag(
        np.diag(sigma_a) * np.diag(psi) / (np.diag(sigma_a) + np.diag(psi))
    )
    assert np.allclose(covariance[1:], expected_covariance, rtol=1e-12, atol=1e-12)


def test_numpy_batched_correlated_and_nearly_singular_are_stable() -> None:
    sigma_a, psi, residuals, calibrated = _batched_inputs()
    means, covariance = NumpyLayer2Backend().posterior_batch(
        sigma_a, psi, residuals, calibrated
    )
    _assert_valid_posterior(means, covariance)
    assert np.array_equal(means[0], np.zeros(3))
    assert np.array_equal(covariance[0], sigma_a[0])


def test_controlled_more_informative_psi_reduces_posterior_covariance() -> None:
    sigma_a = _cases()[1][0]
    psi_three = np.diag([0.8, 0.7, 0.9])
    psi_seven = np.diag([0.3, 0.2, 0.4])
    _, covariance = NumpyLayer2Backend().posterior_batch(
        sigma_a,
        np.stack([psi_three, psi_seven]),
        np.zeros((2, 3)),
        np.array([True, True]),
    )
    assert np.linalg.eigvalsh(covariance[0] - covariance[1]).min() >= -1e-10


def test_interval_batch_is_batched_finite_ordered_and_nonnegative() -> None:
    sigma_a, psi, residuals, calibrated = _batched_inputs()
    backend = NumpyLayer2Backend()
    _, posterior = backend.posterior_batch(sigma_a, psi, residuals, calibrated)
    points = np.array(
        [[0.05, 1.2, 2.1], [0.7, 1.5, 2.5], [0.2, 0.4, 0.9], [1.0, 1.1, 1.2]]
    )
    future = np.stack([np.eye(3) * value for value in (0.2, 0.3, 0.4)])
    multipliers = np.array([[1.1, 1.2, 1.3], [1.4, 1.5, 1.6], [1.7, 1.8, 1.9]])
    indices = np.array([0, 1, 2, 1])
    lower, upper = backend.interval_batch(
        points,
        future,
        posterior,
        indices,
        multipliers,
    )
    assert lower.shape == upper.shape == points.shape
    assert np.isfinite(lower).all() and np.isfinite(upper).all()
    assert (lower >= 0).all()
    assert (lower <= points).all()
    assert (points <= upper).all()
    assert (lower[0] == 0).any()


def test_numpy_repeated_output_is_bitwise_deterministic() -> None:
    arguments = _batched_inputs()
    backend = NumpyLayer2Backend()
    first = backend.posterior_batch(*arguments)
    second = backend.posterior_batch(*arguments)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_lazy_missing_torch_import_does_not_break_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> object:
        raise ModuleNotFoundError("synthetic missing optional torch")

    monkeypatch.setattr(backend_module, "_import_torch", missing)
    numpy_backend = make_layer2_backend("numpy")
    assert numpy_backend.metadata["backend"] == "numpy"
    fallback = make_layer2_backend("torch", allow_backend_fallback=True)
    assert fallback.metadata["backend"] == "numpy"
    assert fallback.metadata["fallback_from"] == "torch"
    with pytest.raises(ModuleNotFoundError):
        make_layer2_backend("torch", allow_backend_fallback=False)


def test_explicit_cuda_device_fallback_policy() -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    resolved, reason = backend_module._resolve_torch_device(
        fake_torch, "cuda", allow_device_fallback=True
    )
    assert resolved == "cpu"
    assert reason is not None
    with pytest.raises(RuntimeError, match="CUDA"):
        backend_module._resolve_torch_device(
            fake_torch, "cuda", allow_device_fallback=False
        )


def _assert_torch_parity(device: str) -> None:
    sigma_a, psi, residuals, calibrated = _batched_inputs()
    numpy_backend = NumpyLayer2Backend()
    torch_backend = TorchLayer2Backend(device=device)
    expected_mean, expected_covariance = numpy_backend.posterior_batch(
        sigma_a, psi, residuals, calibrated
    )
    actual_mean, actual_covariance = torch_backend.posterior_batch(
        sigma_a, psi, residuals, calibrated
    )
    assert np.allclose(
        actual_mean, expected_mean, rtol=PARITY_RTOL, atol=PARITY_ATOL
    )
    assert np.allclose(
        actual_covariance,
        expected_covariance,
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
    )
    assert np.array_equal(actual_mean[0], np.zeros(3))
    assert np.array_equal(actual_covariance[0], sigma_a[0])

    points = np.array([[0.1, 1.0, 2.0], [0.8, 1.2, 1.8], [0.3, 0.6, 0.9]])
    future = np.stack([np.eye(3) * value for value in (0.2, 0.3, 0.4)])
    multipliers = np.array([[1.1, 1.2, 1.3], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]])
    expected_interval = numpy_backend.interval_batch(
        points, future, expected_covariance, None, multipliers
    )
    actual_interval = torch_backend.interval_batch(
        points, future, actual_covariance, None, multipliers
    )
    for actual, expected in zip(actual_interval, expected_interval, strict=True):
        assert np.allclose(actual, expected, rtol=PARITY_RTOL, atol=PARITY_ATOL)

    repeated_mean, repeated_covariance = torch_backend.posterior_batch(
        sigma_a, psi, residuals, calibrated
    )
    assert np.array_equal(actual_mean, repeated_mean)
    assert np.array_equal(actual_covariance, repeated_covariance)


def test_torch_cpu_parity_and_determinism() -> None:
    pytest.importorskip("torch")
    _assert_torch_parity("cpu")


def test_torch_cuda_parity_and_determinism_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    _assert_torch_parity("cuda")
