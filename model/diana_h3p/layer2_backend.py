"""Numerically equivalent NumPy and optional PyTorch algebra for Diana-H3P.

This module deliberately owns only the small, batched linear-algebra portion of
Layer 2. Covariance estimation and participant-block interval calibration remain
in the statistical implementation.  Both backends accept and return NumPy
``float64`` arrays, which makes backend selection unable to alter the public
model contract.  PyTorch is imported lazily and is never required by the NumPy
reference path.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Protocol

import numpy as np


DIMENSION = 3
ALLOWED_BUDGETS = (0, 3, 7)
PARITY_RTOL = 1e-8
PARITY_ATOL = 1e-10


class Layer2Backend(Protocol):
    """Minimal backend contract used by the H3P posterior and intervals."""

    @property
    def metadata(self) -> dict[str, Any]: ...

    def posterior_batch(
        self,
        sigma_a: np.ndarray,
        psi: np.ndarray,
        residual_means: np.ndarray,
        calibrated: bool | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def interval_batch(
        self,
        points: np.ndarray,
        sigma_future: np.ndarray,
        posterior_covariances: np.ndarray,
        participant_index: np.ndarray | None,
        multipliers: np.ndarray,
        *,
        variance_floor: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray]: ...


def _matrix_batch(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (DIMENSION, DIMENSION):
        array = array[None, :, :]
    if array.ndim != 3 or array.shape[1:] != (DIMENSION, DIMENSION):
        raise ValueError(f"{name} must have shape (3,3) or (B,3,3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    if not np.allclose(array, np.swapaxes(array, -1, -2), rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return np.asarray((array + np.swapaxes(array, -1, -2)) / 2.0, dtype=np.float64)


def _vector_batch(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (DIMENSION,):
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != DIMENSION:
        raise ValueError(f"{name} must have shape (3,) or (B,3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _calibrated_batch(value: bool | np.ndarray) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim == 0:
        raw = raw.reshape(1)
    if raw.ndim != 1 or raw.dtype != np.dtype(bool):
        raise ValueError("calibrated must be a scalar or one-dimensional bool array")
    return raw.astype(bool, copy=False)


def _broadcast_first_axis(array: np.ndarray, size: int, name: str) -> np.ndarray:
    if len(array) == size:
        return np.array(array, copy=True)
    if len(array) == 1:
        return np.array(np.broadcast_to(array, (size, *array.shape[1:])), copy=True)
    raise ValueError(f"{name} batch length must be one or {size}")


def _prepare_posterior_inputs(
    sigma_a: np.ndarray,
    psi: np.ndarray,
    residual_means: np.ndarray,
    calibrated: bool | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sigma_a_batch = _matrix_batch(sigma_a, "sigma_a")
    psi_batch = _matrix_batch(psi, "psi")
    residual_batch = _vector_batch(residual_means, "residual_means")
    calibrated_batch = _calibrated_batch(calibrated)
    size = max(
        len(sigma_a_batch),
        len(psi_batch),
        len(residual_batch),
        len(calibrated_batch),
    )
    sigma_a_batch = _broadcast_first_axis(sigma_a_batch, size, "sigma_a")
    psi_batch = _broadcast_first_axis(psi_batch, size, "psi")
    residual_batch = _broadcast_first_axis(
        residual_batch, size, "residual_means"
    )
    calibrated_batch = _broadcast_first_axis(
        calibrated_batch, size, "calibrated"
    ).astype(bool, copy=False)
    if (np.linalg.eigvalsh(sigma_a_batch) <= 0.0).any():
        raise ValueError("sigma_a must be positive definite after PSD flooring")
    if calibrated_batch.any() and (
        np.linalg.eigvalsh(psi_batch[calibrated_batch]) <= 0.0
    ).any():
        raise ValueError("calibrated psi matrices must be positive definite")
    return sigma_a_batch, psi_batch, residual_batch, calibrated_batch


def _prepare_interval_inputs(
    points: np.ndarray,
    sigma_future: np.ndarray,
    posterior_covariances: np.ndarray,
    multipliers: np.ndarray,
    participant_index: np.ndarray | None,
    variance_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    point_array = _vector_batch(points, "points")
    if (point_array < 0.0).any():
        raise ValueError("point predictions must be nonnegative in log1p space")
    future_batch = _matrix_batch(sigma_future, "sigma_future")
    posterior_batch = _matrix_batch(
        posterior_covariances, "posterior_covariances"
    )
    multiplier_batch = _vector_batch(multipliers, "multipliers")
    if (multiplier_batch < 0.0).any():
        raise ValueError("interval multipliers must be nonnegative")
    size = max(len(future_batch), len(posterior_batch), len(multiplier_batch))
    future_batch = _broadcast_first_axis(future_batch, size, "sigma_future")
    posterior_batch = _broadcast_first_axis(
        posterior_batch, size, "posterior_covariances"
    )
    multiplier_batch = _broadcast_first_axis(
        multiplier_batch, size, "multipliers"
    )
    if participant_index is None:
        if size == 1:
            indices = np.zeros(len(point_array), dtype=np.int64)
        elif size == len(point_array):
            indices = np.arange(size, dtype=np.int64)
        else:
            raise ValueError(
                "participant_index is required when rows do not align one-to-one "
                "with covariance batches"
            )
    else:
        raw_indices = np.asarray(participant_index)
        if raw_indices.ndim != 1 or len(raw_indices) != len(point_array):
            raise ValueError("participant_index must have one entry per prediction row")
        if not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError("participant_index must contain integers")
        indices = raw_indices.astype(np.int64, copy=False)
        if (indices < 0).any() or (indices >= size).any():
            raise ValueError(
                "participant_index references an unavailable covariance batch"
            )
    floor = float(variance_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("variance_floor must be finite and positive")
    predictive = future_batch + posterior_batch
    if (np.diagonal(predictive, axis1=-2, axis2=-1) < -1e-12).any():
        raise ValueError("predictive covariance has a negative marginal variance")
    return (
        point_array,
        future_batch,
        posterior_batch,
        multiplier_batch,
        indices,
        floor,
    )


def _numpy_cholesky_solve(cholesky: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    forward = np.linalg.solve(cholesky, right_hand_side)
    return np.linalg.solve(np.swapaxes(cholesky, -1, -2), forward)


@dataclass(frozen=True)
class NumpyLayer2Backend:
    """Canonical SciPy-free NumPy reference backend."""

    fallback_from: str | None = None
    fallback_reason: str | None = None
    name = "numpy"
    device = "cpu"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "numpy",
            "device": "cpu",
            "dtype": "float64",
            "deterministic_operations": True,
            "fallback_from": self.fallback_from,
            "fallback_reason": self.fallback_reason,
        }

    def posterior_batch(
        self,
        sigma_a: np.ndarray,
        psi: np.ndarray,
        residual_means: np.ndarray,
        calibrated: bool | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        sigma_a_batch, psi_batch, residual_batch, calibrated_batch = (
            _prepare_posterior_inputs(sigma_a, psi, residual_means, calibrated)
        )
        means = np.zeros((len(calibrated_batch), DIMENSION), dtype=np.float64)
        # Copying sigma_a before any solve guarantees exact K=0 behavior.
        covariances = np.array(sigma_a_batch, dtype=np.float64, copy=True)
        if not calibrated_batch.any():
            return means, covariances
        selected_a = sigma_a_batch[calibrated_batch]
        system = selected_a + psi_batch[calibrated_batch]
        try:
            cholesky = np.linalg.cholesky(system)
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "sigma_a + psi must be positive definite for posterior solves"
            ) from error
        solved_residual = _numpy_cholesky_solve(
            cholesky, residual_batch[calibrated_batch, :, None]
        )
        solved_a = _numpy_cholesky_solve(cholesky, selected_a)
        selected_means = np.matmul(selected_a, solved_residual)[..., 0]
        selected_covariances = selected_a - np.matmul(selected_a, solved_a)
        selected_covariances = (
            selected_covariances
            + np.swapaxes(selected_covariances, -1, -2)
        ) / 2.0
        if not np.isfinite(selected_means).all() or not np.isfinite(
            selected_covariances
        ).all():
            raise FloatingPointError("posterior solve produced non-finite values")
        means[calibrated_batch] = selected_means
        covariances[calibrated_batch] = selected_covariances
        return means, covariances

    def interval_batch(
        self,
        points: np.ndarray,
        sigma_future: np.ndarray,
        posterior_covariances: np.ndarray,
        participant_index: np.ndarray | None,
        multipliers: np.ndarray,
        *,
        variance_floor: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray]:
        (
            point_array,
            future_batch,
            posterior_batch,
            multiplier_batch,
            indices,
            floor,
        ) = _prepare_interval_inputs(
            points,
            sigma_future,
            posterior_covariances,
            multipliers,
            participant_index,
            variance_floor,
        )
        predictive = future_batch + posterior_batch
        variances = np.diagonal(predictive, axis1=-2, axis2=-1)[indices]
        half_width = multiplier_batch[indices] * np.sqrt(np.maximum(variances, floor))
        lower = np.maximum(point_array - half_width, 0.0)
        upper = point_array + half_width
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise FloatingPointError("interval calculation produced non-finite values")
        return lower, upper


def _import_torch() -> Any:
    """Import torch only when the optional backend is requested."""

    return importlib.import_module("torch")


def _resolve_torch_device(
    torch_module: Any,
    requested_device: str,
    *,
    allow_device_fallback: bool,
) -> tuple[str, str | None]:
    requested = str(requested_device).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("PyTorch device must be one of: auto, cpu, cuda")
    cuda_available = bool(torch_module.cuda.is_available())
    if requested == "auto":
        return ("cuda", None) if cuda_available else ("cpu", None)
    if requested == "cuda" and not cuda_available:
        if allow_device_fallback:
            return "cpu", "CUDA was requested but is unavailable"
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested, None


class TorchLayer2Backend:
    """Lazy optional PyTorch backend with the same NumPy-facing contract."""

    name = "torch"

    def __init__(
        self,
        *,
        device: str = "auto",
        allow_device_fallback: bool = False,
        seed: int = 202407,
    ) -> None:
        torch_module = _import_torch()
        resolved, fallback_reason = _resolve_torch_device(
            torch_module,
            device,
            allow_device_fallback=allow_device_fallback,
        )
        self._torch = torch_module
        self.requested_device = str(device).lower()
        self.device = resolved
        self.device_fallback_reason = fallback_reason
        self.seed = int(seed)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "torch",
            "torch_version": str(self._torch.__version__),
            "requested_device": self.requested_device,
            "device": self.device,
            "device_fallback_reason": self.device_fallback_reason,
            "dtype": "float64",
            "seed": self.seed,
            "deterministic_operations": True,
        }

    def posterior_batch(
        self,
        sigma_a: np.ndarray,
        psi: np.ndarray,
        residual_means: np.ndarray,
        calibrated: bool | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        sigma_a_batch, psi_batch, residual_batch, calibrated_batch = (
            _prepare_posterior_inputs(sigma_a, psi, residual_means, calibrated)
        )
        means = np.zeros((len(calibrated_batch), DIMENSION), dtype=np.float64)
        # Preserve the input bytes for K=0 rather than round-tripping through a device.
        covariances = np.array(sigma_a_batch, dtype=np.float64, copy=True)
        if not calibrated_batch.any():
            return means, covariances
        torch = self._torch
        with torch.no_grad():
            selected_a = torch.as_tensor(
                sigma_a_batch[calibrated_batch],
                dtype=torch.float64,
                device=self.device,
            )
            selected_psi = torch.as_tensor(
                psi_batch[calibrated_batch],
                dtype=torch.float64,
                device=self.device,
            )
            selected_residual = torch.as_tensor(
                residual_batch[calibrated_batch, :, None],
                dtype=torch.float64,
                device=self.device,
            )
            cholesky, info = torch.linalg.cholesky_ex(selected_a + selected_psi)
            if bool(torch.any(info != 0).item()):
                raise np.linalg.LinAlgError(
                    "sigma_a + psi must be positive definite for posterior solves"
                )
            solved_residual = torch.cholesky_solve(selected_residual, cholesky)
            solved_a = torch.cholesky_solve(selected_a, cholesky)
            selected_means = torch.matmul(selected_a, solved_residual)[..., 0]
            selected_covariances = selected_a - torch.matmul(selected_a, solved_a)
            selected_covariances = (
                selected_covariances + selected_covariances.transpose(-1, -2)
            ) / 2.0
            means[calibrated_batch] = selected_means.detach().cpu().numpy()
            covariances[calibrated_batch] = (
                selected_covariances.detach().cpu().numpy()
            )
        if not np.isfinite(means).all() or not np.isfinite(covariances).all():
            raise FloatingPointError("posterior solve produced non-finite values")
        return means, covariances

    def interval_batch(
        self,
        points: np.ndarray,
        sigma_future: np.ndarray,
        posterior_covariances: np.ndarray,
        participant_index: np.ndarray | None,
        multipliers: np.ndarray,
        *,
        variance_floor: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray]:
        (
            point_array,
            future_batch,
            posterior_batch,
            multiplier_batch,
            indices,
            floor,
        ) = _prepare_interval_inputs(
            points,
            sigma_future,
            posterior_covariances,
            multipliers,
            participant_index,
            variance_floor,
        )
        torch = self._torch
        with torch.no_grad():
            point_tensor = torch.as_tensor(
                point_array, dtype=torch.float64, device=self.device
            )
            future_tensor = torch.as_tensor(
                future_batch, dtype=torch.float64, device=self.device
            )
            posterior_tensor = torch.as_tensor(
                posterior_batch, dtype=torch.float64, device=self.device
            )
            multiplier_tensor = torch.as_tensor(
                multiplier_batch, dtype=torch.float64, device=self.device
            )
            index_tensor = torch.as_tensor(
                indices, dtype=torch.long, device=self.device
            )
            predictive = future_tensor + posterior_tensor
            variances = torch.diagonal(
                predictive, dim1=-2, dim2=-1
            ).index_select(0, index_tensor)
            half_width = multiplier_tensor.index_select(0, index_tensor) * torch.sqrt(
                torch.clamp(variances, min=floor)
            )
            lower = torch.clamp(point_tensor - half_width, min=0.0)
            upper = point_tensor + half_width
            lower_array = lower.detach().cpu().numpy()
            upper_array = upper.detach().cpu().numpy()
        if not np.isfinite(lower_array).all() or not np.isfinite(upper_array).all():
            raise FloatingPointError("interval calculation produced non-finite values")
        return lower_array, upper_array


def make_layer2_backend(
    backend: str,
    *,
    device: str = "auto",
    allow_backend_fallback: bool = False,
    allow_device_fallback: bool = False,
    seed: int = 202407,
) -> Layer2Backend:
    """Create a backend without making PyTorch a runtime requirement.

    Backend fallback and CUDA-to-CPU device fallback are separate, explicit
    decisions so manifests can distinguish a missing optional dependency from
    an unavailable accelerator.
    """

    name = str(backend).lower()
    if name == "numpy":
        return NumpyLayer2Backend()
    if name != "torch":
        raise ValueError("Layer-2 backend must be 'numpy' or 'torch'")
    try:
        return TorchLayer2Backend(
            device=device,
            allow_device_fallback=allow_device_fallback,
            seed=seed,
        )
    except (ImportError, ModuleNotFoundError) as error:
        if not allow_backend_fallback:
            raise
        return NumpyLayer2Backend(
            fallback_from="torch",
            fallback_reason=f"optional PyTorch import failed: {type(error).__name__}",
        )


def create_backend(
    name: str, device: str | None = None
) -> Layer2Backend:
    """Stable integration factory; no silent backend or device fallback."""

    return make_layer2_backend(
        name,
        device="auto" if device is None else str(device),
        allow_backend_fallback=False,
        allow_device_fallback=False,
    )


def synchronize_backend(backend: Layer2Backend) -> None:
    """Synchronize CUDA for unbiased external timing; otherwise a no-op."""

    if isinstance(backend, TorchLayer2Backend) and backend.device == "cuda":
        backend._torch.cuda.synchronize()


__all__ = [
    "ALLOWED_BUDGETS",
    "DIMENSION",
    "Layer2Backend",
    "NumpyLayer2Backend",
    "PARITY_ATOL",
    "PARITY_RTOL",
    "TorchLayer2Backend",
    "create_backend",
    "make_layer2_backend",
    "synchronize_backend",
]
