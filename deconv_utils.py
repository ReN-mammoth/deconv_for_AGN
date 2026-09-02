#!/usr/bin/env python3
"""Shared numerical utilities for AGN deconvolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.fft import irfftn, next_fast_len, rfftn


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolverResult:
    """Store the final solver state and convergence histories."""

    I_sm: np.ndarray
    I_sp: np.ndarray
    hellinger_history: list[float]
    objective_history: list[float]
    n_iterations: int
    converged: bool
    interrupted: bool


def prepare_psf(psf: np.ndarray) -> np.ndarray:
    """Validate a PSF image and normalize its total response to one."""

    psf = np.array(psf, dtype=np.float64, copy=True)
    if psf.ndim != 2 or 0 in psf.shape:
        raise ValueError("The PSF must be a non-empty two-dimensional image.")
    if not np.all(np.isfinite(psf)):
        raise ValueError("The PSF contains non-finite pixels.")

    psf_sum = float(np.sum(psf, dtype=np.float64))
    if not np.isfinite(psf_sum) or psf_sum <= 0.0:
        raise ValueError("The PSF sum must be finite and positive.")
    if not np.isclose(psf_sum, 1.0, rtol=1.0e-8, atol=1.0e-12):
        LOGGER.warning("PSF sum is %.16e; normalizing it to one.", psf_sum)
    return psf / psf_sum


class PSFOperator:
    """Apply cached FFT forward and exact adjoint PSF convolutions."""

    def __init__(self, psf: np.ndarray, image_shape: Sequence[int]) -> None:
        psf = prepare_psf(psf)
        if len(image_shape) != 2:
            raise ValueError("image_shape must contain two dimensions.")

        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        if self.image_shape[0] <= 0 or self.image_shape[1] <= 0:
            raise ValueError("image_shape dimensions must be positive.")

        self.psf = psf
        self.psf_shape = psf.shape
        full_shape = tuple(n_image + n_psf - 1 for n_image, n_psf in zip(self.image_shape, self.psf_shape))
        self.fft_shape = tuple(next_fast_len(length) for length in full_shape)
        self._psf_fft = rfftn(self.psf, s=self.fft_shape)
        self._psf_flip_fft = rfftn(self.psf[::-1, ::-1], s=self.fft_shape)
        self._forward_start = tuple((length - 1) // 2 for length in self.psf_shape)
        self._adjoint_start = tuple(length // 2 for length in self.psf_shape)

    def _convolve(self, image: np.ndarray, kernel_fft: np.ndarray, start: tuple[int, int]) -> np.ndarray:
        image = np.asarray(image, dtype=np.float64)
        if image.shape != self.image_shape:
            raise ValueError(f"Image shape must be {self.image_shape}, not {image.shape}.")
        if not np.all(np.isfinite(image)):
            raise ValueError("The convolution input contains non-finite pixels.")

        full = irfftn(rfftn(image, s=self.fft_shape) * kernel_fft, s=self.fft_shape)
        y_start, x_start = start
        ny, nx = self.image_shape
        return np.array(full[y_start:y_start + ny, x_start:x_start + nx], dtype=np.float64, copy=True)

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Apply the PSF response with scipy.signal same-mode alignment."""

        return self._convolve(image, self._psf_fft, self._forward_start)

    def adjoint(self, image: np.ndarray) -> np.ndarray:
        """Apply the exact transpose of the forward PSF response."""

        return self._convolve(image, self._psf_flip_fft, self._adjoint_start)


def build_psf_operator(psf: np.ndarray, image_shape: Sequence[int]) -> PSFOperator:
    """Build a cached PSF forward-adjoint operator pair."""

    return PSFOperator(psf, image_shape)


def apply_vtv(I_sm: np.ndarray) -> np.ndarray:
    """Apply the two-dimensional finite-difference V-transpose-V operator."""

    I_sm = np.asarray(I_sm, dtype=np.float64)
    if I_sm.ndim != 2:
        raise ValueError("I_sm must be a two-dimensional image.")

    result = np.zeros_like(I_sm)
    difference_x = I_sm[:, :-1] - I_sm[:, 1:]
    difference_y = I_sm[:-1, :] - I_sm[1:, :]
    result[:, :-1] += difference_x
    result[:, 1:] -= difference_x
    result[:-1, :] += difference_y
    result[1:, :] -= difference_y
    return result


def hellinger_distance(previous: np.ndarray, current: np.ndarray, eps: float = 1.0e-12) -> float:
    """Calculate the Hellinger distance between normalized non-negative images."""

    previous = np.asarray(previous, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if previous.shape != current.shape:
        raise ValueError("Hellinger-distance inputs must have identical shapes.")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        raise ValueError("Hellinger-distance inputs must be finite.")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive.")

    p = np.clip(previous.ravel(), 0.0, None)
    q = np.clip(current.ravel(), 0.0, None)
    p_sum = float(np.sum(p, dtype=np.float64))
    q_sum = float(np.sum(q, dtype=np.float64))

    if p_sum <= eps and q_sum <= eps:
        return 0.0
    if p_sum <= eps or q_sum <= eps:
        return 1.0

    p /= p_sum
    q /= q_sum
    distance_squared = 0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, dtype=np.float64)
    return float(np.sqrt(max(distance_squared, 0.0)))


def calculate_rmse(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Calculate the root mean squared error between equal-shaped arrays."""

    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if prediction.shape != reference.shape:
        raise ValueError("RMSE inputs must have identical shapes.")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(reference)):
        raise ValueError("RMSE inputs must be finite.")
    difference = prediction - reference
    return float(np.sqrt(np.mean(difference * difference, dtype=np.float64)))


def center_crop_or_pad(image: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Center-crop or zero-pad a two-dimensional image to a target shape."""

    image = np.asarray(image)
    if image.ndim != 2 or len(target_shape) != 2:
        raise ValueError("image and target_shape must be two-dimensional.")

    target_shape = (int(target_shape[0]), int(target_shape[1]))
    if target_shape[0] <= 0 or target_shape[1] <= 0:
        raise ValueError("target_shape dimensions must be positive.")

    copy_shape = (min(image.shape[0], target_shape[0]), min(image.shape[1], target_shape[1]))
    source_start = ((image.shape[0] - copy_shape[0]) // 2, (image.shape[1] - copy_shape[1]) // 2)
    target_start = ((target_shape[0] - copy_shape[0]) // 2, (target_shape[1] - copy_shape[1]) // 2)
    source_slice = tuple(slice(start, start + size) for start, size in zip(source_start, copy_shape))
    target_slice = tuple(slice(start, start + size) for start, size in zip(target_start, copy_shape))

    result = np.zeros(target_shape, dtype=image.dtype)
    result[target_slice] = image[source_slice]
    return result


def normalize_for_comparison(image: np.ndarray) -> np.ndarray:
    """Subtract the median and normalize by the maximum absolute value."""

    image = np.array(image, dtype=np.float64, copy=True)
    if image.ndim != 2 or not np.all(np.isfinite(image)):
        raise ValueError("The comparison image must be finite and two-dimensional.")
    image -= np.median(image)
    maximum = float(np.max(np.abs(image)))
    if maximum > 0.0:
        image /= maximum
    return image


def validate_solver_inputs(Y: np.ndarray, sigma2: np.ndarray, alpha_sm: float, alpha_sp: float,
                           alpha_bl: float, n_iter: int, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Validate solver arrays and scalar parameters."""

    Y = np.asarray(Y, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    if Y.ndim != 2 or sigma2.ndim != 2 or Y.shape != sigma2.shape:
        raise ValueError("Y and sigma2 must be equal-shaped two-dimensional images.")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y contains non-finite pixels.")
    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
        raise ValueError("sigma2 must contain finite positive values.")

    for name, value in (("alpha_sm", alpha_sm), ("alpha_sp", alpha_sp), ("alpha_bl", alpha_bl)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if not isinstance(n_iter, (int, np.integer)) or n_iter <= 0:
        raise ValueError("n_iter must be a positive integer.")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive.")
    return Y, sigma2
