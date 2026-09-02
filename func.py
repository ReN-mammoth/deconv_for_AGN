#!/usr/bin/env python3
"""Objective, gradient, and proximal functions for AGN deconvolution."""

from __future__ import annotations

import numpy as np

from deconv_utils import PSFOperator, apply_vtv


def initialize_components(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Initialize I_sm to zero and I_sp to the observed image."""

    I_sp = np.array(Y, dtype=np.float64, copy=True)
    I_sm = np.zeros_like(I_sp)
    return I_sm, I_sp


def data_fidelity(Y: np.ndarray, I_sm: np.ndarray, I_sp: np.ndarray,
                  sigma2: np.ndarray, T: PSFOperator) -> float:
    """Evaluate the Gaussian data-fidelity term."""

    residual = T.forward(I_sm + I_sp) - Y
    return float(0.5 * np.vdot(residual, residual / sigma2).real)


def smoothness_penalty(I_sm: np.ndarray, alpha_sm: float) -> float:
    """Evaluate alpha_sm times the smoothness term in Eq. (12)."""

    return float(alpha_sm * np.vdot(I_sm, apply_vtv(I_sm)).real)


def sparsity_penalty(I_sp: np.ndarray, alpha_sp: float) -> float:
    """Evaluate alpha_sp times the L1 norm in Eq. (12)."""

    return float(alpha_sp * np.sum(np.abs(I_sp), dtype=np.float64))


def balance_penalty(I_sm: np.ndarray, I_sp: np.ndarray, alpha_bl: float) -> float:
    """Evaluate the point-source balance term in Eq. (12)."""

    return float(alpha_bl * np.vdot(I_sm, I_sp).real)


def differentiable_objective(Y: np.ndarray, I_sm: np.ndarray, I_sp: np.ndarray, sigma2: np.ndarray,
                             T: PSFOperator, alpha_sm: float, alpha_bl: float) -> float:
    """Evaluate the differentiable objective in Eq. (13)."""

    return (data_fidelity(Y, I_sm, I_sp, sigma2, T) + smoothness_penalty(I_sm, alpha_sm)
            + balance_penalty(I_sm, I_sp, alpha_bl))


def objective_function(Y: np.ndarray, I_sm: np.ndarray, I_sp: np.ndarray, sigma2: np.ndarray,
                       T: PSFOperator, alpha_sm: float, alpha_sp: float, alpha_bl: float) -> float:
    """Evaluate the full objective in Eq. (12)."""

    return (differentiable_objective(Y, I_sm, I_sp, sigma2, T, alpha_sm, alpha_bl)
            + sparsity_penalty(I_sp, alpha_sp))


def calculate_data_gradient(Y: np.ndarray, I_sm: np.ndarray, I_sp: np.ndarray,
                            sigma2: np.ndarray, T: PSFOperator) -> np.ndarray:
    """Evaluate the data-term gradient shared by I_sm and I_sp."""

    residual = T.forward(I_sm + I_sp) - Y
    return T.adjoint(residual / sigma2)


def deriv_smooth_img(I_sm: np.ndarray, I_sp: np.ndarray, data_gradient: np.ndarray,
                     alpha_sm: float, alpha_bl: float) -> np.ndarray:
    """Evaluate the gradient of Eq. (13) with respect to I_sm."""

    gradient = np.array(data_gradient, dtype=np.float64, copy=True)
    if alpha_sm != 0.0:
        gradient += 2.0 * alpha_sm * apply_vtv(I_sm)
    if alpha_bl != 0.0:
        gradient += alpha_bl * I_sp
    return gradient


def deriv_sparse_img(I_sm: np.ndarray, I_sp: np.ndarray, data_gradient: np.ndarray,
                     alpha_bl: float) -> np.ndarray:
    """Evaluate the gradient of Eq. (13) with respect to I_sp."""

    gradient = np.array(data_gradient, dtype=np.float64, copy=True)
    if alpha_bl != 0.0:
        gradient += alpha_bl * I_sm
    return gradient


def proximal_update(I_sm: np.ndarray, I_sp: np.ndarray, grad_I_sm: np.ndarray, grad_I_sp: np.ndarray,
                    alpha_sp: float, L: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply the non-negative proximal update in Eqs. (18) and (19)."""

    I_sm_new = np.maximum(I_sm - grad_I_sm / L, 0.0)
    I_sp_new = np.maximum(I_sp - grad_I_sp / L - alpha_sp / L, 0.0)
    return I_sm_new, I_sp_new


def quadratic_upper_bound(f_current: float, I_sm: np.ndarray, I_sp: np.ndarray,
                          I_sm_new: np.ndarray, I_sp_new: np.ndarray, grad_I_sm: np.ndarray,
                          grad_I_sp: np.ndarray, L: float) -> float:
    """Evaluate the quadratic upper bound used in Eq. (20)."""

    delta_I_sm = I_sm_new - I_sm
    delta_I_sp = I_sp_new - I_sp
    linear = np.vdot(grad_I_sm, delta_I_sm).real + np.vdot(grad_I_sp, delta_I_sp).real
    quadratic = 0.5 * L * (np.vdot(delta_I_sm, delta_I_sm).real + np.vdot(delta_I_sp, delta_I_sp).real)
    return float(f_current + linear + quadratic)
