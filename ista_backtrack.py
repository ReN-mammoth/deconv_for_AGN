#!/usr/bin/env python3
"""Backtracking ISTA solver for two-component AGN deconvolution."""

from __future__ import annotations

from logging import Logger

import numpy as np

from deconv_utils import PSFOperator, SolverResult, hellinger_distance, validate_solver_inputs
from func import calculate_data_gradient, deriv_smooth_img, deriv_sparse_img, differentiable_objective, initialize_components, proximal_update, sparsity_penalty


def search_L(Y: np.ndarray, sigma2: np.ndarray, T: PSFOperator, I_sm: np.ndarray, I_sp: np.ndarray,
             grad_I_sm: np.ndarray, grad_I_sp: np.ndarray, f_current: float, alpha_sm: float,
             alpha_sp: float, alpha_bl: float, L_previous: float, eta: float,
             max_backtracking: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Find an accepted proximal update and its Lipschitz constant."""

    L = L_previous
    for _ in range(max_backtracking):
        I_sm_new, I_sp_new = proximal_update(I_sm, I_sp, grad_I_sm, grad_I_sp, alpha_sp, L)
        if np.all(np.isfinite(I_sm_new)) and np.all(np.isfinite(I_sp_new)):
            f_new = differentiable_objective(Y, I_sm_new, I_sp_new, sigma2, T, alpha_sm, alpha_bl)
            delta_I_sm = I_sm_new - I_sm
            delta_I_sp = I_sp_new - I_sp
            linear = np.vdot(grad_I_sm, delta_I_sm).real + np.vdot(grad_I_sp, delta_I_sp).real
            quadratic = 0.5 * L * (np.vdot(delta_I_sm, delta_I_sm).real + np.vdot(delta_I_sp, delta_I_sp).real)
            upper_bound = f_current + linear + quadratic
            tolerance = 1.0e-12 * max(1.0, abs(f_current), abs(f_new), abs(upper_bound))
            if np.isfinite(f_new) and f_new <= upper_bound + tolerance:
                return I_sm_new, I_sp_new, float(f_new), float(L)

        L *= eta
        if not np.isfinite(L):
            break

    raise RuntimeError("Backtracking failed to find an acceptable Lipschitz constant.")


def run_ista_backtrack(Y: np.ndarray, sigma2: np.ndarray, T: PSFOperator, *, alpha_sm: float = 1.0,
                       alpha_sp: float = 1.0, alpha_bl: float = 0.0, n_iter: int = 100,
                       tol: float = 1.0e-12, L0: float = 1.0, eta: float = 1.1,
                       max_backtracking: int = 1000, track_objective: bool = False,
                       logger: Logger | None = None) -> SolverResult:
    """Run ISTA with backtracking for the Lipschitz constant."""

    Y, sigma2 = validate_solver_inputs(Y, sigma2, alpha_sm, alpha_sp, alpha_bl, n_iter, tol)
    if not np.isfinite(L0) or L0 <= 0.0:
        raise ValueError("L0 must be finite and positive.")
    if not np.isfinite(eta) or eta <= 1.0:
        raise ValueError("eta must be finite and greater than one.")
    if max_backtracking <= 0:
        raise ValueError("max_backtracking must be positive.")

    I_sm, I_sp = initialize_components(Y)
    I_prev = I_sm + I_sp
    L_previous = float(L0)
    hellinger_history: list[float] = []
    objective_history: list[float] = []
    converged = False
    interrupted = False

    progress_width = len(str(n_iter))

    try:
        f_current = differentiable_objective(Y, I_sm, I_sp, sigma2, T, alpha_sm, alpha_bl)
        if not np.isfinite(f_current):
            raise FloatingPointError("The initial differentiable objective is not finite.")

        for iteration in range(1, n_iter + 1):
            data_gradient = calculate_data_gradient(Y, I_sm, I_sp, sigma2, T)
            grad_I_sm = deriv_smooth_img(I_sm, I_sp, data_gradient, alpha_sm, alpha_bl)
            grad_I_sp = deriv_sparse_img(I_sm, I_sp, data_gradient, alpha_bl)
            I_sm_new, I_sp_new, f_new, L = search_L(Y, sigma2, T, I_sm, I_sp, grad_I_sm, grad_I_sp,
                                                    f_current, alpha_sm, alpha_sp, alpha_bl, L_previous,
                                                    eta, max_backtracking)

            I_new = I_sm_new + I_sp_new
            distance = hellinger_distance(I_prev, I_new)
            I_sm, I_sp, I_prev = I_sm_new, I_sp_new, I_new
            f_current, L_previous = f_new, L
            hellinger_history.append(distance)

            if track_objective:
                objective_history.append(f_new + sparsity_penalty(I_sp, alpha_sp))
            print(
                f"\rIteration {iteration:{progress_width}d}/{n_iter} "
                f"({100.0 * iteration / n_iter:5.1f}%)",
                end="",
                flush=True,
            )
            if distance < tol:
                converged = True
                break
    except KeyboardInterrupt:
        interrupted = True
        if logger is not None:
            logger.warning("Backtracking ISTA interrupted; returning the last accepted iteration.")

    finally:
        print()

    return SolverResult(I_sm=I_sm, I_sp=I_sp, hellinger_history=hellinger_history,
                        objective_history=objective_history, n_iterations=len(hellinger_history),
                        converged=converged, interrupted=interrupted)