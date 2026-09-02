#!/usr/bin/env python3
"""Fixed-step ISTA solver for two-component AGN deconvolution."""

from __future__ import annotations

from logging import Logger

import numpy as np

from deconv_utils import PSFOperator, SolverResult, hellinger_distance, validate_solver_inputs
from func import calculate_data_gradient, deriv_smooth_img, deriv_sparse_img, initialize_components, objective_function, proximal_update


def run_ista(Y: np.ndarray, sigma2: np.ndarray, T: PSFOperator, *, alpha_sm: float = 1.0,
             alpha_sp: float = 1.0, alpha_bl: float = 0.0, n_iter: int = 100, tol: float = 1.0e-12,
             lip_const: float, track_objective: bool = False, logger: Logger | None = None) -> SolverResult:
    """Run fixed-step ISTA with non-negative smooth and sparse components."""

    Y, sigma2 = validate_solver_inputs(Y, sigma2, alpha_sm, alpha_sp, alpha_bl, n_iter, tol)
    if not np.isfinite(lip_const) or lip_const <= 0.0:
        raise ValueError("lip_const must be finite and positive.")

    I_sm, I_sp = initialize_components(Y)
    I_prev = I_sm + I_sp
    hellinger_history: list[float] = []
    objective_history: list[float] = []
    converged = False
    interrupted = False

    progress_width = len(str(n_iter))

    try:
        for iteration in range(1, n_iter + 1):
            data_gradient = calculate_data_gradient(Y, I_sm, I_sp, sigma2, T)
            grad_I_sm = deriv_smooth_img(I_sm, I_sp, data_gradient, alpha_sm, alpha_bl)
            grad_I_sp = deriv_sparse_img(I_sm, I_sp, data_gradient, alpha_bl)
            I_sm_new, I_sp_new = proximal_update(I_sm, I_sp, grad_I_sm, grad_I_sp, alpha_sp, lip_const)

            if not np.all(np.isfinite(I_sm_new)) or not np.all(np.isfinite(I_sp_new)):
                raise FloatingPointError("ISTA produced a non-finite image update.")

            I_new = I_sm_new + I_sp_new
            distance = hellinger_distance(I_prev, I_new)
            I_sm, I_sp, I_prev = I_sm_new, I_sp_new, I_new
            hellinger_history.append(distance)

            if track_objective:
                objective_history.append(objective_function(Y, I_sm, I_sp, sigma2, T,
                                                            alpha_sm, alpha_sp, alpha_bl))
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
            logger.warning("ISTA interrupted; returning the last completed iteration.")

    finally:
        print()

    return SolverResult(I_sm=I_sm, I_sp=I_sp, hellinger_history=hellinger_history,
                        objective_history=objective_history, n_iterations=len(hellinger_history),
                        converged=converged, interrupted=interrupted)
