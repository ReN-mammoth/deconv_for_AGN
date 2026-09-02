#!/usr/bin/env python3
"""Run two-component PSF deconvolution for AGN images."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import convolve

from deconv_utils import build_psf_operator, calculate_rmse, center_crop_or_pad, normalize_for_comparison, prepare_psf
from ista import run_ista
from ista_backtrack import run_ista_backtrack


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedInputs:
    """Store validated arrays in normalized flux units."""

    Y: np.ndarray
    sigma2: np.ndarray
    psf: np.ndarray
    validation: np.ndarray | None
    scale: float


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Two-component PSF deconvolution for AGN images.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--obs", type=Path, required=True, help="Observed FITS image.")
    parser.add_argument("--var", type=Path, required=True, help="Variance FITS image.")
    parser.add_argument("--var_ext", type=int, default=None, help="Zero-based HDU index for the variance image.")
    parser.add_argument("--psf", type=Path, required=True, help="PSF FITS image.")
    parser.add_argument("--val", type=Path, default=None, help="Optional validation FITS image.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--out_head", required=True, help="Output filename prefix.")
    parser.add_argument("--alpha_sm", type=float, default=1.0, help="Smoothness regularization coefficient.")
    parser.add_argument("--alpha_sp", type=float, default=1.0, help="Sparsity regularization coefficient.")
    parser.add_argument("--alpha_bl", type=float, default=0.0, help="Point-source balance coefficient.")
    parser.add_argument("--NITE", type=int, default=100, help="Maximum iterations.")
    parser.add_argument("--eps", type=float, default=1.0e-12, help="Hellinger-distance convergence tolerance.")
    parser.add_argument("--solver", choices=("backtrack", "ista"), default="backtrack", help="Optimization solver.")
    parser.add_argument("--lip_const", type=float, default=None,
                        help="Fixed Lipschitz constant required by the ISTA solver.")
    parser.add_argument("--track_objective", action="store_true", help="Save the objective value at every iteration.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate command-line arguments."""

    for path in (args.obs, args.var, args.psf):
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    if args.val is not None and not args.val.is_file():
        parser.error(f"Validation file does not exist: {args.val}")
    if args.var_ext is not None and args.var_ext < 0:
        parser.error("--var_ext must be a non-negative integer.")

    if not args.out_head or Path(args.out_head).name != args.out_head:
        parser.error("--out_head must be a filename prefix without directories.")
    if args.NITE <= 0:
        parser.error("--NITE must be positive.")
    if not np.isfinite(args.eps) or args.eps <= 0.0:
        parser.error("--eps must be finite and positive.")

    for name in ("alpha_sm", "alpha_sp", "alpha_bl"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and non-negative.")

    if args.lip_const is not None:
        if not np.isfinite(args.lip_const) or args.lip_const <= 0.0:
            parser.error("--lip_const must be finite and positive.")
    if args.solver == "ista" and args.lip_const is None:
        parser.error("--lip_const is required when --solver ista is selected.")
    if args.solver == "backtrack" and args.lip_const is not None:
        parser.error("--lip_const is only used with --solver ista.")


def load_fits_image(path: Path, name: str, ext: int | None = None) -> np.ndarray:
    """Load a two-dimensional FITS image as float64."""

    image = np.asarray(fits.getdata(path) if ext is None else fits.getdata(path, ext=ext), dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional: {path}")
    return image


def require_finite(image: np.ndarray, name: str) -> None:
    """Reject non-finite image pixels."""

    invalid = ~np.isfinite(image)
    if np.any(invalid):
        count = int(np.count_nonzero(invalid))
        raise ValueError(f"{name} contains {count} non-finite pixels.")


def repair_variance_map(sigma2: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace invalid variance pixels with local eight-neighbor means."""

    repaired = np.array(sigma2, dtype=np.float64, copy=True)
    invalid = (~np.isfinite(repaired)) | (repaired <= 0.0)
    invalid_count = int(np.count_nonzero(invalid))

    if invalid_count == 0:
        return repaired, 0
    if np.all(invalid):
        raise ValueError("The variance image has no finite positive pixels.")

    kernel = np.ones((3, 3), dtype=np.float64)
    kernel[1, 1] = 0.0
    remaining = invalid.copy()

    while np.any(remaining):
        valid = ~remaining
        local_sum = convolve(np.where(valid, repaired, 0.0), kernel, mode="constant", cval=0.0)
        local_count = convolve(valid.astype(np.float64), kernel, mode="constant", cval=0.0)
        fillable = remaining & (local_count > 0.0)
        if not np.any(fillable):
            raise ValueError("The variance image could not be repaired locally.")
        repaired[fillable] = local_sum[fillable] / local_count[fillable]
        remaining[fillable] = False

    return repaired, invalid_count


def prepare_inputs(args: argparse.Namespace) -> PreparedInputs:
    """Load, validate, and normalize the input images."""

    Y = load_fits_image(args.obs, "Observed image")
    sigma2 = load_fits_image(args.var, "Variance image", ext=args.var_ext)
    psf = load_fits_image(args.psf, "PSF image")

    if sigma2.shape != Y.shape:
        raise ValueError(f"Observed and variance images must have identical shapes: {Y.shape} != {sigma2.shape}")

    require_finite(Y, "Observed image")
    sigma2, repaired_count = repair_variance_map(sigma2)
    if repaired_count:
        LOGGER.warning("Replaced %d invalid variance pixels with local means.", repaired_count)
    psf = prepare_psf(psf)

    scale = float(np.sum(Y, dtype=np.float64))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("The total observed flux must be finite and positive.")

    validation = None
    if args.val is not None:
        validation = load_fits_image(args.val, "Validation image")
        require_finite(validation, "Validation image")

    return PreparedInputs(Y=Y / scale, sigma2=sigma2 / (scale**2), psf=psf,
                          validation=validation, scale=scale)


def build_output_paths(out_dir: Path, out_head: str, *, partial: bool,
                       track_objective: bool) -> dict[str, Path]:
    """Build output paths for a completed or interrupted run."""

    stem = f"{out_head}_partial" if partial else out_head
    paths = {
        "reconstruction": out_dir / f"{stem}_rec_img.fits",
        "smooth": out_dir / f"{stem}_rec_smooth.fits",
        "sparse": out_dir / f"{stem}_rec_sparse.fits",
        "rmse": out_dir / f"{stem}_rmse_summary.txt",
        "hellinger_plot": out_dir / f"{stem}_hellinger_plot.png",
        "hellinger_csv": out_dir / f"{stem}_hellinger_history.csv",
    }
    if track_objective:
        paths["objective_csv"] = out_dir / f"{stem}_objective_history.csv"
    return paths


def check_output_paths(args: argparse.Namespace) -> None:
    """Prevent accidental overwriting before the solver starts."""

    if args.overwrite:
        return

    paths: set[Path] = set()
    for partial in (False, True):
        paths.update(build_output_paths(args.out_dir, args.out_head, partial=partial,
                                        track_objective=args.track_objective).values())
    existing = sorted(path for path in paths if path.exists())
    if existing:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(f"Output files already exist. Use --overwrite to replace them:\n{formatted}")


def run_solver(args: argparse.Namespace, inputs: PreparedInputs, T):
    """Run the selected solver with normalized coefficients."""

    alpha_sm = args.alpha_sm / inputs.scale
    alpha_sp = args.alpha_sp / inputs.scale
    alpha_bl = args.alpha_bl / inputs.scale

    common = {
        "Y": inputs.Y,
        "sigma2": inputs.sigma2,
        "T": T,
        "alpha_sm": alpha_sm,
        "alpha_sp": alpha_sp,
        "alpha_bl": alpha_bl,
        "n_iter": args.NITE,
        "tol": args.eps,
        "track_objective": args.track_objective,
        "logger": None,
    }

    LOGGER.info("scale = %.16e", inputs.scale)
    LOGGER.info("alpha_sm: input=%.16e internal=%.16e", args.alpha_sm, alpha_sm)
    LOGGER.info("alpha_sp: input=%.16e internal=%.16e", args.alpha_sp, alpha_sp)
    LOGGER.info("alpha_bl: input=%.16e internal=%.16e", args.alpha_bl, alpha_bl)

    if args.solver == "ista":
        return run_ista(**common, lip_const=args.lip_const)
    return run_ista_backtrack(**common)


def calculate_output_metrics(inputs: PreparedInputs, T, I_rec: np.ndarray) -> tuple[float, float | None]:
    """Calculate observed-image and optional validation RMSE values."""

    rmse_observed = calculate_rmse(T.forward(I_rec), inputs.Y)

    rmse_validation = None
    if inputs.validation is not None:
        validation = center_crop_or_pad(inputs.validation, I_rec.shape)
        validation = np.clip(validation, 0.0, None)
        I_rec_normalized = normalize_for_comparison(I_rec)
        validation_normalized = normalize_for_comparison(validation)
        rmse_validation = calculate_rmse(I_rec_normalized, validation_normalized)

    return rmse_observed, rmse_validation


def make_fits_header(args: argparse.Namespace, inputs: PreparedInputs, result) -> fits.Header:
    """Create common metadata for normalized output images."""

    if result.interrupted:
        status = "INTERRUPTED"
    elif result.converged:
        status = "CONVERGED"
    else:
        status = "MAX_ITER"

    header = fits.Header()
    header["SCALE"] = (inputs.scale, "Observed-image flux normalization")
    header["ALPHASM"] = (args.alpha_sm, "Input alpha_sm")
    header["ALPHASP"] = (args.alpha_sp, "Input alpha_sp")
    header["ALPHABL"] = (args.alpha_bl, "Input alpha_bl")
    header["SOLVER"] = (args.solver, "Optimization solver")
    header["NITEREQ"] = (args.NITE, "Requested maximum iterations")
    header["NITER"] = (result.n_iterations, "Completed iterations")
    header["CONVERG"] = (result.converged, "Convergence criterion reached")
    header["INTRPT"] = (result.interrupted, "Run interrupted by user")
    header["STATUS"] = (status, "Solver termination status")
    header["NORMIMG"] = (True, "Image is stored in normalized flux units")
    header["VARHDU"] = (args.var_ext if args.var_ext is not None else "AUTO", "Variance-image HDU")
    if args.lip_const is not None:
        header["LIPCONST"] = (args.lip_const, "Fixed Lipschitz constant")
    return header


def save_fits_image(path: Path, image: np.ndarray, header: fits.Header, component: str, *, overwrite: bool) -> None:
    """Save a normalized image as float32 FITS data."""

    component_header = header.copy()
    component_header["COMP"] = (component, "Reconstructed component")
    fits.PrimaryHDU(data=np.asarray(image, dtype=np.float32), header=component_header).writeto(path, overwrite=overwrite)


def save_history_csv(path: Path, values: Sequence[float], column: str) -> None:
    """Save a one-dimensional iteration history."""

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("iteration", column))
        writer.writerows((iteration, f"{value:.16e}") for iteration, value in enumerate(values, start=1))


def save_hellinger_plot(path: Path, values: Sequence[float]) -> None:
    """Save the Hellinger-distance convergence plot."""

    values_array = np.asarray(values, dtype=np.float64)
    iterations = np.arange(1, values_array.size + 1)

    figure, axis = plt.subplots(figsize=(8.0, 6.0), dpi=150)
    axis.plot(iterations, values_array, linewidth=1.8)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Hellinger distance")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_rmse_summary(path: Path, rmse_observed: float, rmse_validation: float | None) -> None:
    """Save available RMSE measurements."""

    lines = []
    if rmse_validation is not None:
        lines.append(f"RMSE_rec_vs_val:      {rmse_validation:.16e}")
    lines.append(f"RMSE_convrec_vs_obs: {rmse_observed:.16e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(args: argparse.Namespace, inputs: PreparedInputs, result, T) -> dict[str, Path]:
    """Save reconstructed images, histories, and metrics."""

    paths = build_output_paths(args.out_dir, args.out_head, partial=result.interrupted,
                               track_objective=args.track_objective)
    I_rec = result.I_sm + result.I_sp
    rmse_observed, rmse_validation = calculate_output_metrics(inputs, T, I_rec)
    header = make_fits_header(args, inputs, result)

    save_fits_image(paths["reconstruction"], I_rec, header, "TOTAL", overwrite=args.overwrite)
    save_fits_image(paths["smooth"], result.I_sm, header, "SMOOTH", overwrite=args.overwrite)
    save_fits_image(paths["sparse"], result.I_sp, header, "SPARSE", overwrite=args.overwrite)
    save_rmse_summary(paths["rmse"], rmse_observed, rmse_validation)
    save_history_csv(paths["hellinger_csv"], result.hellinger_history, "hellinger_distance")
    save_hellinger_plot(paths["hellinger_plot"], result.hellinger_history)

    if args.track_objective:
        save_history_csv(paths["objective_csv"], result.objective_history, "objective")

    LOGGER.info("RMSE convolved reconstruction vs. observed = %.6e", rmse_observed)
    if rmse_validation is not None:
        LOGGER.info("RMSE reconstruction vs. validation = %.6e", rmse_validation)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line program."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(parser, args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    check_output_paths(args)

    inputs = prepare_inputs(args)
    T = build_psf_operator(inputs.psf, inputs.Y.shape)
    result = run_solver(args, inputs, T)
    paths = save_outputs(args, inputs, result, T)

    for path in paths.values():
        LOGGER.info("Saved: %s", path)

    # Output-image statistics in original flux units
    output_images = {
        "rec_img": result.I_sm + result.I_sp,
        "rec_smooth": result.I_sm,
        "rec_sparse": result.I_sp,
    }

    LOGGER.info("Output image statistics:")
    for name, image in output_images.items():
        minimum = float(np.min(image) * inputs.scale)
        maximum = float(np.max(image) * inputs.scale)
        mean = float(np.mean(image) * inputs.scale)

        LOGGER.info(
            "%-10s min=%.6e max=%.6e mean=%.6e",
            name,
            minimum,
            maximum,
            mean,
        )

    return 130 if result.interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())