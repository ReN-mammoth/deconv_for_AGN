# AGN Host-Galaxy PSF Deconvolution

[English](README.md) | [日本語](README_ja.md)

This code decomposes an observed AGN image into a smooth extended component $I_{\rm sm}$ and a sparse point-source component $I_{\rm sp}$ while correcting for PSF blurring.

The default solver is ISTA with backtracking. Fixed-step ISTA is also included for comparison and educational use. The method is described in [Kawase et al. (2026)](https://arxiv.org/abs/2605.13735).

## Objective function

The code minimizes

$$
F = \frac{1}{2}\sum_v \frac{[T(I_{\rm sm}+I_{\rm sp})_v-Y_v]^2}{\sigma_v^2}
+ \alpha_{\rm sm}V(I_{\rm sm})
+ \alpha_{\rm sp}\lVert I_{\rm sp}\rVert_1
+ \alpha_{\rm bl}\sum_u I_{{\rm sm},u}I_{{\rm sp},u},
$$

subject to $I_{\rm sm}\geq0$ and $I_{\rm sp}\geq0$.

- $Y$: observed image
- $\sigma^2$: variance image
- $T$: PSF convolution operator
- $\alpha_{\rm sm}$: smoothness coefficient
- $\alpha_{\rm sp}$: sparsity coefficient
- $\alpha_{\rm bl}$: point-source balance coefficient

## Files

| File | Description |
|---|---|
| `AGN_deconv.py` | Main command-line program |
| `ista_backtrack.py` | ISTA with backtracking |
| `ista.py` | Fixed-step ISTA |
| `func.py` | Objective, gradient, and proximal functions |
| `deconv_utils.py` | PSF operators, metrics, and shared utilities |

## Requirements

- Python 3.10 or later
- NumPy
- SciPy
- Astropy
- Matplotlib

```bash
python3 -m pip install numpy scipy astropy matplotlib
```

## Input data

### Observed image: `--obs`

A finite two-dimensional FITS image. Its total flux must be positive. The code does not estimate or subtract the sky background.

### Variance image: `--var`

A two-dimensional FITS image with the same shape as the observed image. It must contain the variance $\sigma^2$, not the standard deviation $\sigma$.

Non-finite or non-positive pixels are replaced with the mean of valid eight-connected neighbors. Contiguous invalid regions are filled iteratively from their boundaries.

If the variance map is stored in an extension of a multi-extension FITS file, specify its zero-based HDU index with `--var_ext`. For example, `--var_ext 3` reads the variance map from HDU 3. If `--var_ext` is omitted, the code uses Astropy's default HDU selection.

### PSF image: `--psf`

A finite two-dimensional FITS image with a positive total sum. The PSF is automatically normalized if its sum is not one.

The code does not recenter, shift, or clip the PSF. The PSF must already have the correct sampling and alignment for the observed image. Odd- and even-sized PSFs are supported.

### Validation image: `--val`

An optional reference FITS image. It is center-cropped or zero-padded to the reconstructed image shape before the validation RMSE is calculated.

## Flux normalization

The command-line program uses

$$
\mathrm{scale}=\sum_vY_{v,\mathrm{original}}, \qquad
Y=\frac{Y_{\mathrm{original}}}{\mathrm{scale}}, \qquad
\sigma^2=\frac{\sigma^2_{\mathrm{original}}}{\mathrm{scale}^2}.
$$

The input regularization coefficients are converted internally as

$$
\alpha_{\rm sm}^{\rm internal}=\frac{\alpha_{\rm sm}}{\mathrm{scale}}, \qquad
\alpha_{\rm sp}^{\rm internal}=\frac{\alpha_{\rm sp}}{\mathrm{scale}}, \qquad
\alpha_{\rm bl}^{\rm internal}=\frac{\alpha_{\rm bl}}{\mathrm{scale}}.
$$

Output FITS images are stored in normalized flux units. Multiply an output image by the `SCALE` value in its FITS header to return to the original input flux units.

## Run the code

```bash
python3 AGN_deconv.py \
  --obs AGNobs.fits \
  --var AGNvar.fits \
  --psf AGNpsf.fits \
  --val HSTval.fits \
  --out_dir results \
  --out_head AGN \
  --alpha_sm 4.0e12 \
  --alpha_sp 4.8e9 \
  --alpha_bl 1.0e12 \
  --NITE 1000 \
  --eps 1.0e-12 \
  --solver backtrack
```

These regularization coefficients are an example for one data configuration and are not universal recommended values.

If the variance map is stored in HDU 3, add `--var_ext 3` to the command. The option is unnecessary when the variance map is read correctly without an explicit HDU index.

To run without a validation image, omit `--val`. To use fixed-step ISTA, specify `--solver ista --lip_const VALUE`, where `VALUE` is a finite positive Lipschitz constant appropriate for the data. A larger value gives a smaller and more stable step, but can slow convergence substantially.

## Options

| Option | Required | Default | Description |
|---|---:|---:|---|
| `--obs` | Yes | — | Observed FITS image |
| `--var` | Yes | — | Variance FITS image |
| `--var_ext` | No | None | Zero-based HDU index for the variance image |
| `--psf` | Yes | — | PSF FITS image |
| `--val` | No | None | Validation FITS image |
| `--out_dir` | Yes | — | Output directory |
| `--out_head` | Yes | — | Output filename prefix |
| `--alpha_sm` | No | `1.0` | Smoothness coefficient |
| `--alpha_sp` | No | `1.0` | Sparsity coefficient |
| `--alpha_bl` | No | `0.0` | Point-source balance coefficient |
| `--NITE` | No | `100` | Maximum iterations |
| `--eps` | No | `1.0e-12` | Hellinger-distance tolerance |
| `--solver` | No | `backtrack` | `backtrack` or `ista` |
| `--lip_const` | For `ista` | None | Fixed Lipschitz constant |
| `--track_objective` | No | Off | Save the objective history |
| `--overwrite` | No | Off | Overwrite existing outputs |

## Outputs

```text
{out_head}_rec_img.fits
{out_head}_rec_smooth.fits
{out_head}_rec_sparse.fits
{out_head}_rmse_summary.txt
{out_head}_hellinger_plot.png
{out_head}_hellinger_history.csv
```

`--track_objective` additionally creates

```text
{out_head}_objective_history.csv
```

- `rec_img` is $I_{\rm sm}+I_{\rm sp}$.
- `rec_smooth` is $I_{\rm sm}$.
- `rec_sparse` is $I_{\rm sp}$.
- The observed-image RMSE compares $T(I_{\rm sm}+I_{\rm sp})$ with $Y$ in normalized flux units.
- If `--val` is supplied, the validation RMSE compares median-subtracted, independently normalized images and measures morphological similarity rather than absolute-flux agreement.

The FITS headers record `SCALE`, `VARHDU`, the input regularization coefficients, solver, iteration counts, and termination status. `VARHDU` contains the selected HDU index or `AUTO` when `--var_ext` was omitted.

## Notes

- Internal calculations use `float64`; output FITS images use `float32`.
- The PSF FFT is cached and reused during the iterations.
- The adjoint is the exact transpose of the forward operator for both odd- and even-sized PSFs.
- Convergence is determined from the Hellinger distance between consecutive reconstructed images.
- Pressing `Ctrl+C` saves the last completed or accepted iteration with `{out_head}_partial_*` filenames.
- Existing outputs are protected unless `--overwrite` is specified.

## Citation

Please cite [Kawase et al. (2026)](https://doi.org/10.48550/arXiv.2605.13735) when using this code.

```bibtex
@ARTICLE{2026arXiv260513735K,
       author = {{Kawase}, Ren and {Shibuya}, Takatoshi and {Matsuda}, Kazunori},
        title = "{A New PSF Deconvolution Algorithm: Simultaneous Spatial Resolution Enhancement and Point Source Removal for Morphological Analysis of AGN Host Galaxies}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics of Galaxies, Cosmology and Nongalactic Astrophysics, Instrumentation and Methods for Astrophysics},
         year = 2026,
        month = may,
          eid = {arXiv:2605.13735},
        pages = {arXiv:2605.13735},
          doi = {10.48550/arXiv.2605.13735},
archivePrefix = {arXiv},
       eprint = {2605.13735},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260513735K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
