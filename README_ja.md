# AGN母銀河のPSFデコンボリューション

[English](README.md) | [日本語](README_ja.md)

このコードは、PSFによるぼけを補正しながら、観測されたAGN画像を平滑な拡張成分 $I_{\rm sm}$ とスパースな点源成分 $I_{\rm sp}$ に分解します。

標準ソルバーはバックトラッキング（backtracking）付きISTAです。比較および学習用として、固定ステップのISTAも収録しています。手法の詳細は[Kawase et al. (2026)](https://arxiv.org/abs/2605.13735)を参照してください。

## 目的関数

このコードは、次の目的関数を最小化します。

$$
F = \frac{1}{2}\sum_v \frac{[T(I_{\rm sm}+I_{\rm sp})_v-Y_v]^2}{\sigma_v^2}
+ \alpha_{\rm sm}V(I_{\rm sm})
+ \alpha_{\rm sp}\lVert I_{\rm sp}\rVert_1
+ \alpha_{\rm bl}\sum_u I_{{\rm sm},u}I_{{\rm sp},u},
$$

制約条件は $I_{\rm sm}\geq0$ および $I_{\rm sp}\geq0$ です。

- $Y$：観測画像
- $\sigma^2$：分散画像
- $T$：PSF畳み込み演算子
- $\alpha_{\rm sm}$：平滑化係数
- $\alpha_{\rm sp}$：スパース性係数
- $\alpha_{\rm bl}$：点源バランス係数

## ファイル構成

| ファイル | 内容 |
|---|---|
| `AGN_deconv.py` | コマンドラインから実行するメインプログラム |
| `ista_backtrack.py` | バックトラッキング付きISTA |
| `ista.py` | 固定ステップのISTA |
| `func.py` | 目的関数、勾配および近接写像の関数 |
| `deconv_utils.py` | PSF演算子、評価指標および共通関数 |

## 必要環境

- Python 3.10以降
- NumPy
- SciPy
- Astropy
- Matplotlib

```bash
python3 -m pip install -r requirements.txt
```

## 入力データ

### 観測画像：`--obs`

有限値からなる2次元FITS画像を指定します。総フラックスは正でなければなりません。このコードでは、スカイ背景（sky background）の推定や減算は行いません。

### 分散画像：`--var`

観測画像と同じ形状を持つ2次元FITS画像を指定します。標準偏差 $\sigma$ ではなく、分散 $\sigma^2$ を入力してください。

非有限値または0以下のピクセルは、有効な8近傍ピクセルの平均値で置換されます。連続した無効領域は、その境界から内側に向かって反復的に補完されます。

分散マップがmulti-extension FITSの拡張HDUに格納されている場合は、`--var_ext`で0始まりのHDU番号を指定します。たとえば、HDU 3から分散マップを読み込む場合は`--var_ext 3`とします。`--var_ext`を省略した場合は、Astropyの既定のHDU選択を使用します。

### PSF画像：`--psf`

総和が正で、有限値からなる2次元FITS画像を指定します。PSFの総和が1でない場合は、自動的に総和1へ正規化されます。

このコードでは、PSFの再中心化、シフト、クリップは行いません。PSFのピクセルスケールと位置合わせは、あらかじめ観測画像に一致させてください。奇数・偶数のどちらの画像サイズのPSFにも対応しています。

### 参照画像：`--val`

任意で指定する参照FITS画像です。参照画像に対するRMSEを計算する前に、再構成画像と同じ形状になるよう中央で切り出すか、ゼロで埋められます。

## フラックスの正規化

メインプログラムでは、次の正規化を行います。

$$
\mathrm{scale}=\sum_vY_{v,\mathrm{original}}, \qquad
Y=\frac{Y_{\mathrm{original}}}{\mathrm{scale}}, \qquad
\sigma^2=\frac{\sigma^2_{\mathrm{original}}}{\mathrm{scale}^2}.
$$

入力された正則化係数は、内部で次のように変換されます。

$$
\alpha_{\rm sm}^{\rm internal}=\frac{\alpha_{\rm sm}}{\mathrm{scale}}, \qquad
\alpha_{\rm sp}^{\rm internal}=\frac{\alpha_{\rm sp}}{\mathrm{scale}}, \qquad
\alpha_{\rm bl}^{\rm internal}=\frac{\alpha_{\rm bl}}{\mathrm{scale}}.
$$

出力FITS画像は、正規化されたフラックス単位で保存されます。元の入力画像と同じフラックス単位へ戻すには、出力画像にFITSヘッダーの`SCALE`を掛けてください。

## 実行方法

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

ここで示した正則化係数は、特定のデータ構成に対する一例であり、すべてのデータに推奨される値ではありません。

分散マップがHDU 3に格納されている場合は、実行コマンドに`--var_ext 3`を追加してください。HDU番号を明示しなくても分散マップを正しく読み込めるFITSでは、このオプションは不要です。

参照画像を使用しない場合は、`--val`を省略してください。固定ステップのISTAを使用する場合は、`--solver ista --lip_const VALUE`を指定します。`VALUE`には、データに対して適切な有限かつ正のLipschitz定数を与えてください。一般に、値を大きくするとステップサイズが小さくなり安定しやすくなりますが、収束が大幅に遅くなる可能性があります。

## オプション

| オプション | 必須 | 既定値 | 内容 |
|---|---:|---:|---|
| `--obs` | はい | — | 観測FITS画像 |
| `--var` | はい | — | 分散FITS画像 |
| `--var_ext` | いいえ | None | 分散画像を読み込む0始まりのHDU番号 |
| `--psf` | はい | — | PSFのFITS画像 |
| `--val` | いいえ | None | 参照FITS画像 |
| `--out_dir` | はい | — | 出力ディレクトリ |
| `--out_head` | はい | — | 出力ファイル名の接頭辞 |
| `--alpha_sm` | いいえ | `1.0` | 平滑化係数 |
| `--alpha_sp` | いいえ | `1.0` | スパース性係数 |
| `--alpha_bl` | いいえ | `0.0` | 点源バランス係数 |
| `--NITE` | いいえ | `100` | 最大反復回数 |
| `--eps` | いいえ | `1.0e-12` | Hellinger距離の収束判定値 |
| `--solver` | いいえ | `backtrack` | `backtrack`または`ista`を選択 |
| `--lip_const` | `ista`のみ | None | 固定Lipschitz定数 |
| `--track_objective` | いいえ | Off | 各反復の目的関数値を保存 |
| `--overwrite` | いいえ | Off | 既存の出力を上書き |

## 出力ファイル

```text
{out_head}_rec_img.fits
{out_head}_rec_smooth.fits
{out_head}_rec_sparse.fits
{out_head}_rmse_summary.txt
{out_head}_hellinger_plot.png
{out_head}_hellinger_history.csv
```

`--track_objective`を指定すると、次のファイルも出力されます。

```text
{out_head}_objective_history.csv
```

- `rec_img`は $I_{\rm sm}+I_{\rm sp}$ です。
- `rec_smooth`は $I_{\rm sm}$ です。
- `rec_sparse`は $I_{\rm sp}$ です。
- 観測画像に対するRMSEは、正規化フラックス単位で $T(I_{\rm sm}+I_{\rm sp})$ と $Y$ を比較します。
- `--val`を指定した場合、参照画像に対するRMSEは中央値を減算して個別に正規化した画像同士を比較します。そのため、絶対フラックスの一致ではなく形態的な類似度を評価します。

FITSヘッダーには、`SCALE`、`VARHDU`、入力した正則化係数、ソルバー、反復回数および終了状態が記録されます。`VARHDU`には、指定したHDU番号、または`--var_ext`を省略した場合は`AUTO`が記録されます。

## 注意事項

- 内部計算には`float64`、出力FITS画像には`float32`を使用します。
- PSFのFFTはキャッシュされ、反復計算中に再利用されます。
- 随伴（adjoint）演算子は、奇数・偶数のどちらのサイズのPSFでも順方向（forward）演算子の厳密な転置です。
- 収束判定には、連続する再構成画像間のHellinger（ヘリンガー）距離を使用します。
- `Ctrl+C`を押すと、最後に完了または採用された反復結果が`{out_head}_partial_*`というファイル名で保存されます。
- `--overwrite`を指定しない限り、既存の出力ファイルは上書きされません。

## 引用

このコードを使用する場合は、[Kawase et al. (2026)](https://doi.org/10.48550/arXiv.2605.13735)を引用してください。

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

## ライセンス

本ソフトウェアはBSD 3-Clause Licenseの下で公開されています。詳細は[LICENSE](LICENSE)を参照してください。