# BlackGEM_ML

A Python pipeline for extracting light curves from variable systems observed by BlackGEM, detecting photometric variability via dip-flagging, and applying unsupervised machine learning (K-Means clustering) to help identify candidate white dwarf eclipsing binary systems.

## Pipeline overview

```
bg_query.py  →  bg_stack_lcs.py  →  bg_lc_flagging.py  →  ml_features.csv  →  bg_ml.ipynb
   (query)         (stack LCs)          (flag dips)         (feature table)      (cluster)
```

| Stage | Script | Purpose |
|---|---|---|
| 1 | `bg_query.py` | Query BlackGEM Google Cloud BigQuery for variable source detections and image paths |
| 2 | `bg_stack_lcs.py` | Stack per-night FITS files into a single multi-night light curve per target |
| 3 | `bg_lc_flagging.py` | Iterative sigma-clip dip detector; computes variability metrics per filter |
| 4 | `bg_main.py` | Orchestrates stages 1–3, loops over all targets, writes `ml_features.csv` |
| 5 | `bg_ml.ipynb` | Loads features, scales, tunes K-Means (elbow + silhouette), produces cluster plots |

## Requirements

See `requirements.txt` for the full list of direct dependencies.

Install with:

```bash
python -m venv ENV_NAME
source ENV_NAME/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python bg_main.py --config example_config.ini
```

Then open and run `bg_ml.ipynb` once `ml_features.csv` has been generated.

### Configuration

Copy `example_config.ini` and fill in the paths for your environment:

```ini
[pipeline]
data_root          = /path/to/data/root
project_id         = your-gcp-project-id
variable_table_loc = dataset.variable_table
bg_detections_table = dataset.detections
bg_images_table    = dataset.images
wd_table           = /path/to/WD/catalogue.fits
hr_table           = /path/to/HR/catalogue.fits
```

`data_root` must contain a `lightcurves/` subdirectory with `*_LC.fits` files. The pipeline creates an `analysis/` subdirectory and writes `ml_features.csv` and `logs/` there.

## Module descriptions

| Module | Description |
|---|---|
| `bg_main.py` | Top-level runner; reads config, loops over LC files, calls dip-flagging, writes `ml_features.csv` |
| `bg_query.py` | BigQuery interface for the BlackGEM detections and images tables on Google Cloud |
| `bg_stack_lcs.py` | Combines per-night BlackGEM observations into a multi-night FITS LC per target |
| `bg_lc_flagging.py` | Iterative sigma-clip dip detector; returns variability metrics (score, n_dips, depth, SNR, etc.) |
| `bg_analysis.py` | BLS and Lomb-Scargle period search utilities (retained for diagnostic use) |
| `bg_plotting.py` | Diagnostic light curve plots, DSS thumbnail stamps, and TESS followup panels |
| `bg_bins.py` | Light curve binning utilities |
| `bg_logger.py` | Logging setup (rotating file handler + console) |
| `bg_tess_followup.py` | TESS light curve retrieval via Lightkurve / MAST |
| `bg_ml.ipynb` | K-Means clustering on `lc_flagging` features; elbow, silhouette, PCA, and scatter plots |

## Output files

| File | Description |
|---|---|
| `ml_features.csv` | ML feature table — one row per target per filter |
| `analysis/{target}/{target}_{filter}_flags_flagged.png` | Flagged light curve plot with dips highlighted |
| `logs/bg_pipeline.log` | Full pipeline run log |
| `outputs/cluster_centers.csv` | K-Means cluster centre values in original feature space |
| `outputs/elbow_method.png` | Inertia vs K (elbow method) |
| `outputs/silhouette_scores.png` | Silhouette score vs K |
| `outputs/cluster_scatter.png` | 2-D cluster scatter (first two features) |
| `outputs/pca_clusters.png` | PCA-compressed 2-D cluster plot |
| `outputs/scaled_hist.png` | Per-feature histograms of scaled values |

## ML features (from `bg_lc_flagging`)

| Feature | Description |
|---|---|
| `score` | Overall variability score (matched-filter SNR × run-length bonus) |
| `n_dips` | Number of detected dip events |
| `sigma` | Robust MAD-based flux scatter |
| `survival_fraction` | Fraction of points surviving sigma-clipping |
| `n_high_cut` | Number of points cut above the upper threshold |
| `p2p_over_mad` | Point-to-point scatter normalised by MAD |
| `duty_cycle` | Fraction of time span spent in dips |
| `spacing_frac_scatter` | Fractional scatter of inter-dip spacings (low → periodic) |
| `consistent_spacings` | 1 if dip spacings are consistent (periodic), else 0 |
| `best_depth_flux` | Flux depth of the highest-SNR dip |
| `best_depth_frac` | Fractional depth of the highest-SNR dip |
| `best_depth_sigma` | Depth in units of sigma for the highest-SNR dip |
| `best_mf_snr` | Matched-filter SNR of the best dip |
| `best_duration` | Duration (days) of the best dip |
| `best_n_points` | Number of in-dip data points for the best dip |
