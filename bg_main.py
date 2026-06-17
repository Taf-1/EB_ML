import bg_lc_flagging
import bg_logger
import argparse
import configparser
import os
from astropy.io import fits
import numpy as np
import pandas as pd
from tqdm import tqdm

def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BG pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    return parser.parse_args()

def load_config(config_path: str) -> dict[str, str]:
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    flat = {}
    for section in cfg.sections():
        for key, val in cfg.items(section):
            flat[key] = val
    return flat

def main():
    """Export info from the config file and run the pipeline"""
    args = arg_parse()
    cfg = load_config(args.config)
    data_root = cfg["data_root"]
    variable_table_loc = cfg["variable_table_loc"]
    project_id = cfg["project_id"]
    logger = bg_logger.BG_logging(stage_name="BG_pipeline", log_file=os.path.join(data_root, "logs/bg_pipeline.log")).setup_logger()
    """bg_query.Google_Cloud_query(logger, variable_table_loc, project_id, os.path.join(data_root, "variable_table.fits")).run_query()"""
    wd_files = [os.path.join(data_root, "lightcurves", f) for f in os.listdir(os.path.join(data_root, "lightcurves")) if f.endswith("_LC.fits")]
    wd_analysis_dir = f"{data_root}/analysis/"
    if not os.path.exists(wd_analysis_dir):
        logger.info(f"Creating directory for analysis results: {wd_analysis_dir}")
        os.mkdir(wd_analysis_dir)
    per_wd_results = []
    for wd_file in tqdm(wd_files, desc="Processing light curve files for each WD", total=len(wd_files)):
        logger.info(f"Processing light curve file: {wd_file}")
        wd_name = os.path.basename(wd_file).replace("_LC.fits", "")
        wd_dir = os.path.join(wd_analysis_dir, wd_name)
        logger.info(f"Creating directory for analysis of {wd_name}: {wd_dir}")
        if not os.path.exists(wd_dir):
            logger.info(f"Directory {wd_dir} does not exist. Creating it now.")
            os.mkdir(wd_dir)
        logger.info(f"Reading light curve data from {wd_file}")
        with fits.open(wd_file) as hdul:
            lc_table = hdul[1].data
        filters = np.unique(lc_table["FILTER"])
        gaia_id = lc_table["source_id"][0]
        ra_deg = lc_table["ra"][0]
        dec_deg = lc_table["dec"][0]
        logger.info(f"Found Gaia ID {gaia_id} with RA {ra_deg} and Dec {dec_deg} in {wd_file}")
        per_filter_data = {}
        for filt in filters:
            logger.info(f"Processing filter {filt} for Gaia ID {gaia_id}")
            mask = lc_table["FILTER"] == filt
            time = lc_table["MJD_OBS"][mask]
            flux = lc_table["FNU_OPT"][mask]
            logger.info(f"Extracted {len(time)} data points for filter {filt} of Gaia ID {gaia_id}")
            logger.info("Normalizing the flux ...")
            flux_err_raw = lc_table["FNU_OPT_ERR"][mask] if "FNU_OPT_ERR" in lc_table.names else np.zeros_like(flux)
            valid = np.isfinite(flux) & (flux > 0)
            time_v = time[valid]
            flux_v = flux[valid]
            flux_err_v = flux_err_raw[valid]
            med_flux = np.median(flux_v)
            flux_norm = flux_v / med_flux
            flux_err_norm = flux_err_v / med_flux
            exp_time = float(np.median(np.diff(np.sort(time_v)))) if len(time_v) > 1 else 1.0 / 1440.0
            logger.info(f"Flux normalized - max={max(flux_norm):.4f} min={min(flux_norm):.4f}, exp_time={exp_time:.6f} days")
            logger.info("Running light curve flagging (dip detection) ...")
            flag_filename = f"{wd_name}_{filt}_flags"
            flag_result = bg_lc_flagging.lc_flagging(
                logger=logger, time=time_v, exp_time=exp_time,
                flux=flux_norm, flux_err=flux_err_norm,
                filename=flag_filename, output_dir=wd_dir
            ).iterative_masking()
            logger.info(f"Flagging complete - n_dips={flag_result['n_dips']}, score={flag_result['score']:.3f}")
            logger.info(f"Flagging result for filter {filt} of Gaia ID {gaia_id} - now saving to dictionary for ML")
            per_filter_data[filt] = {
                "flag_result": flag_result,
                "gaia_id": gaia_id,
                "filter": filt
            }
        logger.info(f"Completed processing for Gaia ID {gaia_id} - now appending the per filter data to the per WD results list")
        per_wd_results.append(per_filter_data)
    rows = []
    for wd in per_wd_results:
        for filt, data in wd.items():
            res = data["flag_result"]
            quality = res["quality"]
            best_dip = max(res["dips"], key=lambda d: d["mf_snr"]) if res["dips"] else {}
            row = {
                "gaia_id": data["gaia_id"],
                "filter": filt,
                "baseline": res["baseline"],
                "sigma": res["sigma"],
                "n_dips": res["n_dips"],
                "score": res["score"],
                "survival_fraction": quality["survival_fraction"],
                "n_high_cut": quality["n_high_cut"],
                "p2p_over_mad": quality["p2p_over_mad"],
                "duty_cycle": quality["duty_cycle"],
                "spacing_frac_scatter": quality["spacing_frac_scatter"],
                "consistent_spacings": int(quality["consistent_spacings"]),
                "best_depth_flux": best_dip.get("depth_flux", np.nan),
                "best_depth_frac": best_dip.get("depth_frac", np.nan),
                "best_depth_sigma": best_dip.get("depth_sigma", np.nan),
                "best_mf_snr": best_dip.get("mf_snr", np.nan),
                "best_duration": best_dip.get("duration", np.nan),
                "best_n_points": best_dip.get("n_points", np.nan),
            }
            rows.append(row)
    ml_df = pd.DataFrame(rows)
    ml_features_filename = os.path.join(data_root, "ml_features.csv")
    logger.info(f"Saving ML-ready features DataFrame to {ml_features_filename}")
    ml_df.to_csv(ml_features_filename, index=False)
    logger.info(f"Saved {len(ml_df)} rows to {ml_features_filename} - BG pipeline complete")

if __name__ == "__main__":
    main()