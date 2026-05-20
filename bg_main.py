import bg_query
import bg_tess_followup
import bg_stack_lcs
import bg_bins
import bg_analysis
import bg_logger
import bg_plotting
import argparse
import configparser
import os
from astropy.io import fits
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.skyview import SkyView
import astropy.units as u
import pandas as pd

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
    bg_detections_table = cfg["bg_detections_table"]
    bg_images_table = cfg["bg_images_table"]
    wd_table = cfg["wd_table"]
    hr_table = cfg["hr_table"]
    logger = bg_logger.BG_logging(stage_name="BG_pipeline", log_file=os.path.join(data_root, "logs/bg_pipeline.log")).setup_logger()
    bg_query.Google_Cloud_query(logger, variable_table_loc, project_id, os.path.join(data_root, "variable_table.fits")).run_query()
    bg_stack_lcs.Stack_LCS(logger, os.path.join(data_root, "variable_table.fits")).stack_light_curves()
    wd_files = [os.path.join(data_root, "lightcurves", f) for f in os.listdir(os.path.join(data_root, "lightcurves")) if f.endswith("_LC.fits")]
    per_wd_results = []
    for wd_file in wd_files:
        logger.info(f"Processing light curve file: {wd_file}")
        wd_name = os.path.basename(wd_file).replace("_LC.fits", "")
        wd_dir = os.path.join(data_root, "analysis", wd_name)
        logger.info(f"Creating directory for analysis of {wd_name}: {wd_dir}")
        if not os.path.exists(wd_dir):
            logger.info(f"Directory {wd_dir} does not exist. Creating it now.")
            os.makedirs(wd_dir)
        os.chdir(wd_dir)
        logger.info(f"Reading light curve data from {wd_file}")
        with fits.open(wd_file) as hdul:
            lc_table = hdul[1].data
        filters = np.unique(lc_table["FILTER"])
        gaia_id = lc_table["source_id"][0]
        ra_deg = lc_table["ra"][0]
        dec_deg = lc_table["dec"][0]
        logger.info(f"Found Gaia ID {gaia_id} with RA {ra_deg} and Dec {dec_deg} in {wd_file}")
        gcs_files = bg_query.BG_images(logger, bg_detections_table, bg_images_table, gaia_id, filters[0]).query_bg_database()
        gcs_files = gcs_files[:5]
        logger.info("Extracted the first 5 gcs files from the BlackGEM google cloud")
        logger.info(f"Querying the sky coordinates: RA - {ra_deg} DEC - {dec_deg}")
        target_sky_coords = SkyCoord(ra=ra_deg, dec=dec_deg, frame='icrs', unit="deg")
        logger.info("Queried the coordinates - now using SkyView to get the DSS images for thumbnail")
        dss_image = SkyView.get_images(position=target_sky_coords, survey=['DSS2 Red'], radius=50*u.arcsec)
        if len(dss_image) == 0:
            logger.info("No images were found for DSS red - trying DSS instead")
            dss_image = SkyView.get_images(position=target_sky_coords, survey=['DSS'], radius=50*u.arcsec)
        logger.info("DSS images collected - transforming to data to be used later on ...")
        dss_data = dss_image[0][0].data if len(dss_image) > 0 else None
        per_filter_data = {}
        for filt in filters:
            logger.info(f"Processing filter {filt} for Gaia ID {gaia_id}")
            mask = lc_table["FILTER"] == filt
            time = lc_table["MJD_OBS"][mask]
            flux = lc_table["FNU_OPT"][mask]
            logger.info(f"Extracted {len(time)} data points for filter {filt} of Gaia ID {gaia_id}")
            logger.info("Normalizing the flux ...")
            ratio = flux / np.median(flux)
            ratio = ratio[~np.isnan(ratio)]
            ratio_norm = ratio / np.median(ratio)
            logger.info(f"Flux has been normalized - max flux - {max(ratio_norm)} min flux - {min(ratio_norm)}")
            logger.info("Ready to conduct the period search using BLS and LS algorithms")
            results_filename = f"{wd_name}_{filt}_res.txt"
            analysis_initialise = bg_analysis.BG_analysis(logger=logger, time=time, flux=ratio_norm, filename=results_filename)
            logger.info("Initialised the BG_analysis - first stage: Running the BLS period search")
            bls_params = analysis_initialise.run_bls()
            logger.info(f"BLS period search complete - BLS params dict includes - {bls_params}")
            logger.info("Second stage: Running the Lomb Scargle period search")
            ls_params = analysis_initialise.run_lomb_scargle()
            logger.info(f"Lomb Scargle period search complete - LS params dict includes - {ls_params}")
            BG_output_filename = f"{wd_name}_BG.png"
            TESS_output_filename = f"{wd_name}_TESS.png"
            logger.info(f"Period search completed - creating the diagnostic plots - {BG_output_filename}")
            bg_plotting.diagnostic_plotting(logger=logger, gcs_files=gcs_files, ra_deg=ra_deg, dec_deg=dec_deg,
                        time=time, flux=ratio_norm, dss_data=dss_data, bls_params=bls_params,
                        ls_params=ls_params, nic_data=wd_table, gaia_id=gaia_id, 
                        output_filename=BG_output_filename, telescope='BG', hr_table=hr_table).bg_diagnostic_plot()
            """Will add TESS followup later on - for now just create the BG diagnostic plot and save the BLS and LS params to a csv file for ML"""
            """logger.info(f"Period search completed - creating the diagnostic plots - {TESS_output_filename}")
            bg_plotting.diagnostic_plotting(logger=logger, gcs_files=gcs_files, ra_deg=ra_deg, dec_deg=dec_deg,
                        time=time, flux=ratio_norm, dss_data=dss_data, bls_params=bls_params,
                        ls_params=ls_params, nic_data=wd_table, gaia_id=gaia_id, 
                        output_filename=TESS_output_filename, telescope='TESS', hr_table=hr_table).bg_diagnostic_plot()"""
            logger.info(f"Diagnostic plots created for filter {filt} of Gaia ID {gaia_id} - now saving the BLS and LS params to a dictionary for ML")
            per_filter_data[filt] = {
                "bls_params": bls_params,
                "ls_params": ls_params,
                "gaia_id": gaia_id,
                "filter": filt
            }
        logger.info(f"Completed processing for Gaia ID {gaia_id} - now appending the per filter data to the per WD results list")
        per_wd_results.append(per_filter_data)
        """APPEND THE BLS_PARAMS, LS_PARAMS, GAIA_ID AND FILTER then save the total list to a csv file for ML"""
    """save out the per_wd_results list to a csv file for ML"""
    per_wd_results_filename = os.path.join(data_root, "per_wd_results.csv")
    logger.info(f"Saving the per WD results to {per_wd_results_filename}")
    per_wd_results_df = pd.DataFrame(per_wd_results)
    per_wd_results_df.to_csv(per_wd_results_filename, index=False)
    logger.info(f"Saved the per WD results to {per_wd_results_filename} - BG pipeline complete")

if __name__ == "__main__":
    main()