import numpy as np 
from lightkurve import search_lightcurvefile
import logging
from tqdm import tqdm

class TESS_Followup:
    def __init__(self, logger: logging.Logger, gaia_id: int):
        self.logger = logger
        self.gaia_id = gaia_id

    def get_combined_tess_lc(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        self.logger.info(f"Searching for TESS light curves for Gaia ID {self.gaia_id}")
        search_result = search_lightcurvefile(f'Gaia DR3 {self.gaia_id}', mission='TESS')
        if len(search_result) == 0:
            self.logger.info(f"No TESS data for Gaia ID {self.gaia_id}")
            return None
        all_time, all_flux, all_flux_err = [], [], []
        for lc_file in tqdm(search_result, desc="Downloading TESS light curves", total=len(search_result)):
            lc = lc_file.download()
            if lc is None:
                continue
            flux = lc.pdcsap_flux
            flux_err = lc.pdcsap_flux_err if lc.pdcsap_flux_err is not None else np.zeros(len(flux))
            time = lc.time.value
            if hasattr(flux, "filled"):
                flux = flux.filled(np.nan)
            if hasattr(flux_err, "filled"):
                flux_err = flux_err.filled(np.nan)
            mask = np.isfinite(time) & np.isfinite(flux)
            all_time.append(time[mask])
            all_flux.append(flux[mask])
            all_flux_err.append(flux_err[mask])
        if len(all_time) == 0:
            return None
        self.logger.info(f"Combining {len(all_time)} TESS light curves for Gaia ID {self.gaia_id}")
        time_combined = np.concatenate(all_time)
        flux_combined = np.concatenate(all_flux)
        flux_err_combined = np.concatenate(all_flux_err)
        flux_combined /= np.nanmedian(flux_combined)
        self.logger.info(f"Combined light curve has {len(time_combined)} data points for Gaia ID {self.gaia_id}")
        return time_combined, flux_combined, flux_err_combined