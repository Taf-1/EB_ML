import os 
from astropy.io import fits
import numpy as np
from tqdm import tqdm
from astropy.table import Table, vstack
import logging

class Stack_LCS:
    def __init__(self, logger: logging.Logger, variable_table: str, data_root: str):
        self.var_table = variable_table
        self.logger = logger
        self.data_root = data_root

    def stack_light_curves(self) -> None:
        self.logger.info("Starting to stack light curves from variable table")
        lc_folder = f"{self.data_root}/lightcurves/"
        if not os.path.exists(lc_folder):
            self.logger.info(f"Creating directory for light curve files: {lc_folder}")
            os.mkdir(lc_folder)
        self.logger.info(f"Reading variable table from {self.var_table}")
        with fits.open(self.var_table) as hdul:
            wd_table = Table(hdul[1].data)
        gaia_ids = np.unique(wd_table["source_id"])
        tess = []
        self.logger.info(f"Found {len(gaia_ids)} unique Gaia IDs in {self.var_table}.")
        for gid in tqdm(gaia_ids, desc=f"Looping over all WDs in {self.var_table}", total=len(gaia_ids)):
            self.logger.info(f"Processing Gaia ID {gid}")
            mask = (wd_table['source_id'] == gid) & (wd_table['QC_FLAG'] == 'green')
            self.logger.info(f"Found {np.sum(mask)} entries for Gaia ID {gid} with QC_FLAG 'green'")
            subset = wd_table[mask]
            if len(subset) > 100:
                self.logger.info(f"Saving light curve for Gaia ID {gid} with {len(subset)} data points")
                output_file = os.path.join(lc_folder, f"Gaia_DR3_{gid}_LC.fits")
                subset.write(output_file, format="fits", overwrite=True)
            else:
                self.logger.info(f"Skipping Gaia ID {gid} due to insufficient data points (found {len(subset)})")
                subset = wd_table[wd_table['source_id'] == gid]
                tess.append(subset[0])
                self.logger.info(f"Added Gaia ID {gid} to TESS follow-up list with {len(subset)} total entries (QC_FLAG not green)")
                continue
        self.logger.info(f"Finished processing all Gaia IDs. Now stacking TESS follow-up candidates into a single table.")
        combined_table = vstack(tess)
        hdu = fits.BinTableHDU(combined_table)
        hdu.writeto(os.path.join(lc_folder, "TESS_followup_candidates.fits"), overwrite=True)
        self.logger.info(f"Saved TESS follow-up candidates to {os.path.join(lc_folder, 'TESS_followup_candidates.fits')}")