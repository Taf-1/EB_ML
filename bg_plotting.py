import logging 
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval, ImageNormalize, LogStretch, MinMaxInterval
from astroquery.mast import Observations
from lightkurve import TessTargetPixelFile
from astropy import units as u
from tqdm import tqdm
import numpy as np
import time
from photutils.aperture import CircularAperture
import gc
from bg_bins import binning
from astropy.io.votable import parse_single_table
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
    "lines.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
})

class thumbnails:
    def __init__(self, logger: logging.Logger, gcs_files: list, ra: float, dec: float, ax):
        self.gcs_files = gcs_files
        self.logger = logger
        self.ra = ra 
        self.dec = dec 
        self.ax = ax

    def bg_thumbnail(self, size=50, stretch="zscale", cmap="viridis", title=True, max_retries=3, retry_delay=5) -> None:
        self.logger.info(f"Starting thumbnail generation for RA={self.ra}, Dec={self.dec} with {len(self.gcs_files)} GCS files.")
        self.logger.info(f"Processing {len(self.gcs_files)} GCS files for thumbnail generation.")
        coord = SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg)
        thumbs, norms = [], []
        for gcs in tqdm(self.gcs_files, desc="Looping over GCS files", total=len(self.gcs_files)):
            for attempt in range(max_retries):
                try:
                    self.logger.info(f"Attempting to read GCS file: {gcs} (attempt {attempt+1}/{max_retries})")
                    with fits.open(gcs, use_fsspec=True, memmap=False) as hdul:
                        hdu = hdul[-1]
                        wcs = WCS(hdu.header)
                        data = hdu.data.copy()
                    break 
                except (IndexError, OSError) as e:
                    self.logger.info(f"GCS read failed on {gcs} (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        self.logger.info(f"Retrying {gcs} after {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        self.logger.info(f"Skipping {gcs} after {max_retries} failed attempts.")
                        data, wcs = None, None
            self.logger.info(f"Finished processing GCS file: {gcs}")
            if data is None:
                self.logger.info(f"Skipping {gcs} due to read failure.")
                continue
            self.logger.info(f"Successfully read GCS file: {gcs}. Generating thumbnail...")
            x, y = wcs.world_to_pixel(coord)
            cutout = Cutout2D(data, position=(x, y), size=size, wcs=wcs)
            thumb = np.rot90(cutout.data, k=3)
            self.logger.info(f"Generated thumbnail for {gcs}. Applying stretch: {stretch}...")
            if stretch == "zscale":
                self.logger.info(f"Applying ZScaleInterval stretch to thumbnail from {gcs}.")
                vmin, vmax = ZScaleInterval().get_limits(thumb)
                norm = ImageNormalize(vmin=vmin, vmax=vmax)
            elif stretch == "log":
                self.logger.info(f"Applying LogStretch to thumbnail from {gcs}.")
                norm = ImageNormalize(stretch=LogStretch())
            else:
                self.logger.info(f"Applying MinMaxInterval stretch to thumbnail from {gcs}.")
                vmin, vmax = MinMaxInterval().get_limits(thumb)
                norm = ImageNormalize(vmin=vmin, vmax=vmax)
            self.logger.info(f"Finished processing thumbnail for {gcs}. Adding to list.")
            thumbs.append(thumb)
            norms.append(norm)
        if len(thumbs) == 0:
            self.logger.info("No thumbnails could be loaded.")
            return None
        self.logger.info("Creating the thumbnail plot...")
        self.ax.axis("off")
        if title:
            self.ax.set_title(f"RA={self.ra:.4f}  Dec={self.dec:.4f}", fontsize=9)
        if len(thumbs) == 1:
            self.logger.info("Only one thumbnail generated. Displaying without animation.")
            self.ax.imshow(thumbs[0], origin="lower", cmap=cmap, norm=norms[0])
            return None
        self.ax.imshow(thumbs[0], origin="lower", cmap=cmap, norm=norms[0])

    def tess_thumbnail(self, size=50, sector=None, cmap="viridis", title=True) -> None:
        self.logger.info(f"Starting TESS thumbnail generation for RA={self.ra}, Dec={self.dec} in sector {sector}.")
        coord = SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg)
        self.logger.info(f"Querying MAST for TESS observations at RA={self.ra}, Dec={self.dec} with a radius of 0.02 deg.")
        obs = Observations.query_criteria(
            obs_collection="TESS",
            dataproduct_type="timeseries",
            coordinates=coord,
            radius="0.02 deg"
        )
        if len(obs) == 0:
            self.logger.info("No TESS observations found for the given coordinates.")
            return None
        self.logger.info(f"Found {len(obs)} TESS observations. Filtering for Target Pixel Files (TPFs)...")
        products = Observations.get_product_list(obs)
        tpfs = products[products["productSubGroupDescription"] == "TP"]
        if len(tpfs) == 0:
            self.logger.info("No Target Pixel Files (TPFs) found for the given coordinates.")
            return None
        self.logger.info(f"Found {len(tpfs)} TP files across sectors:")
        if sector is not None:
            self.logger.info(f"Filtering TP files for sector {sector}...")
            mask = [f"s{sector:04d}" in uri for uri in tpfs["dataURI"]]
            tpfs = tpfs[mask]
            if len(tpfs) == 0:
                self.logger.info(f"No TP found for sector {sector}.")
                return None
        self.logger.info(f"Downloading the first TP file for thumbnail generation...")
        manifest = Observations.download_products(tpfs[:1])
        local_path = manifest["Local Path"][0]
        self.logger.info(f"Downloaded TP file to {local_path}. Generating thumbnail...")
        tpf = TessTargetPixelFile(local_path)
        median_image = np.nanmedian(tpf.flux.value, axis=0)
        cutout = Cutout2D(median_image, position=coord, size=size, wcs=tpf.wcs)
        if len(cutout.data) == 0:
            self.logger.info("No thumbnails could be loaded.")
            return None
        self.logger.info("Creating the thumbnail plot...")
        self.ax.axis("off")
        if title:
            self.ax.set_title(f"RA={self.ra:.4f}  Dec={self.dec:.4f}", fontsize=9)
        if len(cutout.data) == 1:
            self.logger.info("Only one thumbnail generated. Displaying without animation.")
            self.ax.imshow(cutout.data[0], origin="lower", cmap=cmap)
            return None
        self.ax.imshow(cutout.data[0], origin="lower", cmap=cmap)

class diagnostic_plotting:
    def __init__(self, logger: logging.Logger, gcs_files: list, ra_deg: float, dec_deg: float, 
                time: np.ndarray, flux: np.ndarray, dss_data: np.ndarray, bls_params: dict, 
                ls_params: dict, nic_data, gaia_id: int, output_filename: str, telescope: str,
                hr_table: str):
        self.logger = logger
        self.gcs_files = gcs_files
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.time = time
        self.flux = flux
        self.dss_data = dss_data
        self.bls_params = bls_params
        self.ls_params = ls_params
        self.nic_data = nic_data
        self.gaia_id = gaia_id
        self.output_filename = output_filename
        self.telescope = telescope
        self.hr_table = hr_table

    def bg_diagnostic_plot(self, bin_fact=10) -> None:
        self.logger.info(f"Starting diagnostic plot generation for Gaia ID {self.gaia_id} at RA={self.ra_deg}, Dec={self.dec_deg}.")
        target_location = np.where(self.nic_data['dr2_source_id'] == self.gaia_id)[0]
        if len(target_location) >= 1:
            MG = self.nic_data['absG'][target_location]
            bprp = self.nic_data['bp_rp'][target_location]
            g_apparent = self.nic_data["phot_g_mean_mag"][target_location][0]
        else:
            MG, bprp, g_apparent = None, None, None
        self.logger.info("Setting up the diagnostic plot layout...")
        layout = """
                AAAB
                CDEF
                GHIJ
                KLMO
                """
        fig, ax = plt.subplot_mosaic(layout, figsize=(15, 10))
        self.logger.info("Plotting the raw light curve...")
        ax['A'].plot(self.time-np.min(self.time), self.flux, 'k.', alpha=0.5, ms=1, label='Raw lc')
        ax['A'].set_xlabel('MJD - {:}'.format(int(np.min(self.time))))
        ax['A'].set_ylabel("Normalised Flux")
        ax['A'].legend()
        if len(target_location) >= 1:
            ax['A'].set_title("GAIA DR3 ID: {:} ra: {:} dec: {:} Gaia mag: {:.6f}".format(self.gaia_id, self.ra_deg, self.dec_deg, g_apparent))
        else:
            ax['A'].set_title("GAIA DR3 ID: {:} ra: {:} dec: {:}".format(self.gaia_id, self.ra_deg, self.dec_deg))
        self.logger.info("Generating the thumbnail from GCS files...")
        thumbnails(self.logger, self.gcs_files, self.ra_deg, self.dec_deg, ax['B']).bg_thumbnail()
        self.logger.info("Plotting the BLS power spectrum...")
        ax['C'].set_title("P={:.5f} SDE={:.2f} 2P={:.5f}".format(self.bls_params["period_b"], self.bls_params["sde_b"], self.bls_params["period_b"]*2))
        ax['C'].plot(self.bls_params["results_b"].period, self.bls_params["results_b"].power, 'k-', lw=1, label='BLS')
        ax['C'].axvline(self.bls_params["period_b"], lw=2, ls="--", color="red", zorder=0)
        ax['C'].set_xlabel("Period (d)")
        ax['C'].set_ylabel("Power")
        ax['C'].legend()
        self.logger.info("Plotting the Lomb-Scargle 1 power spectra...") 
        ax['D'].set_title("Freq={:.5f} P={:.5f} 2P={:.5f}".format(self.ls_params["frequency_max_l1"], self.ls_params["period_l1"], self.ls_params["period_l1"]*2))
        ax['D'].semilogx(1.0/self.ls_params["frequency_l1"], self.ls_params["power_l1"], 'k-', lw=1, label='LS_1')
        ax['D'].axvline(1.0/self.ls_params["frequency_max_l1"], lw=2, ls="--", color="red", zorder=0)
        for f, l in zip(self.ls_params["ls1_faps"], self.ls_params["prob_lines"]):
            ax['D'].axhline(f, lw=1, ls=l, color="grey")
        ax['D'].set_xlabel("Period (d)")
        ax['D'].legend()
        self.logger.info("Plotting the Lomb-Scargle 2 power spectra...")
        ax['E'].set_title("Freq={:.5f} P={:.5f} 2P={:.5f}".format(self.ls_params["frequency_max_l2"], self.ls_params["period_l2"], self.ls_params["period_l2"]*2))
        ax['E'].semilogx(1.0/self.ls_params["frequency_l2"], self.ls_params["power_l2"], 'k-', lw=1, label='LS_2')
        ax['E'].axvline(1.0/self.ls_params["frequency_max_l2"], lw=2, ls="--", color="red", zorder=0)
        ax['E'].set_xlabel("Period (d)")
        ax['E'].legend()
        self.logger.info("Plotting the DSS image...")
        if len(self.dss_data) <= 1:
            ax['F'].axis('off')
        else:
            self.logger.info(f"DSS data shape: {self.dss_data.shape}. Displaying the image with appropriate scaling.")
            dss_median_value = np.median(self.dss_data)
            dss_vmin = dss_median_value - 0.5 * dss_median_value
            dss_vmax = dss_median_value + 0.5 * dss_median_value
            im_dss = ax['F'].imshow(self.dss_data, vmin=dss_vmin, vmax=dss_vmax, cmap='viridis', origin='lower')
            self.logger.info("DSS image displayed. Adding colorbar and aperture overlay.")
            cbar = fig.colorbar(im_dss, ax=ax['F'], label='Number of Pixels')
            aperture = CircularAperture([self.dss_data.shape[1]/2, self.dss_data.shape[0]/2], 3*25/4)
            aperture.plot(ax=ax['F'], color='r')
            self.logger.info("Aperture overlay added. Finalizing DSS image plot.")
            ax['F'].set_xlabel('X pixels')
            ax['F'].set_ylabel('Y pixels')
            ax['F'].invert_yaxis()
            ax['F'].set_title('DSS Image')
        self.logger.info("Plotting the folded light curves...")
        ax['G'].plot(self.bls_params["t_fold_b"], self.flux, 'k.', alpha=0.5, ms=1, label='Phased BLS')
        if self.telescope == "TESS":
            self.logger.info("Binning the folded light curve for better visualization...")
            binning_instance = binning(self.logger, self.bls_params["t_fold_b"], self.flux, bin_fact=bin_fact)
            t_fold_bin_b, ratio_norm_c_bin_b = binning_instance.bin_data_on_phase()
            ax['G'].plot(t_fold_bin_b, ratio_norm_c_bin_b, 'r.', alpha=0.5, ms=2, label='Phased BLS (bin)')
        ax['G'].set_xlim(-0.5, 0.5)
        ax['G'].set_xlabel("Phase")
        ax['G'].set_ylabel("Normalised Flux")
        ax['G'].legend()
        self.logger.info("Plotting the folded light curve for Lomb-Scargle 1 period...")
        ax['H'].plot(self.ls_params["t_fold_l1"], self.flux, 'k.', alpha=0.5, ms=1, label='Phased L1')
        if self.telescope == "TESS":
            self.logger.info("Binning the folded light curve for Lomb-Scargle 1 period...")
            binning_instance = binning(self.logger, self.ls_params["t_fold_l1"], self.flux, bin_fact=bin_fact)
            t_fold_bin_l1, ratio_norm_c_bin_l1 = binning_instance.bin_data_on_phase()
            ax['H'].plot(t_fold_bin_l1, ratio_norm_c_bin_l1, 'r.', alpha=0.5, ms=2, label='Phased L1 (bin)')
        ax['H'].set_xlim(0.0, 1.0)
        ax['H'].set_xlabel("Phase")
        ax['H'].legend()
        self.logger.info("Plotting the folded light curve for Lomb-Scargle 2 period...")
        ax['I'].plot(self.ls_params["t_fold_l2"], self.flux, 'k.', alpha=0.5, ms=1, label='Phased L2')
        if self.telescope == "TESS":
            self.logger.info("Binning the folded light curve for Lomb-Scargle 2 period...")
            binning_instance = binning(self.logger, self.ls_params["t_fold_l2"], self.flux, bin_fact=bin_fact)
            t_fold_bin_l2, ratio_norm_c_bin_l2 = binning_instance.bin_data_on_phase()
            ax['I'].plot(t_fold_bin_l2, ratio_norm_c_bin_l2, 'r.', alpha=0.5, ms=2, label='Phased L2 (bin)')
        ax['I'].set_xlim(0.0, 1.0)
        ax['I'].set_xlabel("Phase")
        ax['I'].legend()
        self.logger.info("Plotting the Gaia HR diagram...")
        if MG is not None and bprp is not None:
            ax['J'].plot(bprp,MG,'or',markersize=3,zorder=2)
        table = parse_single_table(self.hr_table)
        s_MG = 5 + 5*np.log10(table.array['parallax']/1000) + table.array['phot_g_mean_mag']
        s_bprp = table.array['bp_rp']
        ax['J'].scatter(s_bprp,s_MG,c='grey', s=0.5, zorder=0)
        ax['J'].invert_yaxis()
        ax['J'].set_title('$Gaia$ HR-diagram')
        ax['J'].set_ylabel('$M_G$')
        ax['J'].set_xlabel('$G_{BP}-G_{RP}$')
        self.logger.info("Plotting the folded light curve for twice the BLS period...")
        ax['K'].plot(self.bls_params["t_fold_b_twice"], self.flux, 'k.', alpha=0.5, ms=1, label='Phased BLS')
        if self.telescope == "TESS":
            self.logger.info("Binning the folded light curve for twice the BLS period...")
            binning_instance = binning(self.logger, self.bls_params["t_fold_b_twice"], self.flux, bin_fact=bin_fact)
            t_fold_bin_b_twice, ratio_norm_c_bin_b_twice = binning_instance.bin_data_on_phase()
            ax['K'].plot(t_fold_bin_b_twice, ratio_norm_c_bin_b_twice, 'r.', alpha=0.5, ms=2, label='Phased BLS (bin)')
        ax['K'].set_xlim(-0.5, 0.5)
        ax['K'].set_xlabel("Phase")
        ax['K'].set_ylabel("Normalised Flux")
        ax['K'].legend()
        self.logger.info("Plotting the folded light curve for twice the Lomb-Scargle 1 period...")
        ax['L'].plot(self.ls_params["t_fold_l1_twice"], self.flux, 'k.', alpha=0.5, ms=1, label='Phased L1')
        if self.telescope == "TESS":
            self.logger.info("Binning the folded light curve for twice the Lomb-Scargle 1 period...")
            binning_instance = binning(self.logger, self.ls_params["t_fold_l1_twice"], self.flux, bin_fact=bin_fact)
            t_fold_bin_l1_twice, ratio_norm_c_bin_l1_twice = binning_instance.bin_data_on_phase()
            ax['L'].plot(t_fold_bin_l1_twice, ratio_norm_c_bin_l1_twice, 'r.', alpha=0.5, ms=2, label='Phased L1 (bin)')
        ax['L'].set_xlim(0.0, 1.0)
        ax['L'].set_xlabel("Phase")
        ax['L'].legend()
        self.logger.info("Plotting the folded light curve for twice the Lomb-Scargle 2 period...")
        ax['M'].plot(self.ls_params["t_fold_l2_twice"], self.flux, 'k.', alpha=0.5, ms=1, label='Phased L2')
        if self.telescope == "TESS":
            self.logger.info("Binning the folded light curve for twice the Lomb-Scargle 2 period...")
            binning_instance = binning(self.logger, self.ls_params["t_fold_l2_twice"], self.flux, bin_fact=bin_fact)
            t_fold_bin_l2_twice, ratio_norm_c_bin_l2_twice = binning_instance.bin_data_on_phase()
            ax['M'].plot(t_fold_bin_l2_twice, ratio_norm_c_bin_l2_twice, 'r.', alpha=0.5, ms=2, label='Phased L2 (bin)')
        ax['M'].set_xlim(0.0, 1.0)
        ax['M'].set_xlabel("Phase")
        ax['M'].legend()
        self.logger.info("Plotting the WD cooling sequence from Nicola's catalog...")
        prob_of_wd = self.nic_data['Pwd']
        g_err = self.nic_data['phot_g_mean_mag_error_corrected']
        plx_over_err = self.nic_data['parallax_over_error'] 
        cutoff = (prob_of_wd > 0.75) & (plx_over_err > 10) & (g_err < 0.1)
        cut_data = self.nic_data[cutoff]
        n_MG = cut_data['absG']
        n_bprp = cut_data['bp_rp']
        cut_prob = cut_data['Pwd']
        self.logger.info(f"Plotting {len(cut_data)} points from the WD cooling sequence with a probability cutoff of 0.75.")
        sc = ax['O'].scatter(n_bprp, n_MG, c=cut_prob, cmap='viridis', s=0.5)
        plt.colorbar(sc, label='WD Probability')
        ax['O'].set_ylabel('$M_G$')
        ax['O'].set_xlabel('$G_{BP}-G_{RP}$')
        ax['O'].invert_yaxis()
        if MG is not None and bprp is not None:
            ax['O'].plot(bprp, MG, 'or', markersize=10, zorder=2)
        fig.tight_layout()
        fig.savefig(self.output_filename, dpi=300)
        fig.clf()
        plt.close()
        gc.collect(1)