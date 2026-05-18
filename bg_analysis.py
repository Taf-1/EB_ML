from astropy.timeseries import LombScargle, BoxLeastSquares
import logging
import numpy as np
from scipy.signal import lombscargle

class BG_analysis:
    def __init__(self, logger: logging.Logger, time: np.ndarray, flux: np.ndarray, filename: str):
        self.logger = logger
        self.time = time
        self.flux = flux
        self.filename = filename

    def run_bls(self) -> dict:
        self.logger.info("Performing Box Least Squares analysis on the light curve")
        bls = BoxLeastSquares(self.time, self.flux)
        period_grid = np.linspace(0.005, 5, 40000)
        duration_grid = np.array([0.0025])
        self.logger.info(f"Using period grid from {period_grid[0]} to {period_grid[-1]} with {len(period_grid)} points")
        self.logger.info(f"Using duration grid: {duration_grid}")
        results_b = bls.power(period_grid, duration_grid, oversample=10)
        self.logger.info("Box Least Squares analysis completed")
        index_b = np.argmax(results_b.power)
        peak_b = results_b.power[index_b]
        period_b = results_b.period[index_b]
        t0_b = results_b.transit_time[index_b]
        depth_b = results_b.depth[index_b]
        duration_b = results_b.duration[index_b]
        self.logger.info(f"Best BLS period: {period_b:.5f} days with power {peak_b:.5f}")
        self.logger.info(f"Transit time (t0): {t0_b:.5f} days, Depth: {depth_b:.5f}, Duration: {duration_b:.5f} days")
        value = (peak_b - np.average(results_b.power)) / np.std(results_b.power)
        sde_b = round(float(value), 3)
        self.logger.info(f"Calculated SDE for BLS: {sde_b:.3f}")
        with open(self.filename, 'w') as rf:
                rf.write("#BLS Results:")
                rf.write("#Index SDE Peak T0 Period Depth Duration\n")
                rf.write("{:} {:} {:} {:} {:} {:} {:}\n".format(index_b, sde_b, peak_b, t0_b, period_b, depth_b, duration_b))
        self.logger.info(f"Saved BLS results to {self.filename}")
        self.logger.info("Calculating phase-folded time for BLS period")
        t_fold_b = ((self.time - t0_b) % period_b) / period_b
        loc_b = np.where(t_fold_b > 0.5)[0]
        t_fold_b[loc_b] -= 1
        self.logger.info("Calculating phase-folded time for twice the BLS period")
        t_fold_b_twice = ((self.time - t0_b) % (period_b * 2)) / (period_b * 2)
        loc_b_twice = np.where(t_fold_b_twice > 0.5)[0]
        t_fold_b_twice[loc_b_twice] -= 1
        self.logger.info("Box Least Squares analysis and phase folding completed")
        bls_params = {  
            "results_b": results_b,
            "sde_b": sde_b,
            "period_b": period_b,
            "t_fold_b": t_fold_b,
            "t_fold_b_twice": t_fold_b_twice,
            "time": self.time,
            "flux": self.flux
        }
        self.logger.info("BLS parameters calculated and stored in dictionary")
        return bls_params

    def run_lomb_scargle(self) -> dict:
        self.logger.info("Performing Lomb-Scargle analysis on the light curve")
        ls_max_freq = 1.0/(5.0/60/24)
        ls_min_freq = 1.0/((max(self.time) - min(self.time)))
        self.logger.info(f"Using Lomb-Scargle frequency range from {ls_min_freq:.5f} to {ls_max_freq:.5f} cycles/day")
        ls1 = LombScargle(self.time, self.flux)
        self.logger.infio("Running Lomb-Scargle with nterms=1") 
        try:
            frequency_l1, power_l1 = ls1.autopower(minimum_frequency=ls_min_freq,
                                                maximum_frequency=ls_max_freq,
                                                samples_per_peak=10)
        except np.linalg.LinAlgError as e:
            if 'Singular matrix' in str(e):
                self.logger.info("This is a singular matrix")
                self.logger.info("Computing Lomb-Scargle frequency and power manually for nterms=1")
                frequency_l1 = 2 * np.pi * np.linspace(ls_min_freq, ls_max_freq, 1000)
                power_l1 = lombscargle(self.time, self.flux, frequency_l1)
                power_l1 /= np.max(power_l1)
        self.logger.info("Lomb-Scargle analysis with nterms=1 completed")       
        index_l1 = np.argmax(power_l1)
        power_max_l1 = power_l1[index_l1]
        frequency_max_l1 = frequency_l1[index_l1]
        period_l1 = 1.0/frequency_max_l1
        t0_l1 = np.min(self.time)
        self.logger.info(f"Best LS_1 period: {period_l1:.5f} days with power {power_max_l1:.5f}")
        ls2 = LombScargle(self.time, self.flux, nterms=2)
        self.logger.info("Running Lomb-Scargle with nterms=2")
        try:
            frequency_l2, power_l2 = ls2.autopower(minimum_frequency=ls_min_freq,
                                                maximum_frequency=ls_max_freq,
                                                samples_per_peak=10)
        except np.linalg.LinAlgError as e:
            if 'Singular matrix' in str(e):
                self.logger.info("This is a singular matrix")
                self.logger.info("Computing Lomb-Scargle frequency and power manually for nterms=2")
                frequency_l2 = 2 * np.pi * np.linspace(ls_min_freq, ls_max_freq, 1000)
                power_l2 = lombscargle(self.time, self.flux, frequency_l2)
                power_l2 /= np.max(power_l2)
        self.logger.info("Lomb-Scargle analysis with nterms=2 completed")
        index_l2 = np.argmax(power_l2)
        power_max_l2 = power_l2[index_l2]
        frequency_max_l2 = frequency_l2[index_l2]
        period_l2 = 1.0/frequency_max_l2
        t0_l2 = np.min(self.time)
        self.logger.info(f"Best LS_2 period: {period_l2:.5f} days with power {power_max_l2:.5f}")
        probabilities = [0.1, 0.01, 0.001]
        prob_lines = ["dotted", "dashdot", "dashed"]
        ls1_faps = ls1.false_alarm_level(probabilities, method='bootstrap')
        self.logger.info("Calculated false alarm probabilities for LS_1")
        with open(self.filename, 'a') as rf:
            self.logger.info(f"Saving LS results to {self.filename}")
            rf.write("#LS_1 Results:")
            rf.write("#Index Power_max Freq_max Period T0\n")
            rf.write("{:} {:} {:} {:} {:}\n".format(index_l1, power_max_l1, frequency_max_l1, period_l1, t0_l1))
            rf.write("#LS_2 Results:")
            rf.write("#Index Power_max Freq_max Period T0\n")
            rf.write("{:} {:} {:} {:} {:}\n".format(index_l2, power_max_l2, frequency_max_l2, period_l2, t0_l2))
        self.logger.info("LS results saved to file")
        self.logger.info("Calculating phase-folded time for LS_1 period")  
        t_fold_l1 = ((self.time - t0_l1) % period_l1) / period_l1
        t_fold_l2 = ((self.time - t0_l2) % period_l2) / period_l2
        self.logger.info("Calculating phase-folded time for twice the LS_1 and LS_2 periods")
        t_fold_l1_twice = ((self.time - t0_l1 ) % ( period_l1 * 2 )) / ( period_l1 * 2 )
        t_fold_l2_twice = ((self.time - t0_l2 ) % ( period_l2 * 2 )) / ( period_l2 * 2 )
        self.logger.info("Lomb-Scargle analysis and phase folding completed")
        ls_params = {
            "frequency_l1": frequency_l1,
            "frequency_max_l1": frequency_max_l1,
            "power_l1": power_l1,
            "frequency_l2": frequency_l2,
            "frequency_max_l2": frequency_max_l2,
            "power_l2": power_l2,
            "period_l1": period_l1,
            "t_fold_l1": t_fold_l1,
            "t_fold_l1_twice": t_fold_l1_twice,
            "period_l2": period_l2,
            "t_fold_l2": t_fold_l2,
            "t_fold_l2_twice": t_fold_l2_twice,
            "ls1_faps": ls1_faps,
            "probabilities": probabilities,
            "prob_lines": prob_lines,
            "time": self.time,
            "flux": self.flux
        }
        self.logger.info("LS parameters calculated and stored in dictionary")
        return ls_params
